use std::ffi::CStr;
use std::mem;

use ash::vk;
use indicatif::{ProgressBar, ProgressStyle};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::environment::EnvGpuData;
use crate::gpu_scene::build_gpu_scene;
use crate::scene::Scene;
use crate::vk_runtime::{AccelResource, BufferResource, ImageResource, VkContext};

pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub fov: f32,
    pub aa_samples: u32,
    pub max_depth: i32,
    pub reflection_depth: i32,
    pub refraction_depth: i32,
    pub tonemap: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct GpuParams {
    width: u32,
    height: u32,
    aa_samples: u32,
    max_depth: u32,
    reflection_depth: u32,
    refraction_depth: u32,
    light_count: u32,
    use_env: u32,
    use_sky: u32,
    env_width: u32,
    env_height: u32,
    tonemap: u32,
    exposure: f32,
    fov: f32,
    _pad0: u32,
    _pad1: u32,
}

pub fn render(scene: &Scene, config: &RenderConfig) -> Result<Vec<Vec3>, Box<dyn std::error::Error>> {
    let gpu_scene = build_gpu_scene(scene);
    let env = scene.environment.as_ref().map(|env| env.gpu_data());

    let pb = ProgressBar::new(7);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message("upload buffers");

    let ctx = VkContext::new()?;
    let device = &ctx.device;
    let command_buffer = ctx.command_buffer;
    let queue = ctx.queue;
    let accel_loader = &ctx.accel_loader;

    let vertices_buffer = ctx.create_buffer_with_data(
        &gpu_scene.vertices,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let normals_buffer = ctx.create_buffer_with_data(
        &gpu_scene.normals,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let triangles_buffer = ctx.create_buffer_with_data(
        &gpu_scene.triangles,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let indices_flat = gpu_scene
        .triangles
        .iter()
        .flat_map(|tri| [tri.v0, tri.v1, tri.v2])
        .collect::<Vec<u32>>();

    let indices_buffer = ctx.create_buffer_with_data(
        &indices_flat,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let tri_materials_buffer = ctx.create_buffer_with_data(
        &gpu_scene.tri_materials,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let materials_buffer = ctx.create_buffer_with_data(
        &gpu_scene.materials,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let lights_buffer = ctx.create_buffer_with_data(
        &gpu_scene.lights,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    pb.inc(1);
    pb.set_message("build acceleration");

    let (blas, tlas) = ctx.build_acceleration_structures(
        &vertices_buffer,
        &indices_buffer,
        gpu_scene.vertices.len() as u32,
        gpu_scene.triangles.len() as u32,
    )?;

    pb.inc(1);
    pb.set_message("upload environment");
    let env_data = env.unwrap_or(EnvGpuData {
        data: vec![[0.0, 0.0, 0.0, 1.0]],
        width: 1,
        height: 1,
        exposure: 1.0,
        use_sky: true,
    });

    let env_image = ctx.create_image_with_data(
        env_data.width,
        env_data.height,
        vk::Format::R32G32B32A32_SFLOAT,
        &env_data.data,
    )?;

    let env_sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            },
            None,
        )?
    };

    let output_image = ctx.create_storage_image(
        config.width,
        config.height,
        vk::Format::R32G32B32A32_SFLOAT,
    )?;

    let output_buffer = ctx.create_buffer(
        (config.width * config.height * 4 * mem::size_of::<f32>() as u32) as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let params = GpuParams {
        width: config.width,
        height: config.height,
        aa_samples: config.aa_samples,
        max_depth: config.max_depth.max(1) as u32,
        reflection_depth: config.reflection_depth.max(0) as u32,
        refraction_depth: config.refraction_depth.max(0) as u32,
        light_count: gpu_scene.lights.len() as u32,
        use_env: if env_data.use_sky { 0 } else { 1 },
        use_sky: if env_data.use_sky { 1 } else { 0 },
        env_width: env_data.width,
        env_height: env_data.height,
        tonemap: if config.tonemap { 1 } else { 0 },
        exposure: env_data.exposure,
        fov: config.fov,
        _pad0: 0,
        _pad1: 0,
    };

    let params_buffer = ctx.create_buffer_with_data(
        &[params],
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    )?;

    pb.inc(1);
    pb.set_message("create pipeline");
    let descriptor_set_layout = create_descriptor_set_layout(&device)?;
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            },
            None,
        )?
    };

    let shader_module = ctx.create_shader_module(COMPUTE_SHADER, "ray_query.comp")?;

    let stage_info = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::COMPUTE,
        module: shader_module,
        p_name: CStr::from_bytes_with_nul(b"main\0")?.as_ptr(),
        ..Default::default()
    };

    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo {
                stage: stage_info,
                layout: pipeline_layout,
                ..Default::default()
            }],
            None,
        )
    }
    .map_err(|(_, err)| err)?[0];

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 6,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        },
    ];

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            },
            None,
        )?
    };

    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        })?
    }[0];

    write_descriptor_set(
        &device,
        descriptor_set,
        &tlas,
        &output_image,
        &vertices_buffer,
        &normals_buffer,
        &triangles_buffer,
        &materials_buffer,
        &tri_materials_buffer,
        &lights_buffer,
        &params_buffer,
        env_sampler,
        &env_image,
    );

    pb.inc(1);
    pb.set_message("dispatch");
    unsafe {
        device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

        ctx.transition_image(
            output_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );

        let group_x = (config.width + 7) / 8;
        let group_y = (config.height + 7) / 8;
        device.cmd_dispatch(command_buffer, group_x, group_y, 1);

        ctx.transition_image(
            output_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        let region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: config.width,
                height: config.height,
                depth: 1,
            },
            ..Default::default()
        };

        device.cmd_copy_image_to_buffer(
            command_buffer,
            output_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output_buffer.buffer,
            &[region],
        );

        device.end_command_buffer(command_buffer)?;

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    pb.inc(1);
    pb.set_message("readback");
    let data = unsafe {
        let ptr = device.map_memory(
            output_buffer.memory,
            0,
            output_buffer.size,
            vk::MemoryMapFlags::empty(),
        )?;
        let slice = std::slice::from_raw_parts(
            ptr as *const f32,
            (config.width * config.height * 4) as usize,
        );
        let mut result = Vec::with_capacity((config.width * config.height) as usize);
        for i in 0..(config.width * config.height) as usize {
            let base = i * 4;
            result.push(Vec3::new(slice[base], slice[base + 1], slice[base + 2]));
        }
        device.unmap_memory(output_buffer.memory);
        result
    };

    pb.inc(1);
    pb.finish_with_message("render complete");

    unsafe {
        device.destroy_sampler(env_sampler, None);
        ctx.destroy_image(&env_image);
        ctx.destroy_image(&output_image);
        ctx.destroy_buffer(&params_buffer);
        ctx.destroy_buffer(&output_buffer);
        ctx.destroy_buffer(&lights_buffer);
        ctx.destroy_buffer(&tri_materials_buffer);
        ctx.destroy_buffer(&materials_buffer);
        ctx.destroy_buffer(&indices_buffer);
        ctx.destroy_buffer(&normals_buffer);
        ctx.destroy_buffer(&triangles_buffer);
        ctx.destroy_buffer(&vertices_buffer);

        accel_loader.destroy_acceleration_structure(blas.handle, None);
        accel_loader.destroy_acceleration_structure(tlas.handle, None);
        device.destroy_buffer(blas.buffer, None);
        device.destroy_buffer(tlas.buffer, None);
        device.free_memory(blas.memory, None);
        device.free_memory(tlas.memory, None);

        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    }

    ctx.destroy();
    Ok(data)
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, vk::Result> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 5,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 6,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 7,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 8,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 9,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
    ];

    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                ..Default::default()
            },
            None,
        )
    }
}

fn write_descriptor_set(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    tlas: &AccelResource,
    output_image: &ImageResource,
    vertices: &BufferResource,
    normals: &BufferResource,
    triangles: &BufferResource,
    materials: &BufferResource,
    tri_materials: &BufferResource,
    lights: &BufferResource,
    params: &BufferResource,
    sampler: vk::Sampler,
    env_image: &ImageResource,
) {
    let accel_info = vk::WriteDescriptorSetAccelerationStructureKHR {
        acceleration_structure_count: 1,
        p_acceleration_structures: &tlas.handle,
        ..Default::default()
    };

    let output_info = vk::DescriptorImageInfo {
        image_view: output_image.view,
        image_layout: vk::ImageLayout::GENERAL,
        ..Default::default()
    };

    let vertices_info = vk::DescriptorBufferInfo {
        buffer: vertices.buffer,
        offset: 0,
        range: vertices.size,
    };
    let triangles_info = vk::DescriptorBufferInfo {
        buffer: triangles.buffer,
        offset: 0,
        range: triangles.size,
    };
    let normals_info = vk::DescriptorBufferInfo {
        buffer: normals.buffer,
        offset: 0,
        range: normals.size,
    };
    let materials_info = vk::DescriptorBufferInfo {
        buffer: materials.buffer,
        offset: 0,
        range: materials.size,
    };
    let tri_materials_info = vk::DescriptorBufferInfo {
        buffer: tri_materials.buffer,
        offset: 0,
        range: tri_materials.size,
    };
    let lights_info = vk::DescriptorBufferInfo {
        buffer: lights.buffer,
        offset: 0,
        range: lights.size,
    };
    let params_info = vk::DescriptorBufferInfo {
        buffer: params.buffer,
        offset: 0,
        range: params.size,
    };

    let env_info = vk::DescriptorImageInfo {
        image_view: env_image.view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        sampler,
    };

    let accel_write = vk::WriteDescriptorSet {
        dst_set: descriptor_set,
        dst_binding: 0,
        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        descriptor_count: 1,
        p_next: &accel_info as *const _ as *const _,
        ..Default::default()
    };

    let writes = [
        accel_write,
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            p_image_info: &output_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &vertices_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &normals_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &triangles_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 5,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &materials_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 6,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &tri_materials_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 7,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &lights_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 8,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &params_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 9,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            p_image_info: &env_info,
            ..Default::default()
        },
    ];

    unsafe { device.update_descriptor_sets(&writes, &[]) };
}


const COMPUTE_SHADER: &str = r#"#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct Material {
    vec3 diffuse;
    float specular_exponent;
    vec4 albedo;
    float refractive_index;
    uint flags;
    uint _pad0;
    uint _pad1;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba32f) uniform image2D outImage;
layout(set = 0, binding = 2, std430) readonly buffer Vertices { vec4 vertices[]; };
layout(set = 0, binding = 3, std430) readonly buffer Normals { vec4 normals[]; };
layout(set = 0, binding = 4, std430) readonly buffer Triangles { uvec4 tris[]; };
layout(set = 0, binding = 5, std430) readonly buffer Materials { Material materials[]; };
layout(set = 0, binding = 6, std430) readonly buffer TriMaterials { uint tri_materials[]; };
layout(set = 0, binding = 7, std430) readonly buffer Lights { vec4 lights[]; };
layout(set = 0, binding = 8) uniform Params {
    uint width;
    uint height;
    uint aa_samples;
    uint max_depth;
    uint reflection_depth;
    uint refraction_depth;
    uint light_count;
    uint use_env;
    uint use_sky;
    uint env_width;
    uint env_height;
    uint tonemap;
    float exposure;
    float fov;
    uint _pad0;
    uint _pad1;
} params;
layout(set = 0, binding = 9) uniform sampler2D env_map;

const uint FLAG_CHECKER = 1u;
const float EPS = 2e-3;
const int MAX_STACK = 16;

float halton(uint index, uint base) {
    float result = 0.0;
    float f = 1.0 / float(base);
    uint i = index;
    while (i > 0u) {
        result += float(i % base) * f;
        i /= base;
        f /= float(base);
    }
    return result;
}

uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float rand01(uint seed) {
    return float(wang_hash(seed)) / 4294967296.0;
}

float max_component(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

vec3 reflect_dir(vec3 i, vec3 n) {
    return i - 2.0 * dot(i, n) * n;
}

vec3 refract_dir(vec3 i, vec3 n, float eta_t, float eta_i) {
    float cosi = clamp(-dot(i, n), -1.0, 1.0);
    float eta_i_local = eta_i;
    float eta_t_local = eta_t;
    vec3 n_local = n;
    if (cosi < 0.0) {
        cosi = -cosi;
        n_local = -n_local;
        float tmp = eta_i_local;
        eta_i_local = eta_t_local;
        eta_t_local = tmp;
    }
    float eta = eta_i_local / eta_t_local;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (k < 0.0) {
        return reflect_dir(i, n_local);
    }
    return normalize(i * eta + n_local * (eta * cosi - sqrt(k)));
}

vec3 offset_origin(vec3 point, vec3 normal, vec3 dir) {
    if (dot(dir, normal) < 0.0) {
        return point - normal * EPS;
    }
    return point + normal * EPS;
}

vec3 checker_color(vec3 pos) {
    int checker = (int(0.5 * pos.x + 1000.0) + int(0.5 * pos.z)) & 1;
    if (checker != 0) {
        return vec3(0.3, 0.3, 0.3);
    }
    return vec3(0.3, 0.2, 0.1);
}

vec3 sample_environment(vec3 dir) {
    if (params.use_sky != 0u) {
        vec3 sky_blue = vec3(0.5, 0.7, 1.0);
        vec3 horizon = vec3(1.0, 0.9, 0.7);
        float t = (normalize(dir).y + 1.0) * 0.5;
        return sky_blue * t + horizon * (1.0 - t);
    }

    vec3 n = normalize(dir);
    float phi = atan(n.z, n.x);
    float theta = acos(clamp(-n.y, -1.0, 1.0));
    float u = fract(phi / (2.0 * 3.14159265) + 0.5);
    float v = clamp(theta / 3.14159265, 0.0, 1.0);
    vec3 hdr = texture(env_map, vec2(u, v)).rgb * params.exposure;
    return hdr;
}

bool trace_ray(vec3 origin, vec3 dir, float t_max, out uint prim_id, out vec2 bary, out float t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return false;
    }
    prim_id = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

float shadow_ray(vec3 origin, vec3 dir, float dist) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001, dir, dist);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return 1.0;
    }
    return 0.0;
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    float half_w = float(params.width) * 0.5;
    float half_h = float(params.height) * 0.5;
    float fov_scale = tan(params.fov * 0.5);
    float dir_z = -float(params.height) / (2.0 * fov_scale);

    uint spp = max(params.aa_samples, 1u);
    uint sample_count = spp * spp;
    vec3 final_color = vec3(0.0);

    for (uint s = 0u; s < sample_count; ++s) {
        float jitter_x = halton(s, 2u);
        float jitter_y = halton(s, 3u);

        float dir_x = (float(gid.x) + jitter_x) - half_w;
        float dir_y = -(float(gid.y) + jitter_y) + half_h;
        vec3 ray_dir = normalize(vec3(dir_x, dir_y, dir_z));

        vec3 stack_origin[MAX_STACK];
        vec3 stack_dir[MAX_STACK];
        vec3 stack_weight[MAX_STACK];
        int stack_depth[MAX_STACK];
        int stack_size = 0;

        stack_origin[stack_size] = vec3(0.0, 0.0, 0.0);
        stack_dir[stack_size] = ray_dir;
        stack_weight[stack_size] = vec3(1.0);
        stack_depth[stack_size] = 0;
        stack_size++;

        vec3 sample_color = vec3(0.0);
        int max_depth = min(int(params.max_depth), MAX_STACK - 1);
        int max_reflect = min(int(params.reflection_depth), MAX_STACK - 1);
        int max_refract = min(int(params.refraction_depth), MAX_STACK - 1);

        while (stack_size > 0) {
            stack_size--;
            vec3 origin = stack_origin[stack_size];
            vec3 dir = stack_dir[stack_size];
            vec3 weight = stack_weight[stack_size];
            int depth = stack_depth[stack_size];

            uint prim_id;
            vec2 bary;
            float t;
            if (!trace_ray(origin, dir, 10000.0, prim_id, bary, t)) {
                sample_color += weight * sample_environment(dir);
                continue;
            }

            uvec4 tri = tris[prim_id];
            vec3 v0 = vertices[tri.x].xyz;
            vec3 v1 = vertices[tri.y].xyz;
            vec3 v2 = vertices[tri.z].xyz;
            vec3 n0 = normals[tri.x].xyz;
            vec3 n1 = normals[tri.y].xyz;
            vec3 n2 = normals[tri.z].xyz;
            float w = 1.0 - bary.x - bary.y;
            vec3 hit_pos = v0 * w + v1 * bary.x + v2 * bary.y;
            vec3 normal = normalize(n0 * w + n1 * bary.x + n2 * bary.y);
            if (dot(normal, dir) > 0.0) {
                normal = -normal;
            }

            uint mat_idx = tri_materials[prim_id];
            Material mat = materials[mat_idx];
            vec3 diffuse_color = mat.diffuse;

            if ((mat.flags & FLAG_CHECKER) != 0u) {
                diffuse_color = checker_color(hit_pos);
            }

            float diffuse_intensity = 0.0;
            float specular_intensity = 0.0;

            for (uint li = 0u; li < params.light_count; ++li) {
                vec3 light_pos = lights[li].xyz;
                vec3 light_dir = normalize(light_pos - hit_pos);
                float light_dist = length(light_pos - hit_pos);
                vec3 shadow_origin = offset_origin(hit_pos, normal, light_dir);
                float visibility = shadow_ray(shadow_origin, light_dir, light_dist - EPS);
                if (visibility <= 0.0) {
                    continue;
                }

                diffuse_intensity += max(dot(light_dir, normal), 0.0);
                if (mat.albedo.y > 0.0) {
                    vec3 refl = reflect_dir(-light_dir, normal);
                    specular_intensity += pow(max(dot(-refl, dir), 0.0), mat.specular_exponent);
                }
            }

            vec3 local_color = diffuse_color * diffuse_intensity * mat.albedo.x;
            local_color += vec3(1.0) * specular_intensity * mat.albedo.y;
            sample_color += weight * local_color;

            if (depth >= max_depth) {
                continue;
            }

            if (depth > 3) {
                float p = clamp(max_component(weight), 0.05, 0.95);
                uint seed = gid.x * 1973u + gid.y * 9277u + s * 26699u + uint(depth) * 104729u;
                if (rand01(seed) > p) {
                    continue;
                }
                weight /= p;
            }

            if (mat.albedo.z > 0.0 && depth < max_reflect && stack_size < MAX_STACK) {
                vec3 refl_dir = reflect_dir(dir, normal);
                vec3 refl_origin = offset_origin(hit_pos, normal, refl_dir);
                stack_origin[stack_size] = refl_origin;
                stack_dir[stack_size] = refl_dir;
                stack_weight[stack_size] = weight * mat.albedo.z;
                stack_depth[stack_size] = depth + 1;
                stack_size++;
            }

            if (mat.albedo.w > 0.0 && depth < max_refract && stack_size < MAX_STACK) {
                vec3 refr_dir = refract_dir(dir, normal, mat.refractive_index, 1.0);
                vec3 refr_origin = offset_origin(hit_pos, normal, refr_dir);
                stack_origin[stack_size] = refr_origin;
                stack_dir[stack_size] = refr_dir;
                stack_weight[stack_size] = weight * mat.albedo.w;
                stack_depth[stack_size] = depth + 1;
                stack_size++;
            }
        }

        final_color += sample_color;
    }

    final_color /= float(sample_count);
    imageStore(outImage, ivec2(gid), vec4(final_color, 1.0));
}
"#;
