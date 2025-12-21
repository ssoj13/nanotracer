use std::ffi::CStr;
use std::mem;
use std::time::Instant;

use ash::vk;
use indicatif::{ProgressBar, ProgressStyle};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::environment::EnvGpuData;
use crate::gpu_scene::build_gpu_scene_with_detail_boost;
use crate::scene::Scene;
use crate::splat_gpu::Gaussian;
use crate::vk_runtime::{AccelResource, BufferResource, ImageResource, VkContext};

#[derive(Clone, Copy, Debug)]
pub enum LightSampling {
    All,
    One,
}

pub struct SplatConfigGpu {
    pub density: f32,
    pub sh_samples: u32,
    pub max_depth: i32,
    pub reflection_depth: i32,
    pub refraction_depth: i32,
    pub scale_override: Option<f32>,
    pub tonemap: bool,
    pub glossy_mult: f32,
    pub radiance_clamp: f32,
    pub detail_boost: f32,
    pub detail_boost_max: f32,
    pub light_sampling: LightSampling,
    pub seed: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct GpuParams {
    sample_count: u32,
    tri_count: u32,
    sh_samples: u32,
    max_depth: u32,
    reflection_depth: u32,
    refraction_depth: u32,
    light_count: u32,
    light_sampling: u32,
    use_env: u32,
    use_sky: u32,
    env_width: u32,
    env_height: u32,
    tonemap: u32,
    exposure: f32,
    splat_scale: f32,
    radiance_clamp: f32,
    glossy_mult: f32,
    seed_lo: u32,
    seed_hi: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct GpuGaussian {
    pos: [f32; 4],
    normal: [f32; 4],
    sh_dc: [f32; 4],
    sh_rest: [[f32; 4]; 12],
    opacity_scale: [f32; 4],
    rotation: [f32; 4],
}

pub fn generate_splats_gpu(
    scene: &Scene,
    config: &SplatConfigGpu,
) -> Result<Vec<Gaussian>, Box<dyn std::error::Error>> {
    let gpu_scene = build_gpu_scene_with_detail_boost(
        scene,
        config.detail_boost,
        config.detail_boost_max,
    );
    let env = scene.environment.as_ref().map(|env| env.gpu_data());

    let pb = ProgressBar::new(7);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message("upload buffers");

    let total_start = Instant::now();
    let mut phase_start = Instant::now();

    let total_area: f32 = gpu_scene.tri_areas.iter().sum();
    if total_area <= 0.0 {
        return Ok(Vec::new());
    }
    let sample_count = (total_area * config.density).ceil().max(1.0) as u32;

    let splat_scale = config.scale_override.unwrap_or_else(|| estimate_scale(config.density, 2.0));

    let ctx = VkContext::new()?;
    let device = &ctx.device;
    let accel_loader = &ctx.accel_loader;

    let vertices_buffer = ctx.create_buffer_with_data(
        &gpu_scene.vertices,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let normals_buffer = ctx.create_buffer_with_data(&gpu_scene.normals, vk::BufferUsageFlags::STORAGE_BUFFER)?;

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

    let tri_materials_buffer =
        ctx.create_buffer_with_data(&gpu_scene.tri_materials, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let materials_buffer = ctx.create_buffer_with_data(&gpu_scene.materials, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let lights_buffer = ctx.create_buffer_with_data(&gpu_scene.lights, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let cdf_buffer = ctx.create_buffer_with_data(&gpu_scene.tri_cdf, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let t_buffers = phase_start.elapsed();
    phase_start = Instant::now();

    pb.inc(1);
    pb.set_message("build acceleration");
    let (blas, tlas) = ctx.build_acceleration_structures(
        &vertices_buffer,
        &indices_buffer,
        gpu_scene.vertices.len() as u32,
        gpu_scene.triangles.len() as u32,
    )?;

    let t_accel = phase_start.elapsed();
    phase_start = Instant::now();

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

    let t_env = phase_start.elapsed();
    phase_start = Instant::now();

    let params = GpuParams {
        sample_count,
        tri_count: gpu_scene.triangles.len() as u32,
        sh_samples: config.sh_samples.max(1),
        max_depth: config.max_depth.max(1) as u32,
        reflection_depth: config.reflection_depth.max(0) as u32,
        refraction_depth: config.refraction_depth.max(0) as u32,
        light_count: gpu_scene.lights.len() as u32,
        light_sampling: match config.light_sampling {
            LightSampling::All => 0,
            LightSampling::One => 1,
        },
        use_env: if env_data.use_sky { 0 } else { 1 },
        use_sky: if env_data.use_sky { 1 } else { 0 },
        env_width: env_data.width,
        env_height: env_data.height,
        tonemap: if config.tonemap { 1 } else { 0 },
        exposure: env_data.exposure,
        splat_scale,
        radiance_clamp: config.radiance_clamp.max(0.0),
        glossy_mult: config.glossy_mult.max(1.0),
        seed_lo: (config.seed & 0xFFFF_FFFF) as u32,
        seed_hi: ((config.seed >> 32) & 0xFFFF_FFFF) as u32,
    };

    let params_buffer = ctx.create_buffer_with_data(&[params], vk::BufferUsageFlags::UNIFORM_BUFFER)?;

    let output_buffer = ctx.create_buffer(
        (sample_count as usize * mem::size_of::<GpuGaussian>()) as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    pb.inc(1);
    pb.set_message("create pipeline");
    let descriptor_set_layout = create_splat_descriptor_set_layout(&device)?;
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

    let shader_module = ctx.create_shader_module(SPLAT_SHADER, "ray_query_splats")?;
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
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 7,
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

    write_splat_descriptor_set(
        &device,
        descriptor_set,
        &tlas,
        &vertices_buffer,
        &normals_buffer,
        &triangles_buffer,
        &materials_buffer,
        &tri_materials_buffer,
        &lights_buffer,
        &cdf_buffer,
        &output_buffer,
        &params_buffer,
        env_sampler,
        &env_image,
    );

    let t_pipeline = phase_start.elapsed();
    phase_start = Instant::now();

    pb.inc(1);
    pb.set_message("dispatch");
    unsafe {
        device.begin_command_buffer(ctx.command_buffer, &vk::CommandBufferBeginInfo::default())?;

        device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            ctx.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );

        let group_x = (sample_count + 63) / 64;
        device.cmd_dispatch(ctx.command_buffer, group_x, 1, 1);

        device.end_command_buffer(ctx.command_buffer)?;

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &ctx.command_buffer,
            ..Default::default()
        };
        device.queue_submit(ctx.queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(ctx.queue)?;
    }

    let t_dispatch = phase_start.elapsed();
    phase_start = Instant::now();

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
            ptr as *const GpuGaussian,
            sample_count as usize,
        );
        let mut result = Vec::with_capacity(sample_count as usize);
        for g in slice {
            let sh_rest = g
                .sh_rest
                .iter()
                .flat_map(|v| v.iter().copied())
                .take(45)
                .collect::<Vec<f32>>();
            result.push(Gaussian {
                pos: Vec3::new(g.pos[0], g.pos[1], g.pos[2]),
                normal: Vec3::new(g.normal[0], g.normal[1], g.normal[2]),
                sh_dc: [g.sh_dc[0], g.sh_dc[1], g.sh_dc[2]],
                sh_rest,
                opacity: g.opacity_scale[0],
                scale: [g.opacity_scale[1], g.opacity_scale[2], g.opacity_scale[3]],
                rotation: g.rotation,
            });
        }
        device.unmap_memory(output_buffer.memory);
        result
    };

    pb.inc(1);
    pb.finish_with_message("splats complete");

    let t_readback = phase_start.elapsed();
    let t_total = total_start.elapsed();
    println!(
        "Timing (GPU splats): buffers {:.2}s, accel {:.2}s, env {:.2}s, pipeline {:.2}s, dispatch {:.2}s, readback {:.2}s, total {:.2}s",
        t_buffers.as_secs_f32(),
        t_accel.as_secs_f32(),
        t_env.as_secs_f32(),
        t_pipeline.as_secs_f32(),
        t_dispatch.as_secs_f32(),
        t_readback.as_secs_f32(),
        t_total.as_secs_f32()
    );

    unsafe {
        device.destroy_sampler(env_sampler, None);
    }
    ctx.destroy_image(&env_image);
    ctx.destroy_buffer(&params_buffer);
    ctx.destroy_buffer(&output_buffer);
    ctx.destroy_buffer(&cdf_buffer);
    ctx.destroy_buffer(&lights_buffer);
    ctx.destroy_buffer(&tri_materials_buffer);
    ctx.destroy_buffer(&materials_buffer);
    ctx.destroy_buffer(&indices_buffer);
    ctx.destroy_buffer(&triangles_buffer);
    ctx.destroy_buffer(&normals_buffer);
    ctx.destroy_buffer(&vertices_buffer);

    unsafe {
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

fn estimate_scale(density: f32, overlap: f32) -> f32 {
    let area_per_sample = 1.0 / density;
    let base_radius = (area_per_sample / std::f32::consts::PI).sqrt();
    base_radius * overlap
}

fn create_splat_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, vk::Result> {
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
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
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
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 9,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 10,
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

#[allow(clippy::too_many_arguments)]
fn write_splat_descriptor_set(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    tlas: &AccelResource,
    vertices: &BufferResource,
    normals: &BufferResource,
    triangles: &BufferResource,
    materials: &BufferResource,
    tri_materials: &BufferResource,
    lights: &BufferResource,
    cdf: &BufferResource,
    output: &BufferResource,
    params: &BufferResource,
    sampler: vk::Sampler,
    env_image: &ImageResource,
) {
    let accel_info = vk::WriteDescriptorSetAccelerationStructureKHR {
        acceleration_structure_count: 1,
        p_acceleration_structures: &tlas.handle,
        ..Default::default()
    };

    let vertices_info = vk::DescriptorBufferInfo {
        buffer: vertices.buffer,
        offset: 0,
        range: vertices.size,
    };
    let normals_info = vk::DescriptorBufferInfo {
        buffer: normals.buffer,
        offset: 0,
        range: normals.size,
    };
    let triangles_info = vk::DescriptorBufferInfo {
        buffer: triangles.buffer,
        offset: 0,
        range: triangles.size,
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
    let cdf_info = vk::DescriptorBufferInfo {
        buffer: cdf.buffer,
        offset: 0,
        range: cdf.size,
    };
    let output_info = vk::DescriptorBufferInfo {
        buffer: output.buffer,
        offset: 0,
        range: output.size,
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
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &vertices_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &normals_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &triangles_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &materials_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 5,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &tri_materials_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 6,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &lights_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 7,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &cdf_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 8,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &output_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 9,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            p_buffer_info: &params_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 10,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            p_image_info: &env_info,
            ..Default::default()
        },
    ];

    unsafe { device.update_descriptor_sets(&writes, &[]) };
}


const SPLAT_SHADER: &str = r#"#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Material {
    vec3 diffuse;
    float specular_exponent;
    vec4 albedo;
    float refractive_index;
    uint flags;
    uint _pad0;
    uint _pad1;
};

struct GaussianOut {
    vec4 pos;
    vec4 normal;
    vec4 sh_dc;
    vec4 sh_rest[12];
    vec4 opacity_scale;
    vec4 rotation;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, std430) readonly buffer Vertices { vec4 vertices[]; };
layout(set = 0, binding = 2, std430) readonly buffer Normals { vec4 normals[]; };
layout(set = 0, binding = 3, std430) readonly buffer Triangles { uvec4 tris[]; };
layout(set = 0, binding = 4, std430) readonly buffer Materials { Material materials[]; };
layout(set = 0, binding = 5, std430) readonly buffer TriMaterials { uint tri_materials[]; };
layout(set = 0, binding = 6, std430) readonly buffer Lights { vec4 lights[]; };
layout(set = 0, binding = 7, std430) readonly buffer TriCdf { float tri_cdf[]; };
layout(set = 0, binding = 8, std430) writeonly buffer Gaussians { GaussianOut gaussians[]; };
layout(set = 0, binding = 9) uniform Params {
    uint sample_count;
    uint tri_count;
    uint sh_samples;
    uint max_depth;
    uint reflection_depth;
    uint refraction_depth;
    uint light_count;
    uint light_sampling;
    uint use_env;
    uint use_sky;
    uint env_width;
    uint env_height;
    uint tonemap;
    float exposure;
    float splat_scale;
    float radiance_clamp;
    float glossy_mult;
    uint seed_lo;
    uint seed_hi;
} params;
layout(set = 0, binding = 10) uniform sampler2D env_map;

const float EPS = 2e-3;
const int MAX_STACK = 16;
const float SH_C0 = 0.2820948;
const float SH_C1 = 0.48860252;
const float SH_C2[5] = float[5](1.0925485, -1.0925485, 0.31539157, -1.0925485, 0.54627424);
const float SH_C3[7] = float[7](-0.5900436, 2.8906114, -0.4570458, 0.37317634, -0.4570458, 1.4453057, -0.5900436);

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

const float PI = 3.14159265;
const float GOLDEN = 0.61803398875;

vec3 sample_uniform_hemisphere(uint seed, uint idx, uint count) {
    float u1 = (float(idx) + rand01(seed + idx * 1664525u)) / max(1.0, float(count));
    float u2 = fract((float(idx) + rand01(seed + idx * 1013904223u)) * GOLDEN);
    float z = 1.0 - u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u2;
    return vec3(r * cos(phi), r * sin(phi), z);
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

vec3 tonemap_reinhard(vec3 c) {
    vec3 v = max(c, vec3(0.0));
    return v / (vec3(1.0) + v);
}

vec3 linear_to_srgb(vec3 linear) {
    vec3 v = clamp(linear, vec3(0.0), vec3(1.0));
    vec3 low = v * 12.92;
    vec3 high = 1.055 * pow(v, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 cutoff = vec3(0.0031308);
    return mix(high, low, lessThanEqual(v, cutoff));
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

void build_tangent_frame(vec3 n, out vec3 t, out vec3 b) {
    if (abs(n.y) < 0.9) {
        t = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
    } else {
        t = normalize(cross(n, vec3(1.0, 0.0, 0.0)));
    }
    b = cross(n, t);
}

vec3 trace_path(vec3 origin, vec3 dir, uint seed_base) {
    vec3 stack_origin[MAX_STACK];
    vec3 stack_dir[MAX_STACK];
    vec3 stack_weight[MAX_STACK];
    int stack_depth[MAX_STACK];
    int stack_size = 0;

    stack_origin[stack_size] = origin;
    stack_dir[stack_size] = dir;
    stack_weight[stack_size] = vec3(1.0);
    stack_depth[stack_size] = 0;
    stack_size++;

    vec3 sample_color = vec3(0.0);
    int max_depth = min(int(params.max_depth), MAX_STACK - 1);
    int max_reflect = min(int(params.reflection_depth), MAX_STACK - 1);
    int max_refract = min(int(params.refraction_depth), MAX_STACK - 1);

    while (stack_size > 0) {
        stack_size--;
        vec3 o = stack_origin[stack_size];
        vec3 d = stack_dir[stack_size];
        vec3 weight = stack_weight[stack_size];
        int depth = stack_depth[stack_size];

        uint prim_id;
        vec2 bary;
        float t_hit;
        if (!trace_ray(o, d, 10000.0, prim_id, bary, t_hit)) {
            sample_color += weight * sample_environment(d);
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
        if (dot(normal, d) > 0.0) {
            normal = -normal;
        }

        uint mat_idx = tri_materials[prim_id];
        Material mat = materials[mat_idx];
        vec3 diffuse_color = mat.diffuse;

        if ((mat.flags & 1u) != 0u) {
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
                specular_intensity += pow(max(dot(-refl, d), 0.0), mat.specular_exponent);
            }
        }

        vec3 local_color = diffuse_color * diffuse_intensity * mat.albedo.x;
        local_color += vec3(1.0) * specular_intensity * mat.albedo.y;
        sample_color += weight * local_color;

        if (depth >= max_depth) {
            continue;
        }

        if (mat.albedo.z > 0.0 && depth < max_reflect && stack_size < MAX_STACK) {
            vec3 refl_dir = reflect_dir(d, normal);
            vec3 refl_origin = offset_origin(hit_pos, normal, refl_dir);
            stack_origin[stack_size] = refl_origin;
            stack_dir[stack_size] = refl_dir;
            stack_weight[stack_size] = weight * mat.albedo.z;
            stack_depth[stack_size] = depth + 1;
            stack_size++;
        }

        if (mat.albedo.w > 0.0 && depth < max_refract && stack_size < MAX_STACK) {
            vec3 refr_dir = refract_dir(d, normal, mat.refractive_index, 1.0);
            vec3 refr_origin = offset_origin(hit_pos, normal, refr_dir);
            stack_origin[stack_size] = refr_origin;
            stack_dir[stack_size] = refr_dir;
            stack_weight[stack_size] = weight * mat.albedo.w;
            stack_depth[stack_size] = depth + 1;
            stack_size++;
        }
    }

    return sample_color;
}

vec3 shade_surface(vec3 pos, vec3 normal, vec3 view_dir, Material mat, vec3 diffuse_color, uint seed_base) {
    vec3 incoming_dir = -view_dir;
    bool is_diffuse_only = mat.albedo.z <= 0.0 && mat.albedo.w <= 0.0;

    float diffuse_intensity = 0.0;
    float specular_intensity = 0.0;

    if (!is_diffuse_only || params.light_count > 0u) {
        if (params.light_count > 0u) {
            if (params.light_sampling == 0u) {
                for (uint li = 0u; li < params.light_count; ++li) {
                    vec3 light_pos = lights[li].xyz;
                    vec3 light_dir = normalize(light_pos - pos);
                    float light_dist = length(light_pos - pos);
                    vec3 shadow_origin = offset_origin(pos, normal, light_dir);
                    float visibility = shadow_ray(shadow_origin, light_dir, light_dist - EPS);
                    if (visibility <= 0.0) {
                        continue;
                    }

                    diffuse_intensity += max(dot(light_dir, normal), 0.0);
                    if (!is_diffuse_only) {
                        vec3 refl = reflect_dir(-light_dir, normal);
                        specular_intensity += pow(max(dot(-refl, incoming_dir), 0.0), mat.specular_exponent);
                    }
                }
            } else {
                uint li = min(uint(rand01(seed_base ^ 0x9e3779b9u) * float(params.light_count)), params.light_count - 1u);
                float light_weight = float(params.light_count);

                vec3 light_pos = lights[li].xyz;
                vec3 light_dir = normalize(light_pos - pos);
                float light_dist = length(light_pos - pos);
                vec3 shadow_origin = offset_origin(pos, normal, light_dir);
                float visibility = shadow_ray(shadow_origin, light_dir, light_dist - EPS);
                if (visibility > 0.0) {
                    diffuse_intensity += max(dot(light_dir, normal), 0.0) * light_weight;
                    if (!is_diffuse_only) {
                        vec3 refl = reflect_dir(-light_dir, normal);
                        specular_intensity += pow(max(dot(-refl, incoming_dir), 0.0), mat.specular_exponent) * light_weight;
                    }
                }
            }
        }
    }

    vec3 color = diffuse_color * diffuse_intensity * mat.albedo.x;
    color += vec3(1.0) * specular_intensity * mat.albedo.y;

    if (!is_diffuse_only) {
        if (mat.albedo.z > 0.0) {
            vec3 refl_dir = reflect_dir(incoming_dir, normal);
            vec3 refl_origin = offset_origin(pos, normal, refl_dir);
            color += trace_path(refl_origin, refl_dir, seed_base) * mat.albedo.z;
        }

        if (mat.albedo.w > 0.0) {
            vec3 refr_dir = refract_dir(incoming_dir, normal, mat.refractive_index, 1.0);
            vec3 refr_origin = offset_origin(pos, normal, refr_dir);
            color += trace_path(refr_origin, refr_dir, seed_base) * mat.albedo.w;
        }
    }

    return color;
}

float sh_basis(int index, vec3 dir) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    if (index == 0) return SH_C0;
    if (index == 1) return -SH_C1 * y;
    if (index == 2) return SH_C1 * z;
    if (index == 3) return -SH_C1 * x;

    if (index == 4) return SH_C2[0] * x * y;
    if (index == 5) return SH_C2[1] * y * z;
    if (index == 6) return SH_C2[2] * (2.0 * zz - xx - yy);
    if (index == 7) return SH_C2[3] * x * z;
    if (index == 8) return SH_C2[4] * (xx - yy);

    if (index == 9) return SH_C3[0] * y * (3.0 * xx - yy);
    if (index == 10) return SH_C3[1] * x * y * z;
    if (index == 11) return SH_C3[2] * y * (4.0 * zz - xx - yy);
    if (index == 12) return SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    if (index == 13) return SH_C3[4] * x * (4.0 * zz - xx - yy);
    if (index == 14) return SH_C3[5] * z * (xx - yy);
    if (index == 15) return SH_C3[6] * x * (xx - 3.0 * yy);

    return 0.0;
}

void solve_linear(inout float a[16][16], inout float b[16], out float x[16]) {
    for (int col = 0; col < 16; ++col) {
        int pivot = col;
        float pivot_val = abs(a[col][col]);
        for (int r = col + 1; r < 16; ++r) {
            float v = abs(a[r][col]);
            if (v > pivot_val) {
                pivot_val = v;
                pivot = r;
            }
        }
        if (pivot != col) {
            for (int c = col; c < 16; ++c) {
                float tmp = a[col][c];
                a[col][c] = a[pivot][c];
                a[pivot][c] = tmp;
            }
            float tb = b[col];
            b[col] = b[pivot];
            b[pivot] = tb;
        }
        float diag = a[col][col];
        if (abs(diag) < 1e-10) {
            continue;
        }
        float inv = 1.0 / diag;
        for (int c = col; c < 16; ++c) {
            a[col][c] *= inv;
        }
        b[col] *= inv;
        for (int r = 0; r < 16; ++r) {
            if (r == col) {
                continue;
            }
            float factor = a[r][col];
            if (abs(factor) < 1e-12) {
                continue;
            }
            for (int c = col; c < 16; ++c) {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    for (int i = 0; i < 16; ++i) {
        x[i] = b[i];
    }
}

vec4 quat_from_normal(vec3 normal) {
    vec3 z = vec3(0.0, 0.0, 1.0);
    vec3 n = normalize(normal);
    float dotv = dot(z, n);
    if (dotv > 0.99999) {
        return vec4(1.0, 0.0, 0.0, 0.0);
    }
    if (dotv < -0.99999) {
        return vec4(0.0, 1.0, 0.0, 0.0);
    }
    vec3 axis = normalize(cross(z, n));
    float angle = acos(dotv);
    float half_angle = angle * 0.5;
    float s = sin(half_angle);
    float c = cos(half_angle);
    return vec4(c, axis.x * s, axis.y * s, axis.z * s);
}

uint find_triangle(float r) {
    uint lo = 0u;
    uint hi = params.tri_count - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (r <= tri_cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= params.sample_count) {
        return;
    }

    uint seed_base = params.seed_lo ^ (params.seed_hi * 1664525u);
    float pick = rand01(seed_base + id * 1013904223u);
    uint tri_idx = find_triangle(pick);

    uvec4 tri = tris[tri_idx];
    vec3 v0 = vertices[tri.x].xyz;
    vec3 v1 = vertices[tri.y].xyz;
    vec3 v2 = vertices[tri.z].xyz;
    vec3 n0 = normals[tri.x].xyz;
    vec3 n1 = normals[tri.y].xyz;
    vec3 n2 = normals[tri.z].xyz;

    float r1 = rand01(seed_base + id * 73856093u + 1u);
    float r2 = rand01(seed_base + id * 19349663u + 2u);
    float sqrt_r1 = sqrt(r1);
    float u = 1.0 - sqrt_r1;
    float v = r2 * sqrt_r1;
    float w = 1.0 - u - v;

    vec3 pos = v0 * w + v1 * u + v2 * v;
    vec3 normal = normalize(n0 * w + n1 * u + n2 * v);

    uint mat_idx = tri_materials[tri_idx];
    Material mat = materials[mat_idx];
    vec3 diffuse_color = mat.diffuse;
    if ((mat.flags & 1u) != 0u) {
        diffuse_color = checker_color(pos);
    }

    vec3 tangent;
    vec3 bitangent;
    build_tangent_frame(normal, tangent, bitangent);

    float ata[16][16];
    vec3 atb[16];
    for (int i = 0; i < 16; ++i) {
        atb[i] = vec3(0.0);
        for (int j = 0; j < 16; ++j) {
            ata[i][j] = 0.0;
        }
    }

    uint base_samples = max(params.sh_samples, 1u);
    uint glossy_samples = 0u;
    float glossy_mult = max(params.glossy_mult, 1.0);
    if (mat.albedo.z > 0.0 || mat.albedo.w > 0.0) {
        glossy_samples = uint(float(base_samples) * max(glossy_mult - 1.0, 0.0));
    }

    uint dir_seed = seed_base ^ (id * 1597334677u);
    for (uint s = 0u; s < base_samples; ++s) {
        vec3 local_dir = sample_uniform_hemisphere(dir_seed, s, base_samples);
        vec3 world_dir = normalize(local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal);

        vec3 view_dir = world_dir;
        uint sample_seed = seed_base ^ (id * 747796405u) ^ (s * 277803737u);
        vec3 radiance = shade_surface(pos, normal, view_dir, mat, diffuse_color, sample_seed);
        float clamp_val = max(params.radiance_clamp, 0.0);
        if (clamp_val > 0.0) {
            float luma = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
            if (luma > clamp_val) {
                radiance *= clamp_val / luma;
            }
        }
        vec3 mapped = (params.tonemap != 0u) ? tonemap_reinhard(radiance) : clamp(radiance, vec3(0.0), vec3(1.0));
        vec3 srgb = linear_to_srgb(mapped);
        vec3 b = srgb - vec3(0.5);

        float basis[16];
        for (int i = 0; i < 16; ++i) {
            basis[i] = sh_basis(i, view_dir);
        }

        for (int i = 0; i < 16; ++i) {
            atb[i] += basis[i] * b;
            for (int j = 0; j < 16; ++j) {
                ata[i][j] += basis[i] * basis[j];
            }
        }
    }

    if (glossy_samples > 0u) {
        uint glossy_seed = dir_seed ^ 0x9e3779b9u;
        for (uint s = 0u; s < glossy_samples; ++s) {
            vec3 local_dir = sample_uniform_hemisphere(glossy_seed, s, glossy_samples);
            vec3 world_dir = normalize(local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal);

            vec3 view_dir = world_dir;
            uint sample_seed = seed_base ^ (id * 747796405u) ^ ((s + base_samples) * 277803737u);
            vec3 radiance = shade_surface(pos, normal, view_dir, mat, diffuse_color, sample_seed);
            float clamp_val = max(params.radiance_clamp, 0.0);
            if (clamp_val > 0.0) {
                float luma = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
                if (luma > clamp_val) {
                    radiance *= clamp_val / luma;
                }
            }
            vec3 mapped = (params.tonemap != 0u) ? tonemap_reinhard(radiance) : clamp(radiance, vec3(0.0), vec3(1.0));
            vec3 srgb = linear_to_srgb(mapped);
            vec3 b = srgb - vec3(0.5);

            float basis[16];
            for (int i = 0; i < 16; ++i) {
                basis[i] = sh_basis(i, view_dir);
            }

            for (int i = 0; i < 16; ++i) {
                atb[i] += basis[i] * b;
                for (int j = 0; j < 16; ++j) {
                    ata[i][j] += basis[i] * basis[j];
                }
            }
        }
    }

    for (int i = 0; i < 16; ++i) {
        ata[i][i] += 1e-4;
    }

    float a_r[16][16];
    float a_g[16][16];
    float a_b[16][16];
    float b_r[16];
    float b_g[16];
    float b_b[16];
    for (int i = 0; i < 16; ++i) {
        b_r[i] = atb[i].x;
        b_g[i] = atb[i].y;
        b_b[i] = atb[i].z;
        for (int j = 0; j < 16; ++j) {
            float v = ata[i][j];
            a_r[i][j] = v;
            a_g[i][j] = v;
            a_b[i][j] = v;
        }
    }

    float coeff_r[16];
    float coeff_g[16];
    float coeff_b[16];
    solve_linear(a_r, b_r, coeff_r);
    solve_linear(a_g, b_g, coeff_g);
    solve_linear(a_b, b_b, coeff_b);

    vec3 coeffs[16];
    for (int i = 0; i < 16; ++i) {
        coeffs[i] = vec3(coeff_r[i], coeff_g[i], coeff_b[i]);
    }

    vec3 sh_dc = coeffs[0] / SH_C0;

    GaussianOut out_g;
    out_g.pos = vec4(pos, 1.0);
    out_g.normal = vec4(normal, 0.0);
    out_g.sh_dc = vec4(sh_dc, 0.0);

    // Pack planar format: R[15], G[15], B[15]
    float rest[45];
    int off = 0;
    for (int i = 1; i < 16; ++i) { rest[off++] = coeffs[i].x; }
    for (int i = 1; i < 16; ++i) { rest[off++] = coeffs[i].y; }
    for (int i = 1; i < 16; ++i) { rest[off++] = coeffs[i].z; }

    for (int i = 0; i < 12; ++i) {
        vec4 v4 = vec4(0.0);
        for (int k = 0; k < 4; ++k) {
            int ri = i * 4 + k;
            if (ri < 45) {
                v4[k] = rest[ri];
            }
        }
        out_g.sh_rest[i] = v4;
    }

    float opacity_p = (mat.albedo.w > 0.0) ? 0.15 : 0.98;
    float opacity = log(opacity_p / (1.0 - opacity_p));

    float tangent_sigma = max(params.splat_scale, 1e-6);
    float normal_sigma = max(params.splat_scale * 0.3, 1e-6);
    out_g.opacity_scale = vec4(opacity, log(tangent_sigma), log(tangent_sigma), log(normal_sigma));
    out_g.rotation = quat_from_normal(normal);

    gaussians[id] = out_g;
}
"#;
