use std::ffi::{CStr, CString};
use std::mem;
use std::ptr;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::environment::EnvGpuData;
use crate::gpu_scene::build_gpu_scene;
use crate::scene::Scene;

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

    let entry = unsafe { ash::Entry::load()? };

    let app_name = CString::new("nanotracer-rs")?;
    let engine_name = CString::new("nanotracer")?;
    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        application_version: 0,
        p_engine_name: engine_name.as_ptr(),
        engine_version: 0,
        api_version: vk::make_api_version(0, 1, 2, 0),
        ..Default::default()
    };

    let instance_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let (physical_device, queue_family_index) = pick_device(&instance)?;

    let device_extensions = [
        vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ptr(),
        vk::KHR_RAY_QUERY_NAME.as_ptr(),
        vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ptr(),
        vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
        vk::KHR_SPIRV_1_4_NAME.as_ptr(),
        vk::KHR_SHADER_FLOAT_CONTROLS_NAME.as_ptr(),
    ];

    let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures {
        buffer_device_address: vk::TRUE,
        ..Default::default()
    };
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR {
        ray_query: vk::TRUE,
        p_next: &mut bda_features as *mut _ as *mut _,
        ..Default::default()
    };
    let mut accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
        acceleration_structure: vk::TRUE,
        p_next: &mut ray_query_features as *mut _ as *mut _,
        ..Default::default()
    };

    let queue_priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo {
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: queue_priorities.as_ptr(),
        ..Default::default()
    }];

    let device_info = vk::DeviceCreateInfo {
        queue_create_info_count: queue_info.len() as u32,
        p_queue_create_infos: queue_info.as_ptr(),
        enabled_extension_count: device_extensions.len() as u32,
        pp_enabled_extension_names: device_extensions.as_ptr(),
        p_next: &mut accel_features as *mut _ as *mut _,
        ..Default::default()
    };

    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let accel_loader = ash::khr::acceleration_structure::Device::new(&instance, &device);

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo {
                queue_family_index,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
            None,
        )?
    };

    let command_buffer = unsafe {
        device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })?
    }[0];

    let vertices_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &gpu_scene.vertices,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let normals_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &gpu_scene.normals,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let triangles_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
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

    let indices_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &indices_flat,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
    )?;

    let tri_materials_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &gpu_scene.tri_materials,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let materials_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &gpu_scene.materials,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let lights_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &gpu_scene.lights,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let (blas, tlas) = build_acceleration_structures(
        &instance,
        &device,
        &accel_loader,
        physical_device,
        command_buffer,
        queue,
        &vertices_buffer,
        &indices_buffer,
        gpu_scene.vertices.len() as u32,
        gpu_scene.triangles.len() as u32,
    )?;

    let env_data = env.unwrap_or(EnvGpuData {
        data: vec![[0.0, 0.0, 0.0, 1.0]],
        width: 1,
        height: 1,
        exposure: 1.0,
        use_sky: true,
    });

    let env_image = create_image_with_data(
        &instance,
        &device,
        physical_device,
        command_buffer,
        queue,
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

    let output_image = create_storage_image(
        &instance,
        &device,
        physical_device,
        config.width,
        config.height,
        vk::Format::R32G32B32A32_SFLOAT,
    )?;

    let output_buffer = create_buffer(
        &instance,
        &device,
        physical_device,
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

    let params_buffer = create_buffer_with_data(
        &instance,
        &device,
        physical_device,
        &[params],
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    )?;

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

    let shader_module = create_shader_module(&device, COMPUTE_SHADER)?;

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

    unsafe {
        device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

        transition_image(
            &device,
            command_buffer,
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

        transition_image(
            &device,
            command_buffer,
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

    unsafe {
        device.destroy_sampler(env_sampler, None);
        destroy_image(&device, &env_image);
        destroy_image(&device, &output_image);
        destroy_buffer(&device, &params_buffer);
        destroy_buffer(&device, &output_buffer);
        destroy_buffer(&device, &lights_buffer);
        destroy_buffer(&device, &tri_materials_buffer);
        destroy_buffer(&device, &materials_buffer);
        destroy_buffer(&device, &indices_buffer);
        destroy_buffer(&device, &normals_buffer);
        destroy_buffer(&device, &triangles_buffer);
        destroy_buffer(&device, &vertices_buffer);

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
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(data)
}

fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), Box<dyn std::error::Error>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    for device in physical_devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();

        let queue_family_index = find_queue_family(instance, device)?;
        if queue_family_index.is_none() {
            continue;
        }

        if supports_extensions(instance, device)? {
            println!("Using GPU: {}", name);
            return Ok((device, queue_family_index.unwrap()));
        }
    }

    Err("No suitable Vulkan device found".into())
}

fn find_queue_family(instance: &ash::Instance, device: vk::PhysicalDevice) -> Result<Option<u32>, vk::Result> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in families.iter().enumerate() {
        if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Ok(Some(index as u32));
        }
    }
    Ok(None)
}

fn supports_extensions(instance: &ash::Instance, device: vk::PhysicalDevice) -> Result<bool, vk::Result> {
    let available = unsafe { instance.enumerate_device_extension_properties(device)? };
    let mut required = vec![
        vk::KHR_ACCELERATION_STRUCTURE_NAME.to_string_lossy(),
        vk::KHR_RAY_QUERY_NAME.to_string_lossy(),
        vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.to_string_lossy(),
        vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.to_string_lossy(),
        vk::KHR_SPIRV_1_4_NAME.to_string_lossy(),
        vk::KHR_SHADER_FLOAT_CONTROLS_NAME.to_string_lossy(),
    ];

    for ext in available {
        let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }.to_string_lossy();
        required.retain(|r| r.as_ref() != ext_name.as_ref());
    }

    Ok(required.is_empty())
}

struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

fn create_buffer_with_data<T: Pod>(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<BufferResource, Box<dyn std::error::Error>> {
    let size = (data.len() * mem::size_of::<T>()) as vk::DeviceSize;
    let buffer = create_buffer(
        instance,
        device,
        physical_device,
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    unsafe {
        let ptr = device.map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
        ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
        device.unmap_memory(buffer.memory);
    }

    Ok(buffer)
}

fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<BufferResource, Box<dyn std::error::Error>> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo {
                size,
                usage,
                ..Default::default()
            },
            None,
        )?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(instance, physical_device, requirements.memory_type_bits, properties)?;

    let alloc_flags = vk::MemoryAllocateFlagsInfo {
        flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        ..Default::default()
    };

    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: requirements.size,
        memory_type_index,
        p_next: &alloc_flags as *const _ as *const _,
        ..Default::default()
    };

    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    Ok(BufferResource {
        buffer,
        memory,
        size: requirements.size,
    })
}

fn destroy_buffer(device: &ash::Device, buffer: &BufferResource) {
    unsafe {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, Box<dyn std::error::Error>> {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_properties.memory_type_count {
        if type_filter & (1 << i) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    Err("Unable to find suitable memory type".into())
}

struct AccelResource {
    handle: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct AccelInstance {
    transform: [f32; 12],
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

fn build_acceleration_structures(
    instance: &ash::Instance,
    device: &ash::Device,
    accel_loader: &ash::khr::acceleration_structure::Device,
    physical_device: vk::PhysicalDevice,
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    vertices: &BufferResource,
    indices: &BufferResource,
    vertex_count: u32,
    triangle_count: u32,
) -> Result<(AccelResource, AccelResource), Box<dyn std::error::Error>> {
    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())? };
    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())? };

    let vertex_address = get_buffer_device_address(device, vertices.buffer);
    let index_address = get_buffer_device_address(device, indices.buffer);

    let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR {
        vertex_format: vk::Format::R32G32B32_SFLOAT,
        vertex_data: vk::DeviceOrHostAddressConstKHR { device_address: vertex_address },
        vertex_stride: mem::size_of::<[f32; 4]>() as vk::DeviceSize,
        max_vertex: vertex_count.saturating_sub(1),
        index_type: vk::IndexType::UINT32,
        index_data: vk::DeviceOrHostAddressConstKHR { device_address: index_address },
        ..Default::default()
    };

    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::TRIANGLES,
        geometry: vk::AccelerationStructureGeometryDataKHR { triangles: triangles_data },
        flags: vk::GeometryFlagsKHR::OPAQUE,
        ..Default::default()
    };

    let range = vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count: triangle_count,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    };

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &geometry,
        ..Default::default()
    };

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[triangle_count],
            &mut size_info,
        );
    }

    let blas = create_accel_resource(
        instance,
        device,
        physical_device,
        accel_loader,
        size_info.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
    )?;

    let scratch = create_buffer(
        instance,
        device,
        physical_device,
        size_info.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &geometry,
        dst_acceleration_structure: blas.handle,
        scratch_data: vk::DeviceOrHostAddressKHR {
            device_address: get_buffer_device_address(device, scratch.buffer),
        },
        ..Default::default()
    };

    let range_infos = [range];
    let range_ptrs: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&range_infos];
    unsafe {
        accel_loader.cmd_build_acceleration_structures(command_buffer, &[build_info], &range_ptrs);
    }

    let barrier = vk::MemoryBarrier {
        src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        ..Default::default()
    };
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }

    let blas_address = unsafe {
        accel_loader.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR {
            acceleration_structure: blas.handle,
            ..Default::default()
        })
    };

    let instance_flags =
        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
            | vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw();
    let instance_data = AccelInstance {
        transform: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        instance_custom_index_and_mask: (0 & 0x00FF_FFFF) | (0xFF << 24),
        instance_shader_binding_table_record_offset_and_flags: (0 & 0x00FF_FFFF)
            | (instance_flags << 24),
        acceleration_structure_reference: blas_address,
    };

    let instance_buffer = create_buffer_with_data(
        instance,
        device,
        physical_device,
        &[instance_data],
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )?;

    let tlas_geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                array_of_pointers: vk::FALSE,
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: get_buffer_device_address(device, instance_buffer.buffer),
                },
                ..Default::default()
            },
        },
        ..Default::default()
    };

    let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &tlas_geometry,
        ..Default::default()
    };

    let mut tlas_size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[1],
            &mut tlas_size_info,
        );
    }

    let tlas = create_accel_resource(
        instance,
        device,
        physical_device,
        accel_loader,
        tlas_size_info.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )?;

    let tlas_scratch = create_buffer(
        instance,
        device,
        physical_device,
        tlas_size_info.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &tlas_geometry,
        dst_acceleration_structure: tlas.handle,
        scratch_data: vk::DeviceOrHostAddressKHR {
            device_address: get_buffer_device_address(device, tlas_scratch.buffer),
        },
        ..Default::default()
    };

    let tlas_range = vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count: 1,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    };
    let tlas_ranges = [tlas_range];
    let tlas_ptrs: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&tlas_ranges];
    unsafe {
        accel_loader.cmd_build_acceleration_structures(command_buffer, &[tlas_build_info], &tlas_ptrs);
    }

    unsafe { device.end_command_buffer(command_buffer)? };
    unsafe {
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    destroy_buffer(device, &scratch);
    destroy_buffer(device, &instance_buffer);
    destroy_buffer(device, &tlas_scratch);

    Ok((blas, tlas))
}

fn create_accel_resource(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    accel_loader: &ash::khr::acceleration_structure::Device,
    size: vk::DeviceSize,
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<AccelResource, Box<dyn std::error::Error>> {
    let buffer = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let accel_info = vk::AccelerationStructureCreateInfoKHR {
        buffer: buffer.buffer,
        size,
        ty,
        ..Default::default()
    };
    let handle = unsafe { accel_loader.create_acceleration_structure(&accel_info, None)? };

    Ok(AccelResource {
        handle,
        buffer: buffer.buffer,
        memory: buffer.memory,
    })
}

fn get_buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let info = vk::BufferDeviceAddressInfo { buffer, ..Default::default() };
    unsafe { device.get_buffer_device_address(&info) }
}

struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

fn create_storage_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<ImageResource, Box<dyn std::error::Error>> {
    create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        format,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )
}

fn create_image_with_data<T: Pod>(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    width: u32,
    height: u32,
    format: vk::Format,
    data: &[T],
) -> Result<ImageResource, Box<dyn std::error::Error>> {
    let image = create_image(
        instance,
        device,
        physical_device,
        width,
        height,
        format,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
    )?;

    let staging = create_buffer_with_data(
        instance,
        device,
        physical_device,
        data,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())? };
    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())? };

    transition_image(device, command_buffer, image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

    let region = vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_extent: vk::Extent3D { width, height, depth: 1 },
        ..Default::default()
    };

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    transition_image(device, command_buffer, image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    unsafe { device.end_command_buffer(command_buffer)? };
    unsafe {
        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    destroy_buffer(device, &staging);

    Ok(image)
}

fn create_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<ImageResource, Box<dyn std::error::Error>> {
    let image = unsafe {
        device.create_image(
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D { width, height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            },
            None,
        )?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            },
            None,
        )?
    };

    Ok(ImageResource { image, memory, view })
}

fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..Default::default()
    };

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn destroy_image(device: &ash::Device, image: &ImageResource) {
    unsafe {
        device.destroy_image_view(image.view, None);
        device.destroy_image(image.image, None);
        device.free_memory(image.memory, None);
    }
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

fn create_shader_module(device: &ash::Device, source: &str) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
    let compiler = shaderc::Compiler::new()?;
    let binary = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, "ray_query.comp", "main", None)?;
    let module_info = vk::ShaderModuleCreateInfo {
        code_size: binary.as_binary().len() * 4,
        p_code: binary.as_binary().as_ptr(),
        ..Default::default()
    };
    let module = unsafe { device.create_shader_module(&module_info, None)? };
    Ok(module)
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
const float EPS = 1e-3;
const int MAX_STACK = 8;

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
    if (params.tonemap != 0u) {
        return hdr / (vec3(1.0) + hdr);
    }
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
