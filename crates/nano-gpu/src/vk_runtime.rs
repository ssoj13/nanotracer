use std::ffi::{CStr, CString};
use std::mem;
use std::ptr;

use ash::vk;
use bytemuck::Pod;

pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub accel_loader: ash::khr::acceleration_structure::Device,
}

pub struct BufferResource {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

pub struct ImageResource {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

pub struct AccelResource {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct AccelInstance {
    transform: [f32; 12],
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

impl VkContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
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

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            queue,
            queue_family_index,
            command_pool,
            command_buffer,
            accel_loader,
        })
    }

    pub fn create_buffer_with_data<T: Pod>(
        &self,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferResource, Box<dyn std::error::Error>> {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        let buffer = self.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let ptr = self.device.map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
            ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
            self.device.unmap_memory(buffer.memory);
        }

        Ok(buffer)
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<BufferResource, Box<dyn std::error::Error>> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                None,
            )?
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            &self.instance,
            self.physical_device,
            requirements.memory_type_bits,
            properties,
        )?;

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

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };

        Ok(BufferResource {
            buffer,
            memory,
            size: requirements.size,
        })
    }

    pub fn destroy_buffer(&self, buffer: &BufferResource) {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    fn create_image(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<ImageResource, Box<dyn std::error::Error>> {
        let image = unsafe {
            self.device.create_image(
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

        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_type_index = find_memory_type(
            &self.instance,
            self.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: requirements.size,
            memory_type_index,
            ..Default::default()
        };
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_image_memory(image, memory, 0)? };

        let view = unsafe {
            self.device.create_image_view(
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

    pub fn create_storage_image(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<ImageResource, Box<dyn std::error::Error>> {
        self.create_image(
            width,
            height,
            format,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }

    pub fn create_image_with_data<T: Pod>(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        data: &[T],
    ) -> Result<ImageResource, Box<dyn std::error::Error>> {
        let image = self.create_image(
            width,
            height,
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        let staging = self.create_buffer_with_data(data, vk::BufferUsageFlags::TRANSFER_SRC)?;

        unsafe { self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())? };
        unsafe { self.device.begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())? };

        transition_image(&self.device, self.command_buffer, image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

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
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        transition_image(
            &self.device,
            self.command_buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe { self.device.end_command_buffer(self.command_buffer)? };
        unsafe {
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };
            self.device.queue_submit(self.queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        self.destroy_buffer(&staging);

        Ok(image)
    }

    pub fn destroy_image(&self, image: &ImageResource) {
        unsafe {
            self.device.destroy_image_view(image.view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.memory, None);
        }
    }

    pub fn transition_image(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        transition_image(&self.device, self.command_buffer, image, old_layout, new_layout);
    }

    pub fn create_shader_module(&self, source: &str, name: &str) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        let compiler = shaderc::Compiler::new()?;
        let binary = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, name, "main", None)?;
        let module_info = vk::ShaderModuleCreateInfo {
            code_size: binary.as_binary().len() * 4,
            p_code: binary.as_binary().as_ptr(),
            ..Default::default()
        };
        let module = unsafe { self.device.create_shader_module(&module_info, None)? };
        Ok(module)
    }

    pub fn build_acceleration_structures(
        &self,
        vertices: &BufferResource,
        indices: &BufferResource,
        vertex_count: u32,
        triangle_count: u32,
    ) -> Result<(AccelResource, AccelResource), Box<dyn std::error::Error>> {
        unsafe { self.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())? };
        unsafe { self.device.begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())? };

        let vertex_address = get_buffer_device_address(&self.device, vertices.buffer);
        let index_address = get_buffer_device_address(&self.device, indices.buffer);

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
            self.accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[triangle_count],
                &mut size_info,
            );
        }

        let blas = create_accel_resource(self, size_info.acceleration_structure_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)?;

        let scratch = self.create_buffer(
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
                device_address: get_buffer_device_address(&self.device, scratch.buffer),
            },
            ..Default::default()
        };

        let range_infos = [range];
        let range_ptrs: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&range_infos];
        unsafe {
            self.accel_loader.cmd_build_acceleration_structures(self.command_buffer, &[build_info], &range_ptrs);
        }

        let barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            ..Default::default()
        };
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let blas_address = unsafe {
            self.accel_loader.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR {
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
            // Packed: low 24 bits = instance custom index (0), high 8 bits = visibility mask.
            // Custom index 0 means all triangles share the same gl_InstanceCustomIndexEXT.
            instance_custom_index_and_mask: 0xFFu32 << 24,
            // Packed: low 24 bits = SBT record offset (0, single hit group), high 8 bits = flags.
            instance_shader_binding_table_record_offset_and_flags: instance_flags << 24,
            acceleration_structure_reference: blas_address,
        };

        let instance_buffer = self.create_buffer_with_data(
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
                        device_address: get_buffer_device_address(&self.device, instance_buffer.buffer),
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
            self.accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build_info,
                &[1],
                &mut tlas_size_info,
            );
        }

        let tlas = create_accel_resource(self, tlas_size_info.acceleration_structure_size, vk::AccelerationStructureTypeKHR::TOP_LEVEL)?;

        let tlas_scratch = self.create_buffer(
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
                device_address: get_buffer_device_address(&self.device, tlas_scratch.buffer),
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
            self.accel_loader.cmd_build_acceleration_structures(self.command_buffer, &[tlas_build_info], &tlas_ptrs);
        }

        unsafe { self.device.end_command_buffer(self.command_buffer)? };
        unsafe {
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffer,
                ..Default::default()
            };
            self.device.queue_submit(self.queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        self.destroy_buffer(&scratch);
        self.destroy_buffer(&instance_buffer);
        self.destroy_buffer(&tlas_scratch);

        Ok((blas, tlas))
    }

    pub fn destroy(self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_accel_resource(
    ctx: &VkContext,
    size: vk::DeviceSize,
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<AccelResource, Box<dyn std::error::Error>> {
    let buffer = ctx.create_buffer(
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
    let handle = unsafe { ctx.accel_loader.create_acceleration_structure(&accel_info, None)? };

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
