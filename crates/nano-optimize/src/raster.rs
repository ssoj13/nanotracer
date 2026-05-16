//! Forward splat rasteriser — Phase A2 of the 3DGS training pipeline.
//!
//! Currently lands the projection pass (Phase A2.1): each 3D Gaussian
//! splat is mapped to a 2D screen-space ellipse parameterised by a
//! "conic" (inverse covariance) + pixel-space mean / depth / radius.
//! Subsequent A2.x sub-phases bolt on tile binning, sort, and α-blend
//! using the same `ProjectedSplat` records.
//!
//! The kernel lives in `shaders/project_gaussians.wgsl`. Math follows
//! graphdeco-inria 3DGS `forward.cu` §1.2–1.4 (covariance projection,
//! 3-pixel low-pass filter) with the renderer's right-handed view
//! convention. A CPU oracle in [`cpu_oracle`] mirrors the WGSL exactly
//! so unit tests can verify per-splat parity to ~1e-3.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::gpu::WgpuCtx;
use crate::splat_gpu::GpuSplatBuffer;
use crate::tile_binner::TilingParams;

/// 48-byte per-splat record emitted by the projection kernel.
///
/// Layout maps to three `vec4` slots in WGSL — std430-aligned.
/// `visible == 0.0` marks "behind camera / off-screen / degenerate"
/// and downstream passes must skip those entries.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ProjectedSplat {
    pub mean_xy: [f32; 2],
    pub depth: f32,
    pub radius: f32,
    pub conic: [f32; 3],
    pub visible: f32,
    pub color: [f32; 3],
    pub opacity: f32,
}

const _: () = assert!(core::mem::size_of::<ProjectedSplat>() == 48);

/// Per-splat 2D-space gradient state filled by `rasterize_backward.wgsl`
/// and consumed by `project_backward.wgsl`. 48 bytes (`3 × vec4`),
/// std430-aligned. On the GPU the buffer is bound as
/// `array<atomic<u32>>` so per-pixel contributions can accumulate via
/// CAS-based `atomic_add_f32`; the u32 storage holds raw `f32` bits.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ProjectedGrad {
    /// .xy = dL/dmean (pixel-space), .z = dL/dopacity, .w = padding.
    pub dmean_opacity: [f32; 4],
    /// .xyz = dL/dconic (a, b, c), .w = padding.
    pub dconic: [f32; 4],
    /// .xyz = dL/dcolor (RGB), .w = padding.
    pub dcolor: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ProjectedGrad>() == 48);

/// GPU-side mirror of the WGSL `Camera` struct. 96 bytes, std140 layout.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub near: f32,
    pub focal: f32,
    pub width: f32,
    pub height: f32,
    pub n_splats: u32,
}

const _: () = assert!(core::mem::size_of::<CameraUniform>() == 96);

impl CameraUniform {
    /// Build a uniform from world-space camera pose + image plane.
    ///
    /// `fov_y` is vertical field of view in radians; the focal length
    /// is derived as `H / (2·tan(fov_y/2))` and assumed square (`fx == fy`).
    pub fn from_pose(
        camera_pos: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y: f32,
        width: u32,
        height: u32,
        n_splats: u32,
    ) -> Self {
        let view = Mat4::look_at_rh(camera_pos, target, up);
        let focal = (height as f32) / (2.0 * (fov_y * 0.5).tan());
        Self {
            view: view.to_cols_array_2d(),
            cam_pos: [camera_pos.x, camera_pos.y, camera_pos.z],
            near: 0.01,
            focal,
            width: width as f32,
            height: height as f32,
            n_splats,
        }
    }
}

/// Composite pass uniform mirroring `rasterize.wgsl::Params`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    width: u32,
    height: u32,
    tile_size: u32,
    tiles_x: u32,
}

/// Compiled wgpu pipelines for the rasteriser passes. Build once per
/// training run and reuse for every iteration / reference view.
pub struct Rasterizer {
    project_pipeline: wgpu::ComputePipeline,
    project_bgl: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    composite_backward_pipeline: wgpu::ComputePipeline,
    composite_backward_bgl: wgpu::BindGroupLayout,
    project_backward_pipeline: wgpu::ComputePipeline,
    project_backward_bgl: wgpu::BindGroupLayout,
    tonemap_pipeline: wgpu::ComputePipeline,
    tonemap_bgl: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TonemapParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

impl Rasterizer {
    pub fn new(ctx: &WgpuCtx) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("project_gaussians"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/project_gaussians.wgsl").into(),
                ),
            });

        let entries: [wgpu::BindGroupLayoutEntry; 8] = std::array::from_fn(|i| {
            let (binding_idx, ty) = match i {
                0 => (
                    0u32,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                k @ 1..=6 => (
                    k as u32,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                _ => (
                    7u32,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            };
            wgpu::BindGroupLayoutEntry {
                binding: binding_idx,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        });

        let project_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("project-bgl"),
                entries: &entries,
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("project-pl"),
                bind_group_layouts: &[Some(&project_bgl)],
                immediate_size: 0,
            });

        let project_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("project-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Composite (α-blend) pipeline ----------------------------------
        let composite_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rasterize"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rasterize.wgsl").into()),
            });

        let storage_rw = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage_ro = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let composite_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite-bgl"),
                entries: &[
                    bgl_entry(0, storage_ro),
                    bgl_entry(1, storage_ro),
                    bgl_entry(2, storage_ro),
                    bgl_entry(3, storage_rw),
                    bgl_entry(4, uniform),
                ],
            });
        let composite_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("composite-pl"),
                bind_group_layouts: &[Some(&composite_bgl)],
                immediate_size: 0,
            });
        let composite_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("composite-pipeline"),
                    layout: Some(&composite_layout),
                    module: &composite_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Composite-backward (α-blend reverse) pipeline ---------------
        let composite_backward_module =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("rasterize_backward"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/rasterize_backward.wgsl").into(),
                    ),
                });
        let composite_backward_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("composite-bwd-bgl"),
                    entries: &[
                        bgl_entry(0, storage_ro), // projected
                        bgl_entry(1, storage_ro), // sorted_payloads
                        bgl_entry(2, storage_ro), // tile_ranges
                        bgl_entry(3, storage_ro), // forward_out
                        bgl_entry(4, storage_ro), // dL_dC
                        bgl_entry(5, storage_rw), // projected_grad (atomic)
                        bgl_entry(6, uniform),    // params
                    ],
                });
        let composite_backward_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("composite-bwd-pl"),
                    bind_group_layouts: &[Some(&composite_backward_bgl)],
                    immediate_size: 0,
                });
        let composite_backward_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("composite-bwd-pipeline"),
                    layout: Some(&composite_backward_layout),
                    module: &composite_backward_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Project-backward pipeline ------------------------------------
        let project_backward_module =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("project_backward"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/project_backward.wgsl").into(),
                    ),
                });
        // 14 bindings: camera + 6 forward inputs + 1 grad input + 6 grad outputs.
        let pb_entries: [wgpu::BindGroupLayoutEntry; 14] = std::array::from_fn(|i| {
            let ty = match i {
                0 => uniform,
                1..=7 => storage_ro,
                _ => storage_rw,
            };
            bgl_entry(i as u32, ty)
        });
        let project_backward_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("project-bwd-bgl"),
                    entries: &pb_entries,
                });
        let project_backward_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("project-bwd-pl"),
                    bind_group_layouts: &[Some(&project_backward_bgl)],
                    immediate_size: 0,
                });
        let project_backward_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("project-bwd-pipeline"),
                    layout: Some(&project_backward_layout),
                    module: &project_backward_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Tonemap pipeline ---------------------------------------------
        let tonemap_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tonemap"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tonemap.wgsl").into()),
            });
        let tonemap_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tonemap-bgl"),
                entries: &[
                    bgl_entry(0, storage_ro),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    bgl_entry(2, uniform),
                ],
            });
        let tonemap_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap-pl"),
                bind_group_layouts: &[Some(&tonemap_bgl)],
                immediate_size: 0,
            });
        let tonemap_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("tonemap-pipeline"),
                    layout: Some(&tonemap_layout),
                    module: &tonemap_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            project_pipeline,
            project_bgl,
            composite_pipeline,
            composite_bgl,
            composite_backward_pipeline,
            composite_backward_bgl,
            project_backward_pipeline,
            project_backward_bgl,
            tonemap_pipeline,
            tonemap_bgl,
        }
    }

    /// Reinhard tonemap + sRGB encode + quantise to `Rgba8Unorm`. The
    /// destination must be a storage texture of the same `(width, height)`
    /// as the composite buffer; egui samples it as a plain `egui::Image`.
    pub fn tonemap(
        &self,
        ctx: &WgpuCtx,
        composite: &wgpu::Buffer,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let params = ctx.uniform_buffer(
            "tonemap-params",
            &TonemapParams {
                width,
                height,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap-bg"),
            layout: &self.tonemap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: composite.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(target),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tonemap-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tonemap-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tonemap_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Allocate the per-pixel `vec4` storage buffer used by the
    /// composite pass. Zeroed; caller can reuse the same buffer across
    /// iterations as long as the kernel rewrites every pixel.
    pub fn alloc_image(&self, ctx: &WgpuCtx, width: u32, height: u32) -> wgpu::Buffer {
        ctx.storage_buffer_zeroed("predicted-image", (width as u64) * (height as u64) * 16)
    }

    /// Run the composite pass — α-blend the sorted splat stream into a
    /// `width × height` `vec4<f32>` framebuffer. Caller produces
    /// `sorted_payloads` and `tile_ranges` via [`crate::tile_binner::TileBinner`].
    #[allow(clippy::too_many_arguments)]
    pub fn composite(
        &self,
        ctx: &WgpuCtx,
        projected: &wgpu::Buffer,
        sorted_payloads: &wgpu::Buffer,
        tile_ranges: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &TilingParams,
    ) {
        let tiles_x = params.tiles_x();
        let tiles_y = params.tiles_y();
        let comp_params = ctx.uniform_buffer(
            "composite-params",
            &CompositeParams {
                width: params.width,
                height: params.height,
                tile_size: params.tile_size,
                tiles_x,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite-bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: projected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sorted_payloads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_ranges.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: comp_params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(tiles_x, tiles_y, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Allocate a `ProjectedSplat` output buffer sized for `n` splats.
    pub fn alloc_projected(&self, ctx: &WgpuCtx, n: u32) -> wgpu::Buffer {
        let bytes = (n as u64).max(1) * std::mem::size_of::<ProjectedSplat>() as u64;
        ctx.storage_buffer_zeroed("projected", bytes)
    }

    /// Allocate a zeroed `ProjectedGrad` buffer sized for `n` splats.
    /// The backward rasteriser will use `atomic_add_f32` (CAS on
    /// `u32` bitcasts) to accumulate contributions from multiple
    /// pixels of multiple tiles into the same splat.
    pub fn alloc_projected_grad(&self, ctx: &WgpuCtx, n: u32) -> wgpu::Buffer {
        let bytes = (n as u64).max(1) * std::mem::size_of::<ProjectedGrad>() as u64;
        ctx.storage_buffer_zeroed("projected-grad", bytes)
    }

    /// Reset a `ProjectedGrad` buffer to all zeros. Use between
    /// training iterations instead of reallocating.
    pub fn zero_projected_grad(&self, ctx: &WgpuCtx, buf: &wgpu::Buffer, n: u32) {
        let bytes = (n as usize).max(1) * std::mem::size_of::<ProjectedGrad>();
        let zeros = vec![0u8; bytes];
        ctx.queue.write_buffer(buf, 0, &zeros);
    }

    /// Per-splat backward of the projection pass. Reads
    /// `projected_grad` (filled by `composite_backward`) and chains the
    /// 2D-state gradients back into the 3D splat parameter gradients,
    /// writing into the matching slots of `grads`.
    #[allow(clippy::too_many_arguments)]
    pub fn project_backward(
        &self,
        ctx: &WgpuCtx,
        splats: &GpuSplatBuffer,
        projected_grad: &wgpu::Buffer,
        camera: &CameraUniform,
        grads: &crate::splat_gpu::GradSplatBuffers,
    ) {
        let camera_buf = ctx.uniform_buffer("project-bwd-cam", camera);
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project-bwd-bg"),
            layout: &self.project_backward_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splats.positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splats.rotations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: splats.scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: splats.opacities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: splats.sh_dc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: splats.sh_rest.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: projected_grad.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: grads.d_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: grads.d_rotations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: grads.d_scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: grads.d_opacities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: grads.d_sh_dc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: grads.d_sh_rest.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("project-bwd-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project-bwd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_backward_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(camera.n_splats.div_ceil(64).max(1), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Reverse α-blend: given the forward output, a per-pixel dL/dC
    /// loss gradient, and the tile bins, accumulate per-splat 2D-space
    /// gradients into `projected_grad`. Caller must zero `projected_grad`
    /// beforehand and ensure `dL_dC` is sized `width × height` `vec4`s
    /// (`xyz` carries the gradient; `w` is unused / padding).
    #[allow(clippy::too_many_arguments)]
    pub fn composite_backward(
        &self,
        ctx: &WgpuCtx,
        projected: &wgpu::Buffer,
        sorted_payloads: &wgpu::Buffer,
        tile_ranges: &wgpu::Buffer,
        forward_out: &wgpu::Buffer,
        dl_dc: &wgpu::Buffer,
        projected_grad: &wgpu::Buffer,
        params: &TilingParams,
    ) {
        let tiles_x = params.tiles_x();
        let tiles_y = params.tiles_y();
        let comp_params = ctx.uniform_buffer(
            "composite-bwd-params",
            &CompositeParams {
                width: params.width,
                height: params.height,
                tile_size: params.tile_size,
                tiles_x,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite-bwd-bg"),
            layout: &self.composite_backward_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: projected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sorted_payloads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_ranges.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: forward_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dl_dc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: projected_grad.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: comp_params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite-bwd-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite-bwd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_backward_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(tiles_x, tiles_y, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    // (composite() defined above)

    /// Run the projection pass — writes one `ProjectedSplat` per splat
    /// in `splats` into `projected`. Caller chooses the camera; the
    /// uniform is built on the fly because it changes every iteration.
    pub fn project(
        &self,
        ctx: &WgpuCtx,
        splats: &GpuSplatBuffer,
        camera: &CameraUniform,
        projected: &wgpu::Buffer,
    ) {
        let camera_buf = ctx.uniform_buffer("camera", camera);
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project-bg"),
            layout: &self.project_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splats.positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splats.rotations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: splats.scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: splats.opacities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: splats.sh_dc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: splats.sh_rest.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: projected.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("project-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("project-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = camera.n_splats.div_ceil(64).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// CPU port of `project_gaussians.wgsl` for verification only. Math is
/// kept verbatim — any divergence between this and the kernel is a bug
/// in one of the two; tests pin both implementations against each other.
pub mod cpu_oracle {
    use super::{CameraUniform, ProjectedSplat};
    use crate::splat_store::SplatBuffer;
    use glam::{Mat3, Mat4, Vec3};

    fn quat_to_mat3(q: [f32; 4]) -> Mat3 {
        // SplatBuffer stores rotation as (w, x, y, z) — see splat_store.rs.
        let w = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];
        Mat3::from_cols(
            Vec3::new(
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ),
            Vec3::new(
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ),
            Vec3::new(
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// SH evaluation matching the WGSL `eval_sh` (DC + bands 1..3).
    fn eval_sh(sh_dc: [f32; 3], sh_rest: &[f32], dir: Vec3) -> Vec3 {
        const SH_C0: f32 = 0.2820948;
        const SH_C1: f32 = 0.488602;
        const SH_C2: [f32; 5] = [1.0925485, -1.0925485, 0.31539157, -1.0925485, 0.54627424];
        const SH_C3: [f32; 7] = [
            -0.5900436, 2.8906114, -0.4570458, 0.37317634, -0.4570458, 1.4453057, -0.5900436,
        ];

        let mut rgb = Vec3::from(sh_dc) * SH_C0;
        let (x, y, z) = (dir.x, dir.y, dir.z);
        let (xx, yy, zz) = (x * x, y * y, z * z);

        let b1 = [-SH_C1 * y, SH_C1 * z, -SH_C1 * x];
        let b2 = [
            SH_C2[0] * x * y,
            SH_C2[1] * y * z,
            SH_C2[2] * (2.0 * zz - xx - yy),
            SH_C2[3] * x * z,
            SH_C2[4] * (xx - yy),
        ];
        let b3 = [
            SH_C3[0] * y * (3.0 * xx - yy),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4.0 * zz - xx - yy),
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3[4] * x * (4.0 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3.0 * yy),
        ];

        // Planar layout: R[1..15], G[1..15], B[1..15] inside the 45-element slice.
        let mut acc = Vec3::ZERO;
        for i in 0..3 {
            acc.x += sh_rest[i] * b1[i];
            acc.y += sh_rest[15 + i] * b1[i];
            acc.z += sh_rest[30 + i] * b1[i];
        }
        for i in 0..5 {
            acc.x += sh_rest[3 + i] * b2[i];
            acc.y += sh_rest[18 + i] * b2[i];
            acc.z += sh_rest[33 + i] * b2[i];
        }
        for i in 0..7 {
            acc.x += sh_rest[8 + i] * b3[i];
            acc.y += sh_rest[23 + i] * b3[i];
            acc.z += sh_rest[38 + i] * b3[i];
        }
        rgb += acc;
        (rgb + Vec3::splat(0.5)).max(Vec3::ZERO)
    }

    /// Project one splat exactly as the WGSL kernel would.
    pub fn project_one(splats: &SplatBuffer, idx: usize, cam: &CameraUniform) -> ProjectedSplat {
        let view = Mat4::from_cols_array_2d(&cam.view);
        let p_world = splats.positions[idx];
        let p_cam = view.transform_point3(p_world);
        let depth = -p_cam.z;
        if depth < cam.near {
            return ProjectedSplat::default();
        }
        let t = Vec3::new(p_cam.x, p_cam.y, depth);

        let cx = cam.width * 0.5;
        let cy = cam.height * 0.5;
        let mean_xy = [cx + cam.focal * t.x / t.z, cy - cam.focal * t.y / t.z];

        let rot = quat_to_mat3(splats.rotations[idx]);
        let s = splats.scales[idx];
        let sigma = Vec3::new(s[0].exp(), s[1].exp(), s[2].exp());
        let s2 = sigma * sigma;
        let m_scaled = Mat3::from_cols(rot.x_axis * s2.x, rot.y_axis * s2.y, rot.z_axis * s2.z);
        let cov_w = m_scaled * rot.transpose();

        let w_mat = Mat3::from_cols(
            view.x_axis.truncate(),
            view.y_axis.truncate(),
            view.z_axis.truncate(),
        );
        let cov_c = w_mat * cov_w * w_mat.transpose();

        let inv_z = 1.0 / t.z;
        let inv_z2 = inv_z * inv_z;
        let f = cam.focal;
        let j_r0 = Vec3::new(f * inv_z, 0.0, -f * t.x * inv_z2);
        let j_r1 = Vec3::new(0.0, -f * inv_z, f * t.y * inv_z2);
        let cov_c_r0 = cov_c.x_axis * j_r0.x + cov_c.y_axis * j_r0.y + cov_c.z_axis * j_r0.z;
        let cov_c_r1 = cov_c.x_axis * j_r1.x + cov_c.y_axis * j_r1.y + cov_c.z_axis * j_r1.z;
        let mut s00 = cov_c_r0.dot(j_r0);
        let s01 = cov_c_r0.dot(j_r1);
        let mut s11 = cov_c_r1.dot(j_r1);

        s00 += 0.3;
        s11 += 0.3;
        let det = s00 * s11 - s01 * s01;
        if det <= 0.0 {
            return ProjectedSplat::default();
        }
        let inv_det = 1.0 / det;
        let conic = [s11 * inv_det, -s01 * inv_det, s00 * inv_det];

        let mid = 0.5 * (s00 + s11);
        let lambda = mid + (mid * mid - det).max(0.0).sqrt();
        let radius = (3.0 * lambda.sqrt()).ceil();

        let cam_pos = Vec3::from(cam.cam_pos);
        let to_cam = (cam_pos - p_world).normalize_or_zero();
        let rgb = eval_sh(
            splats.sh_dc[idx],
            &splats.sh_rest[idx * 45..(idx + 1) * 45],
            to_cam,
        );
        let opacity = sigmoid(splats.opacities[idx]);

        ProjectedSplat {
            mean_xy,
            depth,
            radius,
            conic,
            visible: 1.0,
            color: rgb.into(),
            opacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splat_store::SplatBuffer;

    fn unit_splat() -> SplatBuffer {
        let mut buf = SplatBuffer::default();
        buf.push_splat(
            Vec3::new(0.0, 0.0, -5.0), // pos in front of camera at origin
            [1.0, 0.0, 0.0, 0.0],      // identity quat (w, x, y, z)
            [0.0, 0.0, 0.0],           // log σ = 0 → σ = 1
            10.0,                      // very opaque (sigmoid ≈ 1)
            [0.5, 0.3, 0.1],           // SH DC
            &[0.0; 45],                // bands 1..3 = 0
        );
        buf
    }

    fn standard_camera(n: u32) -> CameraUniform {
        CameraUniform::from_pose(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_2, // 90° vertical FoV
            256,
            256,
            n,
        )
    }

    #[test]
    fn cpu_oracle_centers_on_axis() {
        let splats = unit_splat();
        let cam = standard_camera(splats.len() as u32);
        let p = cpu_oracle::project_one(&splats, 0, &cam);
        assert_eq!(p.visible, 1.0);
        // splat sits dead ahead → screen-centre.
        assert!(
            (p.mean_xy[0] - 128.0).abs() < 1e-3,
            "mean_x = {}",
            p.mean_xy[0]
        );
        assert!(
            (p.mean_xy[1] - 128.0).abs() < 1e-3,
            "mean_y = {}",
            p.mean_xy[1]
        );
        assert!((p.depth - 5.0).abs() < 1e-3, "depth = {}", p.depth);
        // Isotropic σ = 1 at depth 5, focal ≈ 128 → 2D σ ≈ 25.6 px (plus low-pass).
        // Radius ≈ 3·σ ≈ 77 px (matches Inria's rule of thumb).
        assert!(p.radius > 50.0 && p.radius < 100.0, "radius = {}", p.radius);
        // Conic is symmetric (isotropic input → diagonal conic, b ≈ 0).
        assert!(p.conic[1].abs() < 1e-3, "off-diagonal = {}", p.conic[1]);
        assert!(
            (p.conic[0] - p.conic[2]).abs() < 1e-3,
            "non-isotropic conic"
        );
    }

    #[test]
    fn composite_single_splat_renders_gaussian() {
        use crate::tile_binner::{TileBinner, TilingParams};

        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };

        // One opaque white splat centred on a 64×64 image (4×4 tiles).
        // Isotropic conic (1, 0, 1) → α = exp(-½·r²) where r is pixel
        // distance from the splat centre.
        let splat = ProjectedSplat {
            mean_xy: [32.5, 32.5],
            depth: 5.0,
            radius: 30.0,
            conic: [0.04, 0.0, 0.04], // σ = 5 → conic = 1/σ²
            visible: 1.0,
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
        };
        let projected = ctx.storage_buffer("composite-projected", &[splat]);
        let params = TilingParams {
            width: 64,
            height: 64,
            tile_size: 16,
            depth_max: 100.0,
        };

        let binner = TileBinner::new(&ctx);
        let res = binner.bin(&ctx, &projected, 1, &params);
        let raster = Rasterizer::new(&ctx);
        let img = raster.alloc_image(&ctx, 64, 64);
        raster.composite(
            &ctx,
            &projected,
            &res.sorted_payloads,
            &res.tile_ranges,
            &img,
            &params,
        );

        let pixels: Vec<[f32; 4]> = ctx.readback(&img, 64 * 64);

        let fetch = |x: u32, y: u32| pixels[(y * 64 + x) as usize];
        let centre = fetch(32, 32);
        let near = fetch(34, 34);
        let edge = fetch(0, 0);
        eprintln!("centre={:?} near={:?} edge={:?}", centre, near, edge);

        // Centre: full coverage (α near 1) and bright.
        assert!(centre[0] > 0.9, "centre RGB too dim: {centre:?}");
        assert!(centre[3] > 0.9, "centre alpha too low: {centre:?}");
        // Near (~3 px out, σ=5): still bright but slightly attenuated.
        assert!(near[0] > 0.6, "near RGB too dim: {near:?}");
        assert!(near[0] < centre[0] + 1e-3, "near should not exceed centre");
        // Far corner: ~45 px out, ≫ 3σ — should be near-zero.
        assert!(edge[0] < 0.05, "edge RGB too bright: {edge:?}");
        assert!(edge[3] < 0.05, "edge alpha too high: {edge:?}");

        // Monotonic falloff along the diagonal from the centre — α should
        // never exceed the centre's α.
        for d in 0..16u32 {
            let p = fetch(32 + d, 32 + d);
            assert!(
                p[3] <= centre[3] + 1e-3,
                "diag falloff broken at +{d}: {p:?}"
            );
        }
    }

    #[test]
    fn backward_accumulates_per_splat_gradients() {
        use crate::tile_binner::{TileBinner, TilingParams};

        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        // Single opaque white splat centred on 32×32 image (2×2 tiles).
        // Isotropic conic — covers ~half the image at σ = 5.
        let splat = ProjectedSplat {
            mean_xy: [16.0, 16.0],
            depth: 5.0,
            radius: 25.0,
            conic: [0.04, 0.0, 0.04],
            visible: 1.0,
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
        };
        let projected = ctx.storage_buffer("bwd-projected", &[splat]);
        let params = TilingParams {
            width: 32,
            height: 32,
            tile_size: 16,
            depth_max: 100.0,
        };
        let binner = TileBinner::new(&ctx);
        let res = binner.bin(&ctx, &projected, 1, &params);

        // Forward composite.
        let raster = Rasterizer::new(&ctx);
        let forward = raster.alloc_image(&ctx, 32, 32);
        raster.composite(
            &ctx,
            &projected,
            &res.sorted_payloads,
            &res.tile_ranges,
            &forward,
            &params,
        );

        // Constant per-pixel loss gradient dL/dC = (1, 1, 1, 0). Makes
        // the expected sign of dopacity and dconic.diag positive (any
        // brighter splat → larger loss).
        let dl_dc_vec: Vec<[f32; 4]> = vec![[1.0, 1.0, 1.0, 0.0]; 32 * 32];
        let dl_dc = ctx.storage_buffer("dl-dc", &dl_dc_vec);

        // Backward.
        let proj_grad = raster.alloc_projected_grad(&ctx, 1);
        raster.composite_backward(
            &ctx,
            &projected,
            &res.sorted_payloads,
            &res.tile_ranges,
            &forward,
            &dl_dc,
            &proj_grad,
            &params,
        );

        let grads: Vec<ProjectedGrad> = ctx.readback(&proj_grad, 1);
        let g = &grads[0];
        eprintln!(
            "dmean_op = {:?} dconic = {:?} dcolor = {:?}",
            g.dmean_opacity, g.dconic, g.dcolor
        );

        // Colour grad: every pixel contributes T_i · α_i · 1 → must be > 0.
        assert!(g.dcolor[0] > 0.1, "dcolor.r too small: {}", g.dcolor[0]);
        assert!(g.dcolor[1] > 0.1, "dcolor.g too small: {}", g.dcolor[1]);
        assert!(g.dcolor[2] > 0.1, "dcolor.b too small: {}", g.dcolor[2]);

        // Opacity grad (at dmean_opacity[2]): same sign as colour grad —
        // raising σ raises α raises C ↑ raises loss.
        assert!(
            g.dmean_opacity[2] > 0.01,
            "dopacity too small: {}",
            g.dmean_opacity[2]
        );

        // Conic-diagonal grads (a, c) should be negative: raising a or c
        // makes the Gaussian sharper / smaller-footprint → less area
        // covered → less colour → LOWER loss. Sign flipped.
        // But conic.x · dx² appears with a negative sign inside power
        // (power = −½(a dx² + …)), and dpower/da = −½ dx². With dx ≠ 0,
        // an increase in a *decreases* G → decreases α → decreases C.
        // So dC/da < 0 → grad < 0 under positive dL/dC.
        assert!(
            g.dconic[0] < -1e-4,
            "dconic.a should be negative: {}",
            g.dconic[0]
        );
        assert!(
            g.dconic[2] < -1e-4,
            "dconic.c should be negative: {}",
            g.dconic[2]
        );

        // 2D-mean gradient — splat is centred and the dL/dC field is
        // perfectly symmetric, so the per-axis sum should net out near 0.
        assert!(
            g.dmean_opacity[0].abs() < 0.5,
            "dmean.x not near 0: {}",
            g.dmean_opacity[0]
        );
        assert!(
            g.dmean_opacity[1].abs() < 0.5,
            "dmean.y not near 0: {}",
            g.dmean_opacity[1]
        );
    }

    #[test]
    fn wgsl_matches_cpu_oracle() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        let splats = unit_splat();
        let cam = standard_camera(splats.len() as u32);

        let gpu = GpuSplatBuffer::upload(&ctx, &splats);
        let raster = Rasterizer::new(&ctx);
        let projected = raster.alloc_projected(&ctx, gpu.n);
        raster.project(&ctx, &gpu, &cam, &projected);
        let gpu_out: Vec<ProjectedSplat> = ctx.readback(&projected, gpu.n as usize);

        let cpu_out = cpu_oracle::project_one(&splats, 0, &cam);
        let g = &gpu_out[0];

        assert_eq!(g.visible, 1.0);
        assert!(
            (g.mean_xy[0] - cpu_out.mean_xy[0]).abs() < 1e-2,
            "mean_x gpu={} cpu={}",
            g.mean_xy[0],
            cpu_out.mean_xy[0]
        );
        assert!((g.mean_xy[1] - cpu_out.mean_xy[1]).abs() < 1e-2);
        assert!((g.depth - cpu_out.depth).abs() < 1e-3);
        assert!((g.radius - cpu_out.radius).abs() < 1.0); // ceil() may differ by ±1 ulp
        for i in 0..3 {
            assert!(
                (g.conic[i] - cpu_out.conic[i]).abs() < 1e-4,
                "conic[{}] gpu={} cpu={}",
                i,
                g.conic[i],
                cpu_out.conic[i]
            );
            assert!(
                (g.color[i] - cpu_out.color[i]).abs() < 1e-3,
                "color[{}] gpu={} cpu={}",
                i,
                g.color[i],
                cpu_out.color[i]
            );
        }
        assert!((g.opacity - cpu_out.opacity).abs() < 1e-5);
    }
}
