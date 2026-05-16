//! Pure-GPU Adam optimiser + post-step constraint kernels.
//!
//! Replaces the per-iteration CPU readback / Adam / re-upload round-trip
//! that used to dominate the training loop. Three compute pipelines:
//!
//!   1. [`AdamGpu::step`]              — one f32 lane per thread, six
//!      dispatches per iteration (pos / rot / scale / opacity / sh_dc /
//!      sh_rest). Padded lanes (vec4 W slot, sh_rest's 3 trailing pads)
//!      carry zero gradient → moments stay at zero → update is `0`.
//!   2. [`AdamGpu::apply_constraints`] — quaternion re-norm + log-σ
//!      clamp + opacity-logit clamp. One thread per splat.
//!   3. [`AdamGpu::accumulate_grad`]   — per-splat L1(d_position)
//!      accumulator feeding the densify gate.
//!
//! `AdamState` (CPU) is still alive: its `t` counter drives bias
//! correction here, and at densify/prune checkpoints we readback the
//! GPU moments, let the existing CPU helpers mutate them in lockstep
//! with `SplatBuffer`, then re-upload.

use bytemuck::{Pod, Zeroable};

use crate::adam::AdamConfig;
use crate::gpu::WgpuCtx;
use crate::splat_gpu::{AdamMomentBuffers, GpuSplatBuffer, GradSplatBuffers};

const SH_REST_PADDED: u32 = 48;

/// Mirrors `shaders/adam_step.wgsl::AdamParams` (32 B, std140 uniform).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamParamsGpu {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    n: u32,
    _pad: u32,
}

/// Mirrors `shaders/apply_constraints.wgsl::ConstraintParams` (32 B).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConstraintParamsGpu {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    log_scale_min: f32,
    log_scale_max: f32,
    opacity_logit_max: f32,
    _pad3: f32,
}

/// Mirrors `shaders/accumulate_grad_pos.wgsl::AccParams` (16 B).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccParamsGpu {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Compiled compute pipelines + bind-group layouts. Build once per
/// training run and reuse for every iteration.
pub struct AdamGpu {
    step_pipeline: wgpu::ComputePipeline,
    step_bgl: wgpu::BindGroupLayout,
    constraint_pipeline: wgpu::ComputePipeline,
    constraint_bgl: wgpu::BindGroupLayout,
    accumulate_pipeline: wgpu::ComputePipeline,
    accumulate_bgl: wgpu::BindGroupLayout,
}

impl AdamGpu {
    pub fn new(ctx: &WgpuCtx) -> Self {
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

        // adam_step pipeline
        let step_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("adam_step"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/adam_step.wgsl").into()),
            });
        let step_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("adam-step-bgl"),
                entries: &[
                    bgl_entry(0, storage_rw),
                    bgl_entry(1, storage_ro),
                    bgl_entry(2, storage_rw),
                    bgl_entry(3, storage_rw),
                    bgl_entry(4, uniform),
                ],
            });
        let step_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("adam-step-pl"),
                bind_group_layouts: &[Some(&step_bgl)],
                immediate_size: 0,
            });
        let step_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adam-step-pipeline"),
                layout: Some(&step_layout),
                module: &step_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // apply_constraints pipeline
        let constraint_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("apply_constraints"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/apply_constraints.wgsl").into(),
                ),
            });
        let constraint_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("constraint-bgl"),
                    entries: &[
                        bgl_entry(0, storage_rw), // rotations
                        bgl_entry(1, storage_rw), // scales
                        bgl_entry(2, storage_rw), // opacities
                        bgl_entry(3, uniform),    // params
                    ],
                });
        let constraint_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("constraint-pl"),
                    bind_group_layouts: &[Some(&constraint_bgl)],
                    immediate_size: 0,
                });
        let constraint_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("constraint-pipeline"),
                    layout: Some(&constraint_layout),
                    module: &constraint_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // accumulate_grad_pos pipeline
        let acc_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("accumulate_grad_pos"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/accumulate_grad_pos.wgsl").into(),
                ),
            });
        let accumulate_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("accumulate-bgl"),
                    entries: &[
                        bgl_entry(0, storage_ro), // d_positions
                        bgl_entry(1, storage_rw), // grad_acc
                        bgl_entry(2, uniform),    // params
                    ],
                });
        let accumulate_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("accumulate-pl"),
                    bind_group_layouts: &[Some(&accumulate_bgl)],
                    immediate_size: 0,
                });
        let accumulate_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("accumulate-pipeline"),
                    layout: Some(&accumulate_layout),
                    module: &acc_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            step_pipeline,
            step_bgl,
            constraint_pipeline,
            constraint_bgl,
            accumulate_pipeline,
            accumulate_bgl,
        }
    }

    /// One Adam step over `n_scalars` f32 lanes. The kernel treats every
    /// bound buffer as an opaque `array<f32>`; bind whatever attribute
    /// (positions / scales / sh_rest / …) you want updated.
    ///
    /// `t` is the *new* step counter (after the bump). bc1/bc2 are
    /// computed as `1 − βᵗ`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        ctx: &WgpuCtx,
        params: &wgpu::Buffer,
        grads: &wgpu::Buffer,
        m: &wgpu::Buffer,
        v: &wgpu::Buffer,
        n_scalars: u32,
        lr: f32,
        t: u64,
        cfg: &AdamConfig,
    ) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adam-step-enc"),
            });
        self.encode_step(
            ctx,
            &mut encoder,
            params,
            grads,
            m,
            v,
            n_scalars,
            lr,
            t,
            cfg,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Record one Adam step into an existing `CommandEncoder` without
    /// submitting. Used by [`Self::step_all`] to coalesce all six
    /// per-attribute dispatches into a single submission — WebGPU's
    /// inter-submission storage-storage hazards are then a non-issue
    /// since every pass shares the same encoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_step(
        &self,
        ctx: &WgpuCtx,
        encoder: &mut wgpu::CommandEncoder,
        params: &wgpu::Buffer,
        grads: &wgpu::Buffer,
        m: &wgpu::Buffer,
        v: &wgpu::Buffer,
        n_scalars: u32,
        lr: f32,
        t: u64,
        cfg: &AdamConfig,
    ) {
        let bc1 = 1.0 - cfg.beta1.powi(t as i32);
        let bc2 = 1.0 - cfg.beta2.powi(t as i32);
        let uniform_buf = ctx.uniform_buffer(
            "adam-step-params",
            &AdamParamsGpu {
                lr,
                beta1: cfg.beta1,
                beta2: cfg.beta2,
                eps: cfg.eps,
                bc1,
                bc2,
                n: n_scalars,
                _pad: 0,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adam-step-bg"),
            layout: &self.step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adam-step-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.step_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n_scalars.div_ceil(256).max(1), 1, 1);
    }

    /// Convenience: step every per-attribute slab in one call. Positions
    /// use `cfg_pos.lr`; rot / scale / opacity / sh_dc / sh_rest share
    /// `cfg_attr.lr`. `t` is the new step counter for this iteration.
    ///
    /// All six dispatches are coalesced into a single `CommandEncoder`
    /// and one `queue.submit` — WebGPU leaves storage-storage hazards
    /// across separate submissions undefined, so the entire iteration
    /// stays in one submission to be portable.
    #[allow(clippy::too_many_arguments)]
    pub fn step_all(
        &self,
        ctx: &WgpuCtx,
        gpu_splats: &GpuSplatBuffer,
        grads: &GradSplatBuffers,
        moments: &AdamMomentBuffers,
        t: u64,
        cfg_pos: &AdamConfig,
        cfg_attr: &AdamConfig,
    ) {
        let n = gpu_splats.n;
        // Padded-vec4 attributes step 4·n f32 lanes. Padding's grad is 0
        // so the update is 0 — harmless.
        let n4 = n * 4;
        let n_rest = n * SH_REST_PADDED;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adam-step-all-enc"),
            });
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.positions,
            &grads.d_positions,
            &moments.m_pos,
            &moments.v_pos,
            n4,
            cfg_pos.lr,
            t,
            cfg_pos,
        );
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.rotations,
            &grads.d_rotations,
            &moments.m_rot,
            &moments.v_rot,
            n4,
            cfg_attr.lr,
            t,
            cfg_attr,
        );
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.scales,
            &grads.d_scales,
            &moments.m_scale,
            &moments.v_scale,
            n4,
            cfg_attr.lr,
            t,
            cfg_attr,
        );
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.opacities,
            &grads.d_opacities,
            &moments.m_op,
            &moments.v_op,
            n,
            cfg_attr.lr,
            t,
            cfg_attr,
        );
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.sh_dc,
            &grads.d_sh_dc,
            &moments.m_dc,
            &moments.v_dc,
            n4,
            cfg_attr.lr,
            t,
            cfg_attr,
        );
        self.encode_step(
            ctx,
            &mut encoder,
            &gpu_splats.sh_rest,
            &grads.d_sh_rest,
            &moments.m_rest,
            &moments.v_rest,
            n_rest,
            cfg_attr.lr,
            t,
            cfg_attr,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Quaternion re-norm + log-σ clamp + opacity-logit clamp. Run
    /// after `step_all` to keep splats physically valid.
    pub fn apply_constraints(
        &self,
        ctx: &WgpuCtx,
        gpu_splats: &GpuSplatBuffer,
        log_scale_min: f32,
        log_scale_max: f32,
        opacity_logit_max: f32,
    ) {
        let uniform_buf = ctx.uniform_buffer(
            "constraint-params",
            &ConstraintParamsGpu {
                n: gpu_splats.n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                log_scale_min,
                log_scale_max,
                opacity_logit_max,
                _pad3: 0.0,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("constraint-bg"),
            layout: &self.constraint_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_splats.rotations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_splats.scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_splats.opacities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("constraint-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("constraint-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.constraint_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gpu_splats.n.div_ceil(64).max(1), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Accumulate per-splat L1(d_position) into `grad_acc`.
    pub fn accumulate_grad(
        &self,
        ctx: &WgpuCtx,
        d_positions: &wgpu::Buffer,
        grad_acc: &wgpu::Buffer,
        n: u32,
    ) {
        let uniform_buf = ctx.uniform_buffer(
            "accumulate-params",
            &AccParamsGpu {
                n,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate-bg"),
            layout: &self.accumulate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: d_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_acc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accumulate-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64).max(1), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Clear `grad_acc` to zero. Cheap — one queue.write_buffer.
    pub fn zero_grad_acc(&self, ctx: &WgpuCtx, grad_acc: &wgpu::Buffer, n: u32) {
        let zeros = vec![0u8; (n as usize).max(1) * 4];
        ctx.queue.write_buffer(grad_acc, 0, &zeros);
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
