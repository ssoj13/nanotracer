//! GPU exclusive prefix-scan over a `u32` storage buffer.
//!
//! Three-level Hillis-Steele scan, supports any `n ≤ 256³ = 16,777,216`.
//! For each level the host dispatches a per-block scan (writes the
//! block's running total to the next level) then an add-offsets pass
//! that bakes the parent level's exclusive prefix into the children.
//!
//! Used by Phase A2.4 tile-binning to convert per-splat tile-touch
//! counts into write offsets. Standalone tests cover an "all ones"
//! input (must produce 0, 1, 2, …) and a random vector compared to a
//! CPU exclusive-scan reference.

use bytemuck::{Pod, Zeroable};

use crate::gpu::WgpuCtx;

/// Largest `n` accepted by [`PrefixScan::scan`]. `256^3` — three levels.
pub const MAX_SCAN_LEN: u32 = 256 * 256 * 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanParams {
    n: u32,
    _pad: [u32; 3], // std140 uniform alignment
}

pub struct PrefixScan {
    scan_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    scan_bgl: wgpu::BindGroupLayout,
    add_bgl: wgpu::BindGroupLayout,
}

impl PrefixScan {
    pub fn new(ctx: &WgpuCtx) -> Self {
        let scan_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scan_block"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scan_block.wgsl").into()),
            });
        let add_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scan_add_offsets"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/scan_add_offsets.wgsl").into(),
                ),
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

        let scan_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scan-bgl"),
                entries: &[bgl(0, storage_rw), bgl(1, storage_rw), bgl(2, uniform)],
            });
        let add_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("add-offsets-bgl"),
                entries: &[bgl(0, storage_rw), bgl(1, storage_ro), bgl(2, uniform)],
            });

        let scan_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scan-pl"),
                bind_group_layouts: &[Some(&scan_bgl)],
                immediate_size: 0,
            });
        let add_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("add-offsets-pl"),
                bind_group_layouts: &[Some(&add_bgl)],
                immediate_size: 0,
            });

        let scan_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scan-pipeline"),
                layout: Some(&scan_layout),
                module: &scan_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let add_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("add-offsets-pipeline"),
                layout: Some(&add_layout),
                module: &add_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            scan_pipeline,
            add_pipeline,
            scan_bgl,
            add_bgl,
        }
    }

    /// Exclusive-scan `data` in place. `n` is the logical element count
    /// (the buffer's allocation may be larger but only the first `n`
    /// entries are touched).
    pub fn scan(&self, ctx: &WgpuCtx, data: &wgpu::Buffer, n: u32) {
        assert!(
            n <= MAX_SCAN_LEN,
            "prefix-scan length {n} exceeds {MAX_SCAN_LEN}"
        );
        if n == 0 {
            return;
        }

        let l1 = n.div_ceil(256);
        let l2 = l1.div_ceil(256);
        // l3 is always 0 or 1: 256^3 fits in 256² × 256 = 65536 × 256 = full max.

        // Block-sum buffers (one per level above 0). For small `n` some
        // are sized 1 just to satisfy bind requirements.
        let sums_l1 = ctx.storage_buffer_zeroed("scan-sums-l1", (l1.max(1) * 4) as u64);
        let sums_l2 = ctx.storage_buffer_zeroed("scan-sums-l2", (l2.max(1) * 4) as u64);
        let dummy = ctx.storage_buffer_zeroed("scan-dummy", 4);

        let params_l0 = ctx.uniform_buffer("scan-params-l0", &ScanParams { n, _pad: [0; 3] });
        let params_l1 = ctx.uniform_buffer(
            "scan-params-l1",
            &ScanParams {
                n: l1,
                _pad: [0; 3],
            },
        );
        let params_l2 = ctx.uniform_buffer(
            "scan-params-l2",
            &ScanParams {
                n: l2,
                _pad: [0; 3],
            },
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scan-encoder"),
            });

        // L0: scan data → block sums to sums_l1.
        self.dispatch_scan(&mut encoder, ctx, data, &sums_l1, &params_l0, l1);

        if l1 > 1 {
            // L1: scan sums_l1 → block sums to sums_l2.
            self.dispatch_scan(&mut encoder, ctx, &sums_l1, &sums_l2, &params_l1, l2);

            if l2 > 1 {
                // L2: scan sums_l2 in-place (single block, sums go to dummy).
                self.dispatch_scan(&mut encoder, ctx, &sums_l2, &dummy, &params_l2, 1);
                // Add L2 offsets back into sums_l1.
                self.dispatch_add(&mut encoder, ctx, &sums_l1, &sums_l2, &params_l1, l1);
            }
            // Add L1 offsets back into data.
            self.dispatch_add(&mut encoder, ctx, data, &sums_l1, &params_l0, n);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch_scan(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &WgpuCtx,
        data: &wgpu::Buffer,
        sums: &wgpu::Buffer,
        params: &wgpu::Buffer,
        workgroups: u32,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-bg"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scan-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.scan_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }

    fn dispatch_add(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &WgpuCtx,
        data: &wgpu::Buffer,
        offsets: &wgpu::Buffer,
        params: &wgpu::Buffer,
        n: u32,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add-offsets-bg"),
            layout: &self.add_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add-offsets-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.add_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(256).max(1), 1, 1);
    }
}

fn bgl(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_scan(ctx: &WgpuCtx, input: &[u32]) -> Vec<u32> {
        let buf = ctx.storage_buffer("scan-test", input);
        let scan = PrefixScan::new(ctx);
        scan.scan(ctx, &buf, input.len() as u32);
        ctx.readback(&buf, input.len())
    }

    fn cpu_exclusive(xs: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(xs.len());
        let mut acc = 0u32;
        for x in xs {
            out.push(acc);
            acc = acc.wrapping_add(*x);
        }
        out
    }

    #[test]
    fn scan_all_ones_yields_indices() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        // Span more than one block (256) and verify offsets too.
        let n = 1000usize;
        let input = vec![1u32; n];
        let out = run_scan(&ctx, &input);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v as usize, i, "scan[{i}] = {v}, expected {i}");
        }
    }

    #[test]
    fn scan_random_matches_cpu_reference() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        // Cover the three-level path: > 256² requires the L2 dispatch.
        let n = 70_000usize;
        let mut rng_state = 0x12345678u32;
        let input: Vec<u32> = (0..n)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                rng_state % 17
            })
            .collect();
        let expected = cpu_exclusive(&input);
        let gpu = run_scan(&ctx, &input);
        assert_eq!(gpu.len(), expected.len());
        for i in 0..n {
            assert_eq!(gpu[i], expected[i], "mismatch at index {i}");
        }
    }
}
