//! Stable GPU radix sort over `u32` keys with a `u32` payload.
//!
//! Implementation: 32 passes of 1-bit-at-a-time stable split.
//! Per pass:
//!   1. `bit_predicate.wgsl` writes `predicate[i] = !bit(keys[i], b)`.
//!   2. [`PrefixScan`] turns `predicate` into an exclusive prefix-sum
//!      (count of "bit==0" elements strictly before `i`).
//!   3. `bit_total_zeros.wgsl` computes the grand total of zero-bit
//!      elements (`scan[n-1] + predicate[n-1]`).
//!   4. `bit_scatter.wgsl` writes each `(key, value)` to its destination:
//!      - bit==0 → `scan[i]`  (zeros pack to the front, in order)
//!      - bit==1 → `total + (i − scan[i])`  (ones pack after, in order)
//!
//! Each pass moves data from one ping-pong pair to the other. After
//! 32 passes (even count), the sorted output is back in the original
//! `keys_a / vals_a` pair.
//!
//! **Stable** — each pass strictly preserves the relative order of
//! elements with equal bit value, which is what makes the LSB-to-MSB
//! traversal converge to a globally sorted result.
//!
//! Cost is `O(32 · (predicate + scan + total + scatter))`, which is
//! several hundred dispatches for a multi-million-element input —
//! still milliseconds on a discrete GPU, and well within the budget
//! of one optimisation iteration. A faster byte-radix variant is
//! tracked as a follow-up performance optimisation.

use crate::gpu::WgpuCtx;
use crate::prefix_scan::PrefixScan;
use bytemuck::{Pod, Zeroable};

const KEY_BITS: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    bit: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct RadixSort {
    predicate_pipeline: wgpu::ComputePipeline,
    total_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    predicate_bgl: wgpu::BindGroupLayout,
    total_bgl: wgpu::BindGroupLayout,
    scatter_bgl: wgpu::BindGroupLayout,
    scan: PrefixScan,
}

impl RadixSort {
    pub fn new(ctx: &WgpuCtx) -> Self {
        let predicate_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bit_predicate"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bit_predicate.wgsl").into()),
            });
        let total_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bit_total_zeros"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/bit_total_zeros.wgsl").into(),
                ),
            });
        let scatter_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bit_scatter"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bit_scatter.wgsl").into()),
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

        let predicate_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("predicate-bgl"),
                entries: &[bgl(0, storage_ro), bgl(1, storage_rw), bgl(2, uniform)],
            });
        let total_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("total-bgl"),
                entries: &[
                    bgl(0, storage_ro),
                    bgl(1, storage_ro),
                    bgl(2, storage_rw),
                    bgl(3, uniform),
                ],
            });
        let scatter_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scatter-bgl"),
                entries: &[
                    bgl(0, storage_ro),
                    bgl(1, storage_ro),
                    bgl(2, storage_rw),
                    bgl(3, storage_rw),
                    bgl(4, storage_ro),
                    bgl(5, storage_ro),
                    bgl(6, uniform),
                ],
            });

        let predicate_pipeline =
            build_pipeline(ctx, "predicate-pl", &predicate_bgl, &predicate_module);
        let total_pipeline = build_pipeline(ctx, "total-pl", &total_bgl, &total_module);
        let scatter_pipeline = build_pipeline(ctx, "scatter-pl", &scatter_bgl, &scatter_module);

        Self {
            predicate_pipeline,
            total_pipeline,
            scatter_pipeline,
            predicate_bgl,
            total_bgl,
            scatter_bgl,
            scan: PrefixScan::new(ctx),
        }
    }

    /// In-place stable sort of `(keys_a, vals_a)`. `keys_b / vals_b` are
    /// scratch buffers of identical size used for ping-pong. After this
    /// call returns, the sorted output sits in `keys_a / vals_a` (32
    /// passes, even count → ends back where it started).
    pub fn sort(
        &self,
        ctx: &WgpuCtx,
        keys_a: &wgpu::Buffer,
        vals_a: &wgpu::Buffer,
        keys_b: &wgpu::Buffer,
        vals_b: &wgpu::Buffer,
        n: u32,
    ) {
        if n == 0 {
            return;
        }
        let predicate = ctx.storage_buffer_zeroed("radix-predicate", (n * 4) as u64);
        let total = ctx.storage_buffer_zeroed("radix-total", 4);

        for bit in 0..KEY_BITS {
            let (src_keys, src_vals, dst_keys, dst_vals) = if bit % 2 == 0 {
                (keys_a, vals_a, keys_b, vals_b)
            } else {
                (keys_b, vals_b, keys_a, vals_a)
            };
            let params = ctx.uniform_buffer(
                "radix-params",
                &Params {
                    n,
                    bit,
                    _pad0: 0,
                    _pad1: 0,
                },
            );

            // 1. Predicate.
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("radix-pred-enc"),
                });
            {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pred-bg"),
                    layout: &self.predicate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src_keys.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: predicate.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pred-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.predicate_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));

            // 2. Exclusive scan over predicate (in place).
            self.scan.scan(ctx, &predicate, n);

            // 3. Compute total zeros.
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("radix-total-enc"),
                });
            {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("total-bg"),
                    layout: &self.total_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src_keys.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: predicate.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: total.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("total-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.total_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));

            // 4. Scatter.
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("radix-scatter-enc"),
                });
            {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scatter-bg"),
                    layout: &self.scatter_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src_keys.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: src_vals.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dst_keys.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dst_vals.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: predicate.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: total.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("scatter-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));
        }
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

fn build_pipeline(
    ctx: &WgpuCtx,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_sort(ctx: &WgpuCtx, keys: &[u32], vals: &[u32]) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(keys.len(), vals.len());
        let n = keys.len() as u32;
        let keys_a = ctx.storage_buffer("k-a", keys);
        let vals_a = ctx.storage_buffer("v-a", vals);
        let keys_b = ctx.storage_buffer("k-b", &vec![0u32; keys.len()]);
        let vals_b = ctx.storage_buffer("v-b", &vec![0u32; vals.len()]);
        let rs = RadixSort::new(ctx);
        rs.sort(ctx, &keys_a, &vals_a, &keys_b, &vals_b, n);
        let sk = ctx.readback::<u32>(&keys_a, keys.len());
        let sv = ctx.readback::<u32>(&vals_a, vals.len());
        (sk, sv)
    }

    #[test]
    fn sort_small_array_stable() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        let keys = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
        let vals = vec![0u32, 1, 2, 3, 4, 5, 6, 7]; // original indices
        let (sk, sv) = run_sort(&ctx, &keys, &vals);
        let expected_keys = [1u32, 1, 2, 3, 4, 5, 6, 9];
        assert_eq!(sk, expected_keys);
        // Stable: for the two `1` keys, original indices 1 and 3 must
        // appear in that order in the sorted payload.
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 3);
    }

    #[test]
    fn sort_random_u32_with_payload() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        let n = 10_000usize;
        let mut rng = 0xdeadbeefu32;
        let keys: Vec<u32> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                rng
            })
            .collect();
        let vals: Vec<u32> = (0..n as u32).collect();
        let (sk, sv) = run_sort(&ctx, &keys, &vals);

        for i in 0..n {
            assert_eq!(sk[i], keys[sv[i] as usize], "payload mismatch at {i}");
        }
        for i in 1..n {
            assert!(
                sk[i - 1] <= sk[i],
                "not sorted at {i}: {} > {}",
                sk[i - 1],
                sk[i]
            );
        }
    }
}
