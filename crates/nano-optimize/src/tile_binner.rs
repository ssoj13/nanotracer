//! Tile-based splat→pixel binning.
//!
//! Pipeline:
//!   1. `tile_count.wgsl`   — per-splat tile-touch count + atomic total
//!   2. CPU readback of the total → exact buffer sizing
//!   3. [`PrefixScan`] over the counts → per-splat write offset
//!   4. `tile_emit.wgsl`    — emit `(tile_id, depth) | splat_idx` pairs
//!   5. [`RadixSort`]       — sort by key (tile-id major, depth minor)
//!   6. `tile_ranges.wgsl`  — derive per-tile `[begin, end)` ranges
//!
//! Output is consumed by the rasteriser (A2.5): per-tile workgroups
//! walk their range of sorted entries front-to-back and α-composite.

use crate::gpu::WgpuCtx;
use crate::prefix_scan::PrefixScan;
use crate::radix_sort::RadixSort;
use bytemuck::{Pod, Zeroable};

/// Image-space binning configuration. `tile_size` is the workgroup size
/// the rasteriser will use later (typically 16 → 16×16 tiles).
#[derive(Debug, Clone, Copy)]
pub struct TilingParams {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    /// Maximum camera-space depth used by the depth-quantisation step.
    /// Splats further than this are clipped into the last bucket and
    /// will sort behind everything else.
    pub depth_max: f32,
}

impl TilingParams {
    pub fn tiles_x(&self) -> u32 {
        self.width.div_ceil(self.tile_size)
    }
    pub fn tiles_y(&self) -> u32 {
        self.height.div_ceil(self.tile_size)
    }
    pub fn num_tiles(&self) -> u32 {
        self.tiles_x() * self.tiles_y()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    n_splats: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmitParams {
    n_splats: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    depth_max: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RangesParams {
    total_pairs: u32,
    num_tiles: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Output of [`TileBinner::bin`]. All buffers live on the GPU; the
/// rasteriser binds them directly.
pub struct TileBinningResult {
    pub sorted_keys: wgpu::Buffer,
    pub sorted_payloads: wgpu::Buffer,
    pub tile_ranges: wgpu::Buffer,
    pub total_pairs: u32,
    pub num_tiles: u32,
}

pub struct TileBinner {
    count_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
    ranges_pipeline: wgpu::ComputePipeline,
    count_bgl: wgpu::BindGroupLayout,
    emit_bgl: wgpu::BindGroupLayout,
    ranges_bgl: wgpu::BindGroupLayout,
    scan: PrefixScan,
    radix: RadixSort,
}

impl TileBinner {
    pub fn new(ctx: &WgpuCtx) -> Self {
        let count_module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_count"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tile_count.wgsl").into()),
        });
        let emit_module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_emit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tile_emit.wgsl").into()),
        });
        let ranges_module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_ranges"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tile_ranges.wgsl").into()),
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

        let count_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("count-bgl"),
            entries: &[bgl(0, storage_ro), bgl(1, storage_rw), bgl(2, storage_rw), bgl(3, uniform)],
        });
        let emit_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("emit-bgl"),
            entries: &[
                bgl(0, storage_ro), bgl(1, storage_ro),
                bgl(2, storage_rw), bgl(3, storage_rw),
                bgl(4, uniform),
            ],
        });
        let ranges_bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ranges-bgl"),
            entries: &[bgl(0, storage_ro), bgl(1, storage_rw), bgl(2, uniform)],
        });

        let count_pipeline = build_pipeline(ctx, "count-pl", &count_bgl, &count_module);
        let emit_pipeline = build_pipeline(ctx, "emit-pl", &emit_bgl, &emit_module);
        let ranges_pipeline = build_pipeline(ctx, "ranges-pl", &ranges_bgl, &ranges_module);

        Self {
            count_pipeline,
            emit_pipeline,
            ranges_pipeline,
            count_bgl,
            emit_bgl,
            ranges_bgl,
            scan: PrefixScan::new(ctx),
            radix: RadixSort::new(ctx),
        }
    }

    /// Bin `n_splats` projected gaussians. Returns the sorted key/payload
    /// pair plus per-tile `[begin, end)` ranges, ready for rasterisation.
    pub fn bin(
        &self,
        ctx: &WgpuCtx,
        projected: &wgpu::Buffer,
        n_splats: u32,
        params: &TilingParams,
    ) -> TileBinningResult {
        let tiles_x = params.tiles_x();
        let tiles_y = params.tiles_y();
        let num_tiles = params.num_tiles();

        // 1. Count tiles per splat + atomic-sum into `total`.
        let counts = ctx.storage_buffer_zeroed("tile-counts", (n_splats.max(1) * 4) as u64);
        let total = ctx.storage_buffer_zeroed("tile-total", 4);
        let count_params = ctx.uniform_buffer("count-params", &CountParams {
            n_splats, tile_size: params.tile_size, tiles_x, tiles_y,
        });

        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tile-count-enc") },
        );
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("count-bg"),
                layout: &self.count_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: projected.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: total.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: count_params.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_splats.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        // 2. Read back total pairs.
        let total_vec: Vec<u32> = ctx.readback(&total, 1);
        let total_pairs = total_vec[0];

        // Empty scene fast path.
        if total_pairs == 0 {
            let sorted_keys = ctx.storage_buffer_zeroed("sorted-keys", 4);
            let sorted_payloads = ctx.storage_buffer_zeroed("sorted-payloads", 4);
            let tile_ranges = ctx.storage_buffer_zeroed("tile-ranges", (num_tiles * 8) as u64);
            return TileBinningResult {
                sorted_keys, sorted_payloads, tile_ranges,
                total_pairs: 0, num_tiles,
            };
        }

        // 3. Exclusive-scan counts → offsets.
        self.scan.scan(ctx, &counts, n_splats);

        // 4. Allocate key/payload buffers (and ping-pong twins).
        let keys_a = ctx.storage_buffer_zeroed("keys-a", (total_pairs as u64) * 4);
        let vals_a = ctx.storage_buffer_zeroed("vals-a", (total_pairs as u64) * 4);
        let keys_b = ctx.storage_buffer_zeroed("keys-b", (total_pairs as u64) * 4);
        let vals_b = ctx.storage_buffer_zeroed("vals-b", (total_pairs as u64) * 4);

        // 5. Emit (key, splat-idx) pairs.
        let emit_params = ctx.uniform_buffer("emit-params", &EmitParams {
            n_splats, tile_size: params.tile_size, tiles_x, tiles_y,
            depth_max: params.depth_max,
            _pad0: 0, _pad1: 0, _pad2: 0,
        });
        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tile-emit-enc") },
        );
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("emit-bg"),
                layout: &self.emit_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: projected.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: keys_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: vals_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: emit_params.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.emit_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_splats.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        // 6. Sort by key.
        self.radix.sort(ctx, &keys_a, &vals_a, &keys_b, &vals_b, total_pairs);

        // 7. Per-tile range derivation.
        let tile_ranges = ctx.storage_buffer_zeroed("tile-ranges", (num_tiles * 8) as u64);
        let ranges_params = ctx.uniform_buffer("ranges-params", &RangesParams {
            total_pairs, num_tiles, _pad0: 0, _pad1: 0,
        });
        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("tile-ranges-enc") },
        );
        {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ranges-bg"),
                layout: &self.ranges_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: keys_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: tile_ranges.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ranges_params.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ranges-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ranges_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(total_pairs.div_ceil(256), 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        TileBinningResult {
            sorted_keys: keys_a,
            sorted_payloads: vals_a,
            tile_ranges,
            total_pairs,
            num_tiles,
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
    bgl_: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl_)],
        immediate_size: 0,
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
    use crate::raster::ProjectedSplat;

    /// Mirror the WGSL `ProjectedSplat` Pod layout to upload synthetic
    /// records — bypasses the projection pass so we can pin the tile
    /// binning to known input geometry.
    fn upload_projected(ctx: &WgpuCtx, splats: &[ProjectedSplat]) -> wgpu::Buffer {
        ctx.storage_buffer("projected-test", splats)
    }

    /// Constructed splat: visible, centred at `(x, y)`, of pixel radius `r`.
    fn vis(x: f32, y: f32, depth: f32, r: f32) -> ProjectedSplat {
        ProjectedSplat {
            mean_xy: [x, y],
            depth,
            radius: r,
            conic: [1.0, 0.0, 1.0],
            visible: 1.0,
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
        }
    }

    #[test]
    fn four_splats_bin_to_expected_tiles() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        // 64×64 image, 16-pixel tiles → 4×4 = 16 tiles. Splats placed at
        // tile centres, radius 4 (sub-tile) so each touches exactly one tile.
        let splats = vec![
            vis(8.0,  8.0,  10.0, 4.0),   // tile (0, 0) → id 0
            vis(40.0, 8.0,  20.0, 4.0),   // tile (2, 0) → id 2
            vis(8.0,  56.0, 30.0, 4.0),   // tile (0, 3) → id 12
            vis(56.0, 56.0,  5.0, 4.0),   // tile (3, 3) → id 15
        ];
        let projected = upload_projected(&ctx, &splats);
        let params = TilingParams { width: 64, height: 64, tile_size: 16, depth_max: 100.0 };
        let binner = TileBinner::new(&ctx);
        let result = binner.bin(&ctx, &projected, splats.len() as u32, &params);

        assert_eq!(result.total_pairs, 4, "expected one tile per splat");
        assert_eq!(result.num_tiles, 16);

        let sorted_keys: Vec<u32> = ctx.readback(&result.sorted_keys, result.total_pairs as usize);
        let sorted_payloads: Vec<u32> = ctx.readback(&result.sorted_payloads, result.total_pairs as usize);
        let ranges: Vec<u32> = ctx.readback(&result.tile_ranges, (result.num_tiles * 2) as usize);

        // Tile-ids of sorted keys, in order.
        let tiles: Vec<u32> = sorted_keys.iter().map(|k| k >> 16).collect();
        assert_eq!(tiles, vec![0, 2, 12, 15], "tile order: {:?}", tiles);

        // Each tile range covers exactly one entry.
        let nonzero_ranges: Vec<(u32, u32, u32)> = (0..result.num_tiles)
            .filter_map(|tid| {
                let begin = ranges[(tid * 2) as usize];
                let end = ranges[(tid * 2 + 1) as usize];
                if end > begin { Some((tid, begin, end)) } else { None }
            })
            .collect();
        let want = [(0u32, 0u32, 1u32), (2, 1, 2), (12, 2, 3), (15, 3, 4)];
        assert_eq!(nonzero_ranges, want);

        // Payload at each slot points back to the right splat. Since each
        // splat touches exactly one tile and the four splat tile-ids are
        // already in ascending order, the payloads stay in input order.
        assert_eq!(sorted_payloads, vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn two_splats_same_tile_sorted_by_depth() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        // Both splats land in tile (1, 1) of a 64×64 / 16-tile image.
        // Tile id = 1·4 + 1 = 5. Splat 0 is far (depth 80), splat 1 close (depth 5).
        // After sort, splat 1 should come first (smaller depth → smaller key).
        let splats = vec![
            vis(24.0, 24.0, 80.0, 4.0),
            vis(24.0, 24.0,  5.0, 4.0),
        ];
        let projected = upload_projected(&ctx, &splats);
        let params = TilingParams { width: 64, height: 64, tile_size: 16, depth_max: 100.0 };
        let binner = TileBinner::new(&ctx);
        let result = binner.bin(&ctx, &projected, splats.len() as u32, &params);

        assert_eq!(result.total_pairs, 2);
        let sorted_keys: Vec<u32> = ctx.readback(&result.sorted_keys, 2);
        let sorted_payloads: Vec<u32> = ctx.readback(&result.sorted_payloads, 2);
        let tiles: Vec<u32> = sorted_keys.iter().map(|k| k >> 16).collect();
        assert_eq!(tiles, vec![5, 5]);
        assert_eq!(sorted_payloads, vec![1u32, 0u32], "payload should be depth-sorted");

        // Tile 5 range covers both entries.
        let ranges: Vec<u32> = ctx.readback(&result.tile_ranges, (result.num_tiles * 2) as usize);
        assert_eq!(ranges[10], 0, "tile 5 begin");
        assert_eq!(ranges[11], 2, "tile 5 end");
    }
}
