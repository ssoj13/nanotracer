//! Per-splat emission of `(tile_key, splat_index)` pairs.
//!
//! `offsets[i]` is the exclusive scan over the per-splat tile counts
//! produced by `tile_count.wgsl` — the first write slot for splat `i`.
//! For each (tx, ty) inside the splat's screen-space tile bbox, write
//! a key encoding `(tile_id << 16) | depth_quantised_16bit` plus the
//! splat index as payload.
//!
//! Sorting on this key puts entries in `tile_id` major order, then
//! ascending depth — exactly what the rasteriser needs to α-blend
//! front-to-back within each tile.

struct Params {
    n_splats:  u32,
    tile_size: u32,
    tiles_x:   u32,
    tiles_y:   u32,
    depth_max: f32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

struct ProjectedSplat {
    mean_depth_radius: vec4<f32>,
    conic_visible:     vec4<f32>,
    color_opacity:     vec4<f32>,
};

@group(0) @binding(0) var<storage, read>          projected: array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read>          offsets:   array<u32>;
@group(0) @binding(2) var<storage, read_write>    keys:      array<u32>;
@group(0) @binding(3) var<storage, read_write>    payloads:  array<u32>;
@group(0) @binding(4) var<uniform>                params:    Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_splats) { return; }

    let p = projected[i];
    if (p.conic_visible.w == 0.0) { return; }

    let mean  = p.mean_depth_radius.xy;
    let depth = p.mean_depth_radius.z;
    let r     = p.mean_depth_radius.w;
    let ts = f32(params.tile_size);
    let min_tx = max(i32(floor((mean.x - r) / ts)), 0);
    let min_ty = max(i32(floor((mean.y - r) / ts)), 0);
    let max_tx = min(i32(floor((mean.x + r) / ts)), i32(params.tiles_x) - 1);
    let max_ty = min(i32(floor((mean.y + r) / ts)), i32(params.tiles_y) - 1);
    if (max_tx < min_tx || max_ty < min_ty) { return; }

    let d_norm = clamp(depth / params.depth_max, 0.0, 1.0);
    let d_u16 = u32(d_norm * 65535.0);

    var write_idx = offsets[i];
    for (var ty: i32 = min_ty; ty <= max_ty; ty = ty + 1) {
        for (var tx: i32 = min_tx; tx <= max_tx; tx = tx + 1) {
            let tile_id = u32(ty) * params.tiles_x + u32(tx);
            let key = (tile_id << 16u) | d_u16;
            keys[write_idx]     = key;
            payloads[write_idx] = i;
            write_idx = write_idx + 1u;
        }
    }
}
