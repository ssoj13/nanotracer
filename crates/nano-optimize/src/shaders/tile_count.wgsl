//! Per-splat tile-touch count.
//!
//! For each visible projected splat, computes the screen-space tile
//! bounding box from `(mean_xy, radius)` and writes the number of
//! 16×16 tiles it touches into `counts[i]`. Also accumulates the
//! grand total of (splat, tile) pairs into `total[0]` via atomic add —
//! the host reads this after dispatch to size the keys/payloads
//! buffers exactly.

struct Params {
    n_splats:  u32,
    tile_size: u32,
    tiles_x:   u32,
    tiles_y:   u32,
};

struct ProjectedSplat {
    mean_depth_radius: vec4<f32>,
    conic_visible:     vec4<f32>,
    color_opacity:     vec4<f32>,
};

@group(0) @binding(0) var<storage, read>          projected: array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read_write>    counts:    array<u32>;
@group(0) @binding(2) var<storage, read_write>    total:     array<atomic<u32>>;
@group(0) @binding(3) var<uniform>                params:    Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_splats) { return; }

    let p = projected[i];
    if (p.conic_visible.w == 0.0) {
        counts[i] = 0u;
        return;
    }

    let mean = p.mean_depth_radius.xy;
    let r = p.mean_depth_radius.w;
    let ts = f32(params.tile_size);
    let min_tx = max(i32(floor((mean.x - r) / ts)), 0);
    let min_ty = max(i32(floor((mean.y - r) / ts)), 0);
    let max_tx = min(i32(floor((mean.x + r) / ts)), i32(params.tiles_x) - 1);
    let max_ty = min(i32(floor((mean.y + r) / ts)), i32(params.tiles_y) - 1);

    if (max_tx < min_tx || max_ty < min_ty) {
        counts[i] = 0u;
        return;
    }

    let count = u32((max_tx - min_tx + 1) * (max_ty - min_ty + 1));
    counts[i] = count;
    atomicAdd(&total[0], count);
}
