//! Per-tile front-to-back α-blending rasteriser.
//!
//! One workgroup per 16×16 tile, 256 threads (one per pixel). Each
//! tile owns a `[begin, end)` slice of the sorted (tile-id, depth) key
//! stream. Splats are loaded into workgroup-shared memory in chunks of
//! 256 — each thread loads one, then all threads composite all 256
//! against their own pixel before moving on. This amortises the global
//! storage reads over 256 pixels.
//!
//! Reference: graphdeco-inria `cuda_rasterizer/forward.cu` §1.5 + brush
//! `kernels/rasterize.wgsl`. The blend equation is the standard
//! front-to-back over operator:
//!
//!     α  = clamp(σ · exp(−½·xᵀ·Σ⁻¹·x), 0, αmax)
//!     C  ← C + T·α·c
//!     T  ← T · (1 − α)
//!
//! Output is an `rgba32f` storage buffer of `width × height` `vec4`s:
//! `xyz` carries the composed colour, `w` carries `1 − T` (i.e. coverage).

struct Params {
    width:     u32,
    height:    u32,
    tile_size: u32,
    tiles_x:   u32,
};

struct ProjectedSplat {
    mean_depth_radius: vec4<f32>,
    conic_visible:     vec4<f32>,
    color_opacity:     vec4<f32>,
};

@group(0) @binding(0) var<storage, read>          projected:        array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read>          sorted_payloads:  array<u32>;
@group(0) @binding(2) var<storage, read>          tile_ranges:      array<u32>;
@group(0) @binding(3) var<storage, read_write>    output:           array<vec4<f32>>;
@group(0) @binding(4) var<uniform>                params:           Params;

const CHUNK: u32 = 256u;
const TILE_PIXELS: u32 = 256u; // 16 × 16

var<workgroup> shared_mean:    array<vec2<f32>, CHUNK>;
var<workgroup> shared_conic:   array<vec3<f32>, CHUNK>;
var<workgroup> shared_color:   array<vec3<f32>, CHUNK>;
var<workgroup> shared_opacity: array<f32,       CHUNK>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)              tile3:    vec3<u32>,
    @builtin(local_invocation_id)       lid:      vec3<u32>,
    @builtin(local_invocation_index)    lidx:     u32,
) {
    let tx = tile3.x;
    let ty = tile3.y;
    let tile_id = ty * params.tiles_x + tx;
    let begin = tile_ranges[tile_id * 2u + 0u];
    let end   = tile_ranges[tile_id * 2u + 1u];

    let pixel_x = tx * params.tile_size + lid.x;
    let pixel_y = ty * params.tile_size + lid.y;
    let in_bounds = (pixel_x < params.width) && (pixel_y < params.height);
    let pix = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    var color = vec3<f32>(0.0);
    var T: f32 = 1.0;
    var done = !in_bounds;

    if (end > begin) {
        var idx = begin;
        while (idx < end) {
            let chunk_end = min(idx + CHUNK, end);
            let chunk_size = chunk_end - idx;

            // Cooperative load: thread `lidx` brings in the lidx-th splat
            // of the current chunk. Out-of-range threads load nothing.
            if (lidx < chunk_size) {
                let sidx = sorted_payloads[idx + lidx];
                let p = projected[sidx];
                shared_mean[lidx]    = p.mean_depth_radius.xy;
                shared_conic[lidx]   = p.conic_visible.xyz;
                shared_color[lidx]   = p.color_opacity.xyz;
                shared_opacity[lidx] = p.color_opacity.w;
            }
            workgroupBarrier();

            if (!done) {
                for (var k: u32 = 0u; k < chunk_size; k = k + 1u) {
                    let dx = pix - shared_mean[k];
                    let c = shared_conic[k];
                    // ½ · xᵀ · conic · x  (conic = [[a, b], [b, c_yy]])
                    let power = -0.5 * (
                        c.x * dx.x * dx.x
                      + 2.0 * c.y * dx.x * dx.y
                      + c.z * dx.y * dx.y
                    );
                    if (power > 0.0) { continue; }
                    let g = exp(power);
                    let alpha_raw = shared_opacity[k] * g;
                    let alpha = min(alpha_raw, 0.99);
                    if (alpha < (1.0 / 255.0)) { continue; }

                    color = color + T * alpha * shared_color[k];
                    T = T * (1.0 - alpha);
                    if (T < 1e-4) { done = true; break; }
                }
            }

            workgroupBarrier();
            idx = chunk_end;
        }
    }

    if (in_bounds) {
        let pidx = pixel_y * params.width + pixel_x;
        output[pidx] = vec4<f32>(color, 1.0 - T);
    }
}
