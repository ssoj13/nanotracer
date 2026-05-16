//! Per-tile backward α-blend.
//!
//! Forward composites splats in tile order (already sorted front-to-back)
//! and produces, per pixel:
//!   • final colour C
//!   • final transmittance T  (stored as 1−T in the alpha channel)
//!
//! Backward walks the **same** sorted slice but in reverse, maintaining
//! T and an accumulator S = Σ_{j>i} T_j α_j c_j. For each splat i the
//! per-pixel gradient terms are:
//!
//!     T_i        = T / (1 − α_i)
//!     dL/dc_i    = dL/dC · T_i · α_i                     (vec3)
//!     dL/dα_i    = dL/dC · (T_i · c_i − S / (1 − α_i))   (scalar)
//!
//! Chaining α_i back to splat parameters (α_raw = σ · exp(−½ xᵀΣ⁻¹x),
//! clamped to 0.99):
//!
//!     ∂α/∂σ           = G                  (with saturation gate)
//!     ∂α/∂conic.{a,b,c} = −½ σ G · {dx², 2·dx·dy, dy²}
//!     ∂α/∂mean        = σ G · (conic · dx)        (rederivation of ∂power/∂mean)
//!
//! Per-pixel contributions accumulate into `projected_grad[splat]` via
//! a CAS-based `atomic_add_f32` because many pixels in many tiles can
//! write to the same splat's gradient slot.

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

@group(0) @binding(0) var<storage, read>          projected:       array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read>          sorted_payloads: array<u32>;
@group(0) @binding(2) var<storage, read>          tile_ranges:     array<u32>;
@group(0) @binding(3) var<storage, read>          forward_out:     array<vec4<f32>>;   // (rgb, 1-T)
@group(0) @binding(4) var<storage, read>          dL_dC:           array<vec4<f32>>;   // .xyz = dL/dC, .w unused
@group(0) @binding(5) var<storage, read_write>    projected_grad:  array<atomic<u32>>; // 12 u32 / splat (f32 bits)
@group(0) @binding(6) var<uniform>                params:          Params;

const CHUNK: u32 = 256u;
const ALPHA_MIN: f32 = 1.0 / 255.0;
const ALPHA_MAX: f32 = 0.99;
const ONE_MINUS_ALPHA_EPS: f32 = 1e-4;

var<workgroup> shared_mean:    array<vec2<f32>, CHUNK>;
var<workgroup> shared_conic:   array<vec3<f32>, CHUNK>;
var<workgroup> shared_color:   array<vec3<f32>, CHUNK>;
var<workgroup> shared_opacity: array<f32,       CHUNK>;
var<workgroup> shared_splat:   array<u32,       CHUNK>;

// CAS-based atomic-add-f32. WGSL lacks native float atomics so we
// reinterpret the u32 storage as f32 bits and retry until the
// compare-and-swap wins.
fn atomic_add_f32(target_idx: u32, value: f32) {
    var old_bits = atomicLoad(&projected_grad[target_idx]);
    loop {
        let new_value = bitcast<f32>(old_bits) + value;
        let new_bits  = bitcast<u32>(new_value);
        let result = atomicCompareExchangeWeak(&projected_grad[target_idx], old_bits, new_bits);
        if (result.exchanged) { break; }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)              tile3: vec3<u32>,
    @builtin(local_invocation_id)       lid:   vec3<u32>,
    @builtin(local_invocation_index)    lidx:  u32,
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

    // Per-pixel forward state (final transmittance + loss gradient).
    var T: f32 = 1.0;
    if (in_bounds) {
        T = 1.0 - forward_out[pixel_y * params.width + pixel_x].w;
    }
    var grad_C = vec3<f32>(0.0);
    if (in_bounds) {
        grad_C = dL_dC[pixel_y * params.width + pixel_x].xyz;
    }
    var S = vec3<f32>(0.0);
    var done = !in_bounds || (end <= begin);

    if (!done) {
        // Walk the tile's sorted slice in REVERSE in chunks of 256.
        // Chunks are filled cooperatively then processed in-thread, just
        // like the forward pass — except chunk boundaries also reverse.
        var rem = end - begin;
        while (rem > 0u) {
            let take = min(rem, CHUNK);
            // The kth slot in our reversed chunk corresponds to
            // global index (end - 1 - (CHUNK - 1 - k)) when full, or
            // adjusted when `take < CHUNK`. Easier: load reversed.
            // Cooperative load: thread lidx fetches the (take-1-lidx)-th
            // entry from the end of `rem` — i.e. ascending by lidx we
            // store from newest to oldest.
            let chunk_start = begin + (rem - take);
            if (lidx < take) {
                let global_idx = chunk_start + lidx; // ascending forward index
                let sidx = sorted_payloads[global_idx];
                let p = projected[sidx];
                shared_mean[lidx]    = p.mean_depth_radius.xy;
                shared_conic[lidx]   = p.conic_visible.xyz;
                shared_color[lidx]   = p.color_opacity.xyz;
                shared_opacity[lidx] = p.color_opacity.w;
                shared_splat[lidx]   = sidx;
            }
            workgroupBarrier();

            if (!done) {
                // Iterate in reverse within the loaded chunk.
                for (var k_signed: i32 = i32(take) - 1; k_signed >= 0; k_signed = k_signed - 1) {
                    let k = u32(k_signed);
                    let mean    = shared_mean[k];
                    let conic   = shared_conic[k];
                    let color   = shared_color[k];
                    let sigma   = shared_opacity[k];
                    let sidx    = shared_splat[k];

                    let dx = pix - mean;
                    let power = -0.5 * (
                        conic.x * dx.x * dx.x
                      + 2.0 * conic.y * dx.x * dx.y
                      + conic.z * dx.y * dx.y
                    );
                    if (power > 0.0) { continue; }
                    let g = exp(power);
                    let alpha_raw = sigma * g;
                    let alpha = min(alpha_raw, ALPHA_MAX);
                    if (alpha < ALPHA_MIN) { continue; }

                    let one_minus_alpha = max(1.0 - alpha, ONE_MINUS_ALPHA_EPS);
                    T = T / one_minus_alpha;          // T was T_{i+1}, now T_i
                    let contrib = T * alpha * color;  // = T_i α_i c_i

                    // Gradient wrt colour and α at this splat (per pixel).
                    let dC_dc = T * alpha;            // scalar
                    let grad_color_pix = grad_C * dC_dc;

                    let dC_dalpha = T * color - S / one_minus_alpha;
                    let grad_alpha_pix = dot(grad_C, dC_dalpha);

                    // Saturation gate: clamp at 0.99 zeros out the
                    // α-raw gradient there. Below clamp we propagate.
                    let saturation = select(0.0, 1.0, alpha_raw < ALPHA_MAX);
                    let grad_alpha_raw = grad_alpha_pix * saturation;

                    // Chain into σ (opacity), conic, and 2D mean.
                    let grad_sigma_pix = grad_alpha_raw * g;
                    // dα_raw/dG = σ, dG/dpower = G  ; power = −½ xᵀΣ⁻¹x
                    let grad_power_pix = grad_alpha_raw * sigma * g;
                    // power = −½(a dx² + 2b dx·dy + c dy²)
                    let grad_a = grad_power_pix * -0.5 * (dx.x * dx.x);
                    let grad_b = grad_power_pix * -0.5 * (2.0 * dx.x * dx.y);
                    let grad_c = grad_power_pix * -0.5 * (dx.y * dx.y);
                    // ∂power/∂mean.x = a·dx + b·dy   (after × −0.5 cancels with chain → sign flips)
                    //                = ½ · 2 · (a dx + b dy)   — kept on a single line for clarity:
                    // d power / d mean.x = -0.5 · (2 a dx · (-1) + 2 b dy · (-1)) = (a dx + b dy)
                    let grad_mean_x = grad_power_pix * (conic.x * dx.x + conic.y * dx.y);
                    let grad_mean_y = grad_power_pix * (conic.y * dx.x + conic.z * dx.y);

                    // Atomic-accumulate into projected_grad[sidx].
                    let base = sidx * 12u;
                    atomic_add_f32(base + 0u, grad_mean_x);
                    atomic_add_f32(base + 1u, grad_mean_y);
                    atomic_add_f32(base + 2u, grad_sigma_pix);
                    // pad at +3
                    atomic_add_f32(base + 4u, grad_a);
                    atomic_add_f32(base + 5u, grad_b);
                    atomic_add_f32(base + 6u, grad_c);
                    // pad at +7
                    atomic_add_f32(base + 8u,  grad_color_pix.x);
                    atomic_add_f32(base + 9u,  grad_color_pix.y);
                    atomic_add_f32(base + 10u, grad_color_pix.z);
                    // pad at +11

                    // Update reverse-walk state for the next (earlier) splat.
                    S = S + contrib;
                }
            }
            workgroupBarrier();
            rem = rem - take;
        }
    }
}
