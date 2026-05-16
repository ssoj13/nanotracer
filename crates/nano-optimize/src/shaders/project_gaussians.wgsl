//! Per-splat world→screen projection.
//!
//! Each invocation handles one Gaussian splat. Outputs to `projected[i]`:
//!   • screen-space mean (pixel coords)
//!   • depth (positive, in front of camera)
//!   • screen-space radius (3σ extent of the 2D covariance)
//!   • 2D inverse covariance "conic" — three floats encoding [[a, b], [b, c]]
//!   • visibility flag (0 = culled, 1 = on-screen)
//!   • view-dependent SH-evaluated RGB
//!   • opacity in [0, 1] (sigmoid of the stored logit)
//!
//! Conventions match the renderer's pinhole model:
//!   • Right-handed view space: −Z forward, +X right, +Y up.
//!   • Image y axis flipped on emit (pixel.y goes from 0 at the top).
//!   • Focal length: `f = H / (2 · tan(fov_y / 2))`. Square pixels, so `fx == fy`.
//!
//! Reference algorithm: graphdeco-inria 3DGS forward.cu §1.2–1.4 + brush
//! `kernels/project_gaussians.wgsl`. Notation here follows Inria's
//! convention (the Jacobian uses positive-z depth via `t.z = −cam_z`).

struct Camera {
    view: mat4x4<f32>,
    cam_pos: vec3<f32>,
    near: f32,
    focal: f32,
    width: f32,
    height: f32,
    n_splats: u32,
};

// 48 bytes per splat (3 × vec4) — keep std430-aligned.
struct ProjectedSplat {
    // .xy = pixel-space mean, .z = depth (positive in front), .w = 3σ pixel radius
    mean_depth_radius: vec4<f32>,
    // .xyz = 2D conic (inv Σ_2D), .w = visible flag (0 / 1)
    conic_visible: vec4<f32>,
    // .xyz = RGB after SH evaluation + tonemap-free clamp, .w = sigmoid(opacity_logit)
    color_opacity: vec4<f32>,
};

@group(0) @binding(0) var<uniform>          camera:     Camera;
@group(0) @binding(1) var<storage, read>    positions:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>    rotations:  array<vec4<f32>>;   // (w, x, y, z)
@group(0) @binding(3) var<storage, read>    scales:     array<vec4<f32>>;   // log σ
@group(0) @binding(4) var<storage, read>    opacities:  array<f32>;         // logit
@group(0) @binding(5) var<storage, read>    sh_dc:      array<vec4<f32>>;   // band 0
@group(0) @binding(6) var<storage, read>    sh_rest:    array<vec4<f32>>;   // 12 vec4 per splat
@group(0) @binding(7) var<storage, read_write> projected: array<ProjectedSplat>;

// Real-SH basis constants — Condon–Shortley, identical set as nano-shaders.
const SH_C0: f32 = 0.2820948;
const SH_C1: f32 = 0.488602;
const SH_C2_0: f32 =  1.0925485;
const SH_C2_1: f32 = -1.0925485;
const SH_C2_2: f32 =  0.31539157;
const SH_C2_3: f32 = -1.0925485;
const SH_C2_4: f32 =  0.54627424;
const SH_C3_0: f32 = -0.5900436;
const SH_C3_1: f32 =  2.8906114;
const SH_C3_2: f32 = -0.4570458;
const SH_C3_3: f32 =  0.37317634;
const SH_C3_4: f32 = -0.4570458;
const SH_C3_5: f32 =  1.4453057;
const SH_C3_6: f32 = -0.5900436;

// Splats store rotation as (w, x, y, z). Convert to a 3×3 column-major
// rotation matrix matching `glam::Quat::to_mat3`.
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y*y + z*z),       2.0 * (x*y + w*z),         2.0 * (x*z - w*y)),
        vec3<f32>(      2.0 * (x*y - w*z), 1.0 - 2.0 * (x*x + z*z),         2.0 * (y*z + w*x)),
        vec3<f32>(      2.0 * (x*z + w*y),       2.0 * (y*z - w*x), 1.0 - 2.0 * (x*x + y*y)),
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Evaluate the splat's view-dependent colour in direction `dir` (from
// splat → camera, unit vector). Pulls 16 SH coefficients per channel:
// `sh_dc[i].xyz` is band 0, and the 45 floats in `sh_rest` are packed
// planar (R[1..15], G[1..15], B[1..15]) inside 12 vec4 slots per splat.
fn eval_sh(splat_idx: u32, dir: vec3<f32>) -> vec3<f32> {
    // Band 0.
    var rgb = SH_C0 * sh_dc[splat_idx].rgb;

    // Bands 1..3 — 15 coefficients per channel. Read planar.
    let base = splat_idx * 12u;
    var rest: array<f32, 48>;
    for (var k: u32 = 0u; k < 12u; k = k + 1u) {
        let v = sh_rest[base + k];
        rest[k * 4u + 0u] = v.x;
        rest[k * 4u + 1u] = v.y;
        rest[k * 4u + 2u] = v.z;
        rest[k * 4u + 3u] = v.w;
    }
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;

    // band 1 — indices 1..3
    let b1 = array<f32, 3>(-SH_C1 * y,  SH_C1 * z, -SH_C1 * x);
    // band 2 — indices 4..8
    let b2 = array<f32, 5>(
        SH_C2_0 * x * y,
        SH_C2_1 * y * z,
        SH_C2_2 * (2.0 * zz - xx - yy),
        SH_C2_3 * x * z,
        SH_C2_4 * (xx - yy),
    );
    // band 3 — indices 9..15
    let b3 = array<f32, 7>(
        SH_C3_0 *  y * (3.0 * xx - yy),
        SH_C3_1 *  x * y * z,
        SH_C3_2 *  y * (4.0 * zz - xx - yy),
        SH_C3_3 *  z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        SH_C3_4 *  x * (4.0 * zz - xx - yy),
        SH_C3_5 *  z * (xx - yy),
        SH_C3_6 *  x * (xx - 3.0 * yy),
    );
    // Apply per channel. Coefficients are planar (R[1..15], G[1..15], B[1..15]).
    // First 3 = band 1, next 5 = band 2, last 7 = band 3.
    var acc = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        acc.r = acc.r + rest[0u + i]      * b1[i];
        acc.g = acc.g + rest[15u + i]     * b1[i];
        acc.b = acc.b + rest[30u + i]     * b1[i];
    }
    for (var i: u32 = 0u; i < 5u; i = i + 1u) {
        acc.r = acc.r + rest[3u + i]      * b2[i];
        acc.g = acc.g + rest[18u + i]     * b2[i];
        acc.b = acc.b + rest[33u + i]     * b2[i];
    }
    for (var i: u32 = 0u; i < 7u; i = i + 1u) {
        acc.r = acc.r + rest[8u + i]      * b3[i];
        acc.g = acc.g + rest[23u + i]     * b3[i];
        acc.b = acc.b + rest[38u + i]     * b3[i];
    }
    rgb = rgb + acc;

    // 3DGS convention: viewers add a constant 0.5 offset before clamping
    // — the LSQ fit in nano-splat outputs (srgb - 0.5) so reconstruction
    // re-adds it here. Clamp to non-negative; over-1 values stay (will be
    // tonemapped or clamped by the rasteriser).
    return max(rgb + vec3<f32>(0.5), vec3<f32>(0.0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= camera.n_splats) {
        return;
    }

    // Default "culled" record — overwritten only when the splat is visible.
    var out: ProjectedSplat;
    out.mean_depth_radius = vec4<f32>(0.0);
    out.conic_visible = vec4<f32>(0.0);
    out.color_opacity = vec4<f32>(0.0);

    // World → camera.
    let p_world = positions[idx].xyz;
    let view4 = camera.view * vec4<f32>(p_world, 1.0);
    let cam_xyz = view4.xyz / view4.w;

    // Right-handed view: forward is −Z, so positive depth = −cam_z.
    let depth = -cam_xyz.z;
    if (depth < camera.near) {
        projected[idx] = out;
        return;
    }
    let t = vec3<f32>(cam_xyz.x, cam_xyz.y, depth);

    // Mean projection. Image y is flipped on emit (pixel.y = 0 at top).
    let cx = camera.width  * 0.5;
    let cy = camera.height * 0.5;
    let mean_xy = vec2<f32>(
        cx + (camera.focal * t.x) / t.z,
        cy - (camera.focal * t.y) / t.z,
    );

    // 3D world-space covariance Σ_3D = R · diag(σ²) · R^T.
    let R = quat_to_mat3(rotations[idx]);
    let sigma = exp(scales[idx].xyz);
    let s2 = sigma * sigma;
    // M = R · diag(σ²)  (column-scale R)
    let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
    // Σ = M · R^T  (compute as transpose then multiply)
    let cov_w = M * transpose(R);

    // World → camera covariance: Σ_c = W · Σ · W^T, where W = top-left 3×3 of view.
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let cov_c = W * cov_w * transpose(W);

    // Jacobian of the pinhole projection at `t` (right-handed). Image y
    // is flipped, so the y row picks up an extra sign relative to Inria.
    let inv_z = 1.0 / t.z;
    let inv_z2 = inv_z * inv_z;
    let fx = camera.focal;
    let fy = camera.focal;
    // J is 2×3. We need J · cov_c · J^T → 2×2. Express J via row vectors:
    //   J_r0 · cov_c · J_r0  →  Σ_2D[0,0]
    //   J_r0 · cov_c · J_r1  →  Σ_2D[0,1]
    //   J_r1 · cov_c · J_r1  →  Σ_2D[1,1]
    let j_r0 = vec3<f32>( fx * inv_z,        0.0,           -fx * t.x * inv_z2);
    let j_r1 = vec3<f32>(      0.0,         -fy * inv_z,     fy * t.y * inv_z2);
    let cov_c_r0 = cov_c[0] * j_r0.x + cov_c[1] * j_r0.y + cov_c[2] * j_r0.z;
    let cov_c_r1 = cov_c[0] * j_r1.x + cov_c[1] * j_r1.y + cov_c[2] * j_r1.z;
    var s00 = dot(cov_c_r0, j_r0);
    var s01 = dot(cov_c_r0, j_r1);
    var s11 = dot(cov_c_r1, j_r1);

    // Inria's 3-pixel low-pass filter ("dilation") — prevents aliasing
    // when a splat collapses to subpixel size.
    s00 = s00 + 0.3;
    s11 = s11 + 0.3;

    let det = s00 * s11 - s01 * s01;
    if (det <= 0.0) {
        projected[idx] = out;
        return;
    }
    let inv_det = 1.0 / det;
    // conic = inv(Σ_2D) — order (a, b, c) for the bilinear form
    // 0.5 · (a·dx² + 2·b·dx·dy + c·dy²).
    let conic = vec3<f32>(s11 * inv_det, -s01 * inv_det, s00 * inv_det);

    // 3σ pixel-space radius from the larger eigenvalue of Σ_2D.
    let mid = 0.5 * (s00 + s11);
    let lambda = mid + sqrt(max(mid * mid - det, 0.0));
    let radius = ceil(3.0 * sqrt(lambda));

    // Frustum cull: bail when no pixel of a 16×16 tile could be touched.
    let max_pixel = max(camera.width, camera.height) + radius;
    if (mean_xy.x < -radius || mean_xy.x >= camera.width + radius
     || mean_xy.y < -radius || mean_xy.y >= camera.height + radius
     || radius >= max_pixel) {
        projected[idx] = out;
        return;
    }

    // View-dependent SH colour. `view_dir` is splat → camera, unit.
    let to_cam = normalize(camera.cam_pos - p_world);
    let rgb = eval_sh(idx, to_cam);
    let alpha = sigmoid(opacities[idx]);

    out.mean_depth_radius = vec4<f32>(mean_xy, depth, radius);
    out.conic_visible = vec4<f32>(conic, 1.0);
    out.color_opacity = vec4<f32>(rgb, alpha);
    projected[idx] = out;
}
