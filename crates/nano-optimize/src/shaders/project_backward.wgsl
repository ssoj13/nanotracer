//! Per-splat backward of `project_gaussians.wgsl`.
//!
//! Receives 2D-space gradients (dL/dmean, dL/dconic, dL/dcolor, dL/dopacity)
//! from `rasterize_backward.wgsl` and chains them back to 3D-space splat
//! parameters: dL/d(position, rotation, log-scale, opacity_logit, SH_DC, SH_rest).
//!
//! One thread per splat — no atomics on the output gradient buffers.
//!
//! Chain rule summary:
//!   • mean_2D  ← projection of t = view · p_world  (∂mean/∂t known analytically)
//!   • t        ← view-rotation W applied to p_world
//!   • conic    = inv(Σ_2D + filter·I)             (∂conic/∂Σ_2D = −conic · • · conic)
//!   • Σ_2D     = (J·W) · Σ_3D · (J·W)ᵀ            (∂Σ_2D/∂Σ_3D = sandwich)
//!   • Σ_3D     = R · diag(σ²) · Rᵀ                (∂/∂R, ∂/∂σ via matrix calculus)
//!   • R        ← quaternion (w, x, y, z)          (closed-form polynomial gradients)
//!   • σ_i      = exp(log_σ_i)                     (multiplies grad by σ_i)
//!   • α_raw σ in raster   = sigmoid(opacity_logit) (×σ(1−σ) chain)
//!   • color   = max(SH(dir) + 0.5, 0)             (basis values; clamp gradient ignored
//!                                                  when colour is non-negative, the
//!                                                  common case — a known limitation)
//!
//! The SH `dir` itself depends on `p_world` (it is `normalize(cam_pos − p_world)`),
//! but the dependence is weak compared with the direct projection effect on
//! mean_xy and dropping it gives a known minor approximation, matching brush.

struct Camera {
    view:     mat4x4<f32>,
    cam_pos:  vec3<f32>,
    near:     f32,
    focal:    f32,
    width:    f32,
    height:   f32,
    n_splats: u32,
};

@group(0) @binding(0)  var<uniform>          camera:         Camera;
@group(0) @binding(1)  var<storage, read>    positions:      array<vec4<f32>>;
@group(0) @binding(2)  var<storage, read>    rotations:      array<vec4<f32>>;   // (w, x, y, z)
@group(0) @binding(3)  var<storage, read>    scales:         array<vec4<f32>>;   // log-σ
@group(0) @binding(4)  var<storage, read>    opacities:      array<f32>;         // logit
@group(0) @binding(5)  var<storage, read>    sh_dc:          array<vec4<f32>>;
@group(0) @binding(6)  var<storage, read>    sh_rest:        array<vec4<f32>>;   // 12 vec4 / splat
@group(0) @binding(7)  var<storage, read>    projected_grad: array<vec4<f32>>;   // 3 vec4 / splat
@group(0) @binding(8)  var<storage, read_write> d_positions: array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read_write> d_rotations: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> d_scales:    array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> d_opacities: array<f32>;
@group(0) @binding(12) var<storage, read_write> d_sh_dc:     array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> d_sh_rest:   array<vec4<f32>>;

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

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z),       2.0*(x*y + w*z),         2.0*(x*z - w*y)),
        vec3<f32>(      2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z),         2.0*(y*z + w*x)),
        vec3<f32>(      2.0*(x*z + w*y),       2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y)),
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= camera.n_splats) { return; }

    // Forward intermediates ------------------------------------------------
    let p_world = positions[idx].xyz;
    let view4 = camera.view * vec4<f32>(p_world, 1.0);
    let cam_xyz = view4.xyz / view4.w;
    let depth = -cam_xyz.z;
    if (depth < camera.near) {
        // Same cull as forward — zero the output grads so the caller
        // sees a well-defined "no gradient" state for invisible splats.
        d_positions[idx] = vec4<f32>(0.0);
        d_rotations[idx] = vec4<f32>(0.0);
        d_scales[idx]    = vec4<f32>(0.0);
        d_opacities[idx] = 0.0;
        d_sh_dc[idx]     = vec4<f32>(0.0);
        let base = idx * 12u;
        for (var i: u32 = 0u; i < 12u; i = i + 1u) {
            d_sh_rest[base + i] = vec4<f32>(0.0);
        }
        return;
    }
    let t = vec3<f32>(cam_xyz.x, cam_xyz.y, depth);
    let inv_z = 1.0 / t.z;
    let inv_z2 = inv_z * inv_z;
    let f = camera.focal;

    let R = quat_to_mat3(rotations[idx]);
    let log_sigma = scales[idx].xyz;
    let sigma = exp(log_sigma);
    let s2 = sigma * sigma;
    let M = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
    let cov_world = M * transpose(R);

    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let cov_cam = W * cov_world * transpose(W);

    // J rows (matches project_gaussians.wgsl).
    let j_r0 = vec3<f32>( f * inv_z,        0.0,           -f * t.x * inv_z2);
    let j_r1 = vec3<f32>(      0.0,        -f * inv_z,      f * t.y * inv_z2);
    let cov_c_r0 = cov_cam[0] * j_r0.x + cov_cam[1] * j_r0.y + cov_cam[2] * j_r0.z;
    let cov_c_r1 = cov_cam[0] * j_r1.x + cov_cam[1] * j_r1.y + cov_cam[2] * j_r1.z;
    var s00 = dot(cov_c_r0, j_r0) + 0.3;
    let s01 = dot(cov_c_r0, j_r1);
    var s11 = dot(cov_c_r1, j_r1) + 0.3;

    let det = s00 * s11 - s01 * s01;
    if (det <= 0.0) {
        d_positions[idx] = vec4<f32>(0.0);
        d_rotations[idx] = vec4<f32>(0.0);
        d_scales[idx]    = vec4<f32>(0.0);
        d_opacities[idx] = 0.0;
        d_sh_dc[idx]     = vec4<f32>(0.0);
        let base = idx * 12u;
        for (var i: u32 = 0u; i < 12u; i = i + 1u) {
            d_sh_rest[base + i] = vec4<f32>(0.0);
        }
        return;
    }
    let inv_det = 1.0 / det;
    let conic_a = s11 * inv_det;
    let conic_b = -s01 * inv_det;
    let conic_c = s00 * inv_det;

    // 2D-state gradient (input) -------------------------------------------
    let pg0 = projected_grad[idx * 3u + 0u];        // dmean.xy + dopacity (scalar) + pad
    let pg1 = projected_grad[idx * 3u + 1u];        // dconic.xyz + pad
    let pg2 = projected_grad[idx * 3u + 2u];        // dcolor.xyz + pad
    let dmean_pix = pg0.xy;
    let dopacity_sigmoid = pg0.z;
    let dconic = pg1.xyz;
    let dcolor = pg2.xyz;

    // 1. Opacity logit gradient via sigmoid'(x) = σ(1−σ). The "σ" used
    //    in the raster (color_opacity.w) is already sigmoid(logit).
    let opacity_sigmoid = sigmoid(opacities[idx]);
    let d_opacity_logit = dopacity_sigmoid * opacity_sigmoid * (1.0 - opacity_sigmoid);

    // 2. SH coefficient gradients ------------------------------------------
    let to_cam = normalize(camera.cam_pos - p_world);
    let x = to_cam.x; let y = to_cam.y; let z = to_cam.z;
    let xx = x * x; let yy = y * y; let zz = z * z;
    // DC contribution: ∂(C_k)/∂(sh_dc_k) = SH_C0 (same for all channels).
    d_sh_dc[idx] = vec4<f32>(dcolor * SH_C0, 0.0);

    // Bands 1..3 basis vector — 15 entries per channel.
    var basis: array<f32, 15>;
    basis[0]  = -SH_C1 * y;
    basis[1]  =  SH_C1 * z;
    basis[2]  = -SH_C1 * x;
    basis[3]  =  SH_C2_0 * x * y;
    basis[4]  =  SH_C2_1 * y * z;
    basis[5]  =  SH_C2_2 * (2.0 * zz - xx - yy);
    basis[6]  =  SH_C2_3 * x * z;
    basis[7]  =  SH_C2_4 * (xx - yy);
    basis[8]  =  SH_C3_0 * y * (3.0 * xx - yy);
    basis[9]  =  SH_C3_1 * x * y * z;
    basis[10] =  SH_C3_2 * y * (4.0 * zz - xx - yy);
    basis[11] =  SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    basis[12] =  SH_C3_4 * x * (4.0 * zz - xx - yy);
    basis[13] =  SH_C3_5 * z * (xx - yy);
    basis[14] =  SH_C3_6 * x * (xx - 3.0 * yy);
    // 45 grads packed planar (R[1..15], G[1..15], B[1..15]) into 12 vec4.
    var rest_grads: array<f32, 48>;
    for (var k: u32 = 0u; k < 48u; k = k + 1u) { rest_grads[k] = 0.0; }
    for (var k: u32 = 0u; k < 15u; k = k + 1u) {
        rest_grads[k]        = dcolor.x * basis[k];
        rest_grads[15u + k]  = dcolor.y * basis[k];
        rest_grads[30u + k]  = dcolor.z * basis[k];
    }
    let base_sh = idx * 12u;
    for (var v: u32 = 0u; v < 12u; v = v + 1u) {
        d_sh_rest[base_sh + v] = vec4<f32>(
            rest_grads[v * 4u + 0u],
            rest_grads[v * 4u + 1u],
            rest_grads[v * 4u + 2u],
            rest_grads[v * 4u + 3u],
        );
    }

    // 3. Backward through conic → Σ_2D → Σ_3D → R, σ ---------------------
    // dL/dΣ_2D = −conic · dL/dconic · conic (Σ_2D symmetric 2×2).
    let cm = mat2x2<f32>(
        vec2<f32>(conic_a, conic_b),
        vec2<f32>(conic_b, conic_c),
    );
    let dconic_mat = mat2x2<f32>(
        vec2<f32>(dconic.x,         dconic.y * 0.5),
        vec2<f32>(dconic.y * 0.5,   dconic.z),
    );
    // Symmetry care: dconic.y is the gradient of the OFF-diagonal element
    // b that appears twice in the symmetric matrix → halve before sandwich,
    // double after — the conventional Σ⁻¹ derivative for symmetric inputs.
    let dsigma2d_full = -1.0 * (cm * dconic_mat * cm);
    var ds00 = dsigma2d_full[0][0];
    var ds11 = dsigma2d_full[1][1];
    var ds01 = (dsigma2d_full[0][1] + dsigma2d_full[1][0]);  // symmetric → add both off-diags

    // 4. Σ_2D = J·cov_cam·Jᵀ → dL/dcov_cam (3×3 symmetric).
    // Let M = J·W (2×3). Σ_2D = M·Σ_3D·Mᵀ. dL/dΣ_3D = Mᵀ·dL/dΣ_2D·M.
    // We computed J already (j_r0, j_r1). cov_cam == W·Σ_3D·Wᵀ.
    let dsigma_2d = mat2x2<f32>(
        vec2<f32>(ds00, ds01 * 0.5),
        vec2<f32>(ds01 * 0.5, ds11),
    );
    // J as mat2x3 (cols = J rows are length-3). Build a "JT" 3x2.
    let j_row_x = vec3<f32>(j_r0.x, j_r1.x, 0.0);     // unused, kept for clarity
    // Compute dL/dcov_cam = Jᵀ · dL/dΣ_2D · J (3×3 symmetric).
    // dL/dcov_cam[i,j] = Σ_a Σ_b J[a,i] · dL/dΣ_2D[a,b] · J[b,j]
    var dcov_cam_00: f32 = 0.0; var dcov_cam_01: f32 = 0.0; var dcov_cam_02: f32 = 0.0;
    var dcov_cam_11: f32 = 0.0; var dcov_cam_12: f32 = 0.0;
    var dcov_cam_22: f32 = 0.0;
    for (var a: u32 = 0u; a < 2u; a = a + 1u) {
        for (var b: u32 = 0u; b < 2u; b = b + 1u) {
            let coef = dsigma_2d[b][a]; // matrix index order in WGSL
            let ja: vec3<f32> = select(j_r1, j_r0, a == 0u);
            let jb: vec3<f32> = select(j_r1, j_r0, b == 0u);
            dcov_cam_00 = dcov_cam_00 + ja.x * coef * jb.x;
            dcov_cam_01 = dcov_cam_01 + ja.x * coef * jb.y;
            dcov_cam_02 = dcov_cam_02 + ja.x * coef * jb.z;
            dcov_cam_11 = dcov_cam_11 + ja.y * coef * jb.y;
            dcov_cam_12 = dcov_cam_12 + ja.y * coef * jb.z;
            dcov_cam_22 = dcov_cam_22 + ja.z * coef * jb.z;
        }
    }
    // 5. cov_cam = W · cov_world · Wᵀ  → dL/dcov_world = Wᵀ · dL/dcov_cam · W.
    let dcov_cam = mat3x3<f32>(
        vec3<f32>(dcov_cam_00, dcov_cam_01, dcov_cam_02),
        vec3<f32>(dcov_cam_01, dcov_cam_11, dcov_cam_12),
        vec3<f32>(dcov_cam_02, dcov_cam_12, dcov_cam_22),
    );
    let dcov_world = transpose(W) * dcov_cam * W;

    // 6. cov_world = R · S² · Rᵀ. dL/dR = 2 · dL/dcov_world · R · S².
    //    dL/d(σ_i²) = (R_col_i)ᵀ · dL/dcov_world · R_col_i.
    let R_S2 = mat3x3<f32>(R[0] * s2.x, R[1] * s2.y, R[2] * s2.z);
    let dR = (dcov_world + transpose(dcov_world)) * R_S2;
    // dR is a 3×3 matrix of dL/dR[i,j]. Convert column-major to row-major
    // when chaining to quaternion components (the closed-form derivatives
    // are written in row,col notation).
    // Notation: G[row][col]. dR[col][row] in WGSL.
    let G00 = dR[0][0]; let G10 = dR[0][1]; let G20 = dR[0][2];
    let G01 = dR[1][0]; let G11 = dR[1][1]; let G21 = dR[1][2];
    let G02 = dR[2][0]; let G12 = dR[2][1]; let G22 = dR[2][2];

    // 7. σ² gradient → log-σ gradient.
    //    dL/d(σ_i²) = (R[col=i])ᵀ · dcov_world · R[col=i]
    //    dL/d(log σ_i) = dL/d(σ_i²) · 2 · σ_i² (chain via σ² = exp(2 log σ)).
    let r0 = R[0]; let r1 = R[1]; let r2 = R[2];
    let d_s2_x = dot(r0, dcov_world * r0);
    let d_s2_y = dot(r1, dcov_world * r1);
    let d_s2_z = dot(r2, dcov_world * r2);
    let d_log_sigma = vec3<f32>(d_s2_x, d_s2_y, d_s2_z) * 2.0 * s2;
    d_scales[idx] = vec4<f32>(d_log_sigma, 0.0);

    // 8. Quaternion gradient — closed form from quat_to_mat3 partial
    //    derivatives, with G stored in (row, col) order above.
    let q = rotations[idx];
    let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
    let d_qw = 2.0 * ( qz * (G10 - G01) + qy * (G02 - G20) + qx * (G21 - G12));
    let d_qx = 2.0 * ( qy * (G01 + G10) + qz * (G02 + G20) + qw * (G21 - G12)) - 4.0 * qx * (G11 + G22);
    let d_qy = -4.0 * qy * (G00 + G22)
             + 2.0 * (qx * (G01 + G10) + qw * (G02 - G20) + qz * (G12 + G21));
    let d_qz = -4.0 * qz * (G00 + G11)
             + 2.0 * (qw * (G10 - G01) + qx * (G02 + G20) + qy * (G12 + G21));
    d_rotations[idx] = vec4<f32>(d_qw, d_qx, d_qy, d_qz);

    // 9. Opacity logit.
    d_opacities[idx] = d_opacity_logit;

    // 10. Position gradient: chain dmean_pix → t (camera-space) → p_world.
    //     mean.x = cx + f·t.x/t.z   → ∂mean.x/∂t.x =  f/t.z,
    //                                  ∂mean.x/∂t.z = −f·t.x/t.z²
    //     mean.y = cy − f·t.y/t.z   → ∂mean.y/∂t.y = −f/t.z,
    //                                  ∂mean.y/∂t.z =  f·t.y/t.z²
    let dt_dx = vec3<f32>( f * inv_z,        0.0,                  -f * t.x * inv_z2);
    let dt_dy = vec3<f32>( 0.0,             -f * inv_z,             f * t.y * inv_z2);
    let dL_dt = dt_dx * dmean_pix.x + dt_dy * dmean_pix.y;
    //     t.z stored as −cam_xyz.z, so ∂t/∂cam = diag(1, 1, −1).
    let dL_dcam = vec3<f32>(dL_dt.x, dL_dt.y, -dL_dt.z);
    // p_cam = W · p_world (after dropping translation) → dL/dp_world = Wᵀ · dL/dcam.
    let dL_dpos = transpose(W) * dL_dcam;
    d_positions[idx] = vec4<f32>(dL_dpos, 0.0);
}
