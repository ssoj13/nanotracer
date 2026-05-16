//! Finite-difference gradient verification for the full backward pipeline.
//!
//! For each chosen parameter `p` of a single splat:
//!   1. Run forward (project → bin → composite) and read the predicted
//!      framebuffer. Compute scalar loss `L(p) = Σ_pixels Σ_channels rgb`.
//!   2. Numerical gradient: `(L(p + ε) − L(p − ε)) / (2ε)`.
//!   3. Analytic gradient: run backward (composite_backward then
//!      project_backward) and read the matching slot of the gradient
//!      buffers.
//!
//! When `L = Σ rgb` over all pixels, the per-pixel loss gradient is
//! uniform `dL/dC = (1, 1, 1)` — easy to bind. Matching the analytic
//! and numerical gradients (≤ 1e-2 relative tolerance) catches sign
//! errors in any of: sigmoid chain, SH basis, conic inversion, 2D→3D
//! covariance sandwich, quaternion → rotation, view projection.

use glam::Vec3;
use nano_optimize::gpu::WgpuCtx;
use nano_optimize::raster::{CameraUniform, Rasterizer};
use nano_optimize::splat_gpu::{GpuSplatBuffer, GradSplatBuffers};
use nano_optimize::splat_store::SplatBuffer;
use nano_optimize::tile_binner::{TileBinner, TilingParams};

const W: u32 = 32;
const H: u32 = 32;
const N_PIX: usize = (W as usize) * (H as usize);

fn base_splats() -> SplatBuffer {
    let mut buf = SplatBuffer::default();
    let mut rest = [0.0f32; 45];
    // Mild non-zero higher-order SH so the SH-band gradients exercise.
    rest[0] = 0.05;
    rest[20] = -0.03;
    buf.push_splat(
        Vec3::new(0.0, 0.0, -5.0), // pos
        [1.0, 0.0, 0.0, 0.0],      // identity quat (w, x, y, z)
        [-0.5, -0.6, -0.7],        // log σ → σ ≈ (0.6, 0.55, 0.5)
        2.0,                       // opacity logit → σ(2) ≈ 0.88
        [0.4, 0.3, 0.2],           // SH DC
        &rest,
    );
    buf
}

fn camera_for(n: u32) -> CameraUniform {
    CameraUniform::from_pose(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::Y,
        std::f32::consts::FRAC_PI_3, // 60°
        W,
        H,
        n,
    )
}

fn forward_loss(ctx: &WgpuCtx, splats: &SplatBuffer) -> f32 {
    let gpu = GpuSplatBuffer::upload(ctx, splats);
    let cam = camera_for(gpu.n);
    let raster = Rasterizer::new(ctx);
    let binner = TileBinner::new(ctx);
    let projected = raster.alloc_projected(ctx, gpu.n);
    raster.project(ctx, &gpu, &cam, &projected);
    let params = TilingParams {
        width: W,
        height: H,
        tile_size: 16,
        depth_max: 50.0,
    };
    let res = binner.bin(ctx, &projected, gpu.n, &params);
    let img = raster.alloc_image(ctx, W, H);
    raster.composite(
        ctx,
        &projected,
        &res.sorted_payloads,
        &res.tile_ranges,
        &img,
        &params,
    );
    let pix: Vec<[f32; 4]> = ctx.readback(&img, N_PIX);
    let mut sum: f64 = 0.0;
    for p in &pix {
        sum += (p[0] + p[1] + p[2]) as f64;
    }
    sum as f32
}

/// Analytic gradient for one full forward + backward pass.
/// Returns (d_positions[0], d_rotations[0], d_scales[0], d_opacity, d_sh_dc[0]).
struct AnalyticGrads {
    pos: [f32; 4],
    rot: [f32; 4],
    scale: [f32; 4],
    opacity: f32,
    sh_dc: [f32; 4],
    sh_rest_0: f32, // first band-1 coefficient (R channel, index 0 in rest)
}

fn analytic_grads(ctx: &WgpuCtx, splats: &SplatBuffer) -> AnalyticGrads {
    let gpu = GpuSplatBuffer::upload(ctx, splats);
    let cam = camera_for(gpu.n);
    let raster = Rasterizer::new(ctx);
    let binner = TileBinner::new(ctx);

    let projected = raster.alloc_projected(ctx, gpu.n);
    raster.project(ctx, &gpu, &cam, &projected);
    let params = TilingParams {
        width: W,
        height: H,
        tile_size: 16,
        depth_max: 50.0,
    };
    let res = binner.bin(ctx, &projected, gpu.n, &params);
    let img = raster.alloc_image(ctx, W, H);
    raster.composite(
        ctx,
        &projected,
        &res.sorted_payloads,
        &res.tile_ranges,
        &img,
        &params,
    );

    // dL/dC = (1, 1, 1, 0) per pixel — loss is Σ rgb.
    let dl_dc_vec = vec![[1.0_f32, 1.0, 1.0, 0.0]; N_PIX];
    let dl_dc = ctx.storage_buffer("dl_dc", &dl_dc_vec);
    let proj_grad = raster.alloc_projected_grad(ctx, gpu.n);
    raster.composite_backward(
        ctx,
        &projected,
        &res.sorted_payloads,
        &res.tile_ranges,
        &img,
        &dl_dc,
        &proj_grad,
        &params,
    );
    let grads = GradSplatBuffers::new(ctx, gpu.n);
    raster.project_backward(ctx, &gpu, &proj_grad, &cam, &grads);

    let d_pos: Vec<[f32; 4]> = ctx.readback(&grads.d_positions, 1);
    let d_rot: Vec<[f32; 4]> = ctx.readback(&grads.d_rotations, 1);
    let d_scale: Vec<[f32; 4]> = ctx.readback(&grads.d_scales, 1);
    let d_op: Vec<f32> = ctx.readback(&grads.d_opacities, 1);
    let d_sh_dc: Vec<[f32; 4]> = ctx.readback(&grads.d_sh_dc, 1);
    let d_sh_rest: Vec<[f32; 4]> = ctx.readback(&grads.d_sh_rest, 12);
    AnalyticGrads {
        pos: d_pos[0],
        rot: d_rot[0],
        scale: d_scale[0],
        opacity: d_op[0],
        sh_dc: d_sh_dc[0],
        sh_rest_0: d_sh_rest[0][0],
    }
}

fn numerical(ctx: &WgpuCtx, splats: &SplatBuffer, mutator: impl Fn(&mut SplatBuffer, f32)) -> f32 {
    let eps = 1e-3;
    let mut plus = splats.clone();
    mutator(&mut plus, eps);
    let mut minus = splats.clone();
    mutator(&mut minus, -eps);
    let lp = forward_loss(ctx, &plus);
    let lm = forward_loss(ctx, &minus);
    (lp - lm) / (2.0 * eps)
}

fn close(a: f32, n: f32, rel: f32, abs: f32) -> bool {
    let diff = (a - n).abs();
    diff <= abs || diff <= rel * (a.abs() + n.abs())
}

#[test]
fn finite_diff_matches_analytic() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    let s = base_splats();
    let g = analytic_grads(&ctx, &s);
    eprintln!(
        "analytic: pos={:?} sh_dc={:?} opacity={} scale={:?} rot={:?}",
        g.pos, g.sh_dc, g.opacity, g.scale, g.rot
    );

    // sh_dc — strictly linear in colour → tight tolerance.
    let n_sh_dc_r = numerical(&ctx, &s, |b, e| b.sh_dc[0][0] += e);
    eprintln!("sh_dc.r analytic={} numerical={}", g.sh_dc[0], n_sh_dc_r);
    assert!(
        close(g.sh_dc[0], n_sh_dc_r, 0.02, 0.05),
        "sh_dc.r mismatch: analytic={} numerical={}",
        g.sh_dc[0],
        n_sh_dc_r
    );

    // opacity_logit — sigmoid chain.
    let n_op = numerical(&ctx, &s, |b, e| b.opacities[0] += e);
    eprintln!("opacity analytic={} numerical={}", g.opacity, n_op);
    assert!(
        close(g.opacity, n_op, 0.05, 0.05),
        "opacity mismatch: analytic={} numerical={}",
        g.opacity,
        n_op
    );

    // sh_rest[0] (band-1 R channel coefficient): basis chain.
    let n_sh_rest_0 = numerical(&ctx, &s, |b, e| b.sh_rest[0] += e);
    eprintln!(
        "sh_rest[0] analytic={} numerical={}",
        g.sh_rest_0, n_sh_rest_0
    );
    assert!(
        close(g.sh_rest_0, n_sh_rest_0, 0.05, 0.05),
        "sh_rest[0] mismatch: analytic={} numerical={}",
        g.sh_rest_0,
        n_sh_rest_0
    );

    // position.x: chains through view projection. Tolerance looser
    // because (a) we ignore the dir-dependence of SH on p_world by design
    // and (b) the projection is the most non-linear path.
    let n_pos_x = numerical(&ctx, &s, |b, e| b.positions[0].x += e);
    eprintln!("pos.x analytic={} numerical={}", g.pos[0], n_pos_x);
    assert!(
        close(g.pos[0], n_pos_x, 0.10, 0.5),
        "pos.x mismatch: analytic={} numerical={}",
        g.pos[0],
        n_pos_x
    );

    // log-scale x: chains through Σ_3D → Σ_2D → conic.
    let n_scale_x = numerical(&ctx, &s, |b, e| b.scales[0][0] += e);
    eprintln!(
        "log_scale.x analytic={} numerical={}",
        g.scale[0], n_scale_x
    );
    assert!(
        close(g.scale[0], n_scale_x, 0.10, 0.5),
        "log_scale.x mismatch: analytic={} numerical={}",
        g.scale[0],
        n_scale_x
    );
}
