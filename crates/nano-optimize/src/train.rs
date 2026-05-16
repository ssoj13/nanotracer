//! High-level training-loop driver.
//!
//! Phase A2.6 wires the forward rasteriser end-to-end:
//!   1. Bake reference views via the raytracer.
//!   2. Seed splats via the forward-fit (`nano-splat`).
//!   3. Upload to GPU once (`GpuSplatBuffer`).
//!   4. Per iteration:
//!       - Cycle to the next reference view.
//!       - Project splats into screen space (`Rasterizer::project`).
//!       - Bin into 16×16 tiles + sort (`TileBinner`).
//!       - α-composite into a predicted frame (`Rasterizer::composite`).
//!       - Read back + compute MSE vs reference. Log periodically.
//!   5. Dump iter-0 predicted frame to `train_predicted.png` for visual
//!      verification.
//!
//! Backward (Phase A3) and Adam updates are still placeholders — Adam
//! state grows / increments its step counter so future phases can drop
//! gradients in without restructuring.

use glam::Vec3;
use nano_core::scene::Scene;
use nano_io::utils::save_image;
use nano_splat::SplatConfigGpu;

use crate::adam::{AdamConfig, AdamState};
use crate::gpu::WgpuCtx;
use crate::raster::{CameraUniform, Rasterizer};
use crate::reference::{BakeConfig, bake_references};
use crate::splat_gpu::{GpuSplatBuffer, GradSplatBuffers};
use crate::splat_store::SplatBuffer;
use crate::tile_binner::{TileBinner, TilingParams};

// Quaternion re-normalisation after each step keeps rotations on the
// unit hyper-sphere; Adam happily drifts off otherwise. Hard cap on
// log-σ and opacity-logit prevents runaway exponentials.
const LOG_SCALE_MIN: f32 = -10.0;
const LOG_SCALE_MAX: f32 =   5.0;
const OPACITY_LOGIT_MAX: f32 = 10.0;

/// Top-level training configuration.
pub struct TrainConfig {
    pub iterations: u32,
    pub max_splats: usize,
    pub reference: BakeConfig,
    pub seed: SplatConfigGpu,
    pub adam_pos: AdamConfig,
    pub adam_attr: AdamConfig,
}

/// Run the optimisation loop and return the final splat buffer.
///
/// Phase A2.6 — forward rasteriser only. Adam state advances but no
/// gradient is applied. The returned `SplatBuffer` is identical to the
/// seed; backward (A3) and Adam updates (A3/A4) make this stage start
/// shaping the splats.
pub fn train(
    scene: &Scene,
    cfg: &TrainConfig,
) -> Result<SplatBuffer, Box<dyn std::error::Error>> {
    eprintln!(
        "[train] baking {} reference views at {}×{}...",
        cfg.reference.views, cfg.reference.width, cfg.reference.height
    );
    let refs = bake_references(scene, &cfg.reference)?;
    eprintln!("[train] {} references ready", refs.len());

    eprintln!("[train] seeding splats via forward fit...");
    let seeds = nano_splat::generate_splats_gpu(scene, &cfg.seed)?;
    let splats = SplatBuffer::from_gaussians(&seeds);
    eprintln!("[train] {} seed splats", splats.len());

    // Adam state is sized to match the seed splats. Densify (A5) will
    // grow these in lock-step with the splat buffer.
    let mut adam_pos = AdamState::new(cfg.adam_pos, splats.len() * 3);
    let mut adam_rot = AdamState::new(cfg.adam_attr, splats.len() * 4);
    let mut adam_scale = AdamState::new(cfg.adam_attr, splats.len() * 3);
    let mut adam_op = AdamState::new(cfg.adam_attr, splats.len());
    let mut adam_dc = AdamState::new(cfg.adam_attr, splats.len() * 3);
    let mut adam_rest = AdamState::new(cfg.adam_attr, splats.len() * 45);

    eprintln!("[train] initialising wgpu rasteriser...");
    let ctx = WgpuCtx::new()?;
    let mut splats = splats; // shadowed mutable for in-place Adam updates
    let mut gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
    let rasterizer = Rasterizer::new(&ctx);
    let binner = TileBinner::new(&ctx);

    // Prune cadence — every K iters past the warmup, sweep splats whose
    // sigmoid-opacity has fallen below 0.005 (logit ≈ −5.3). These are
    // splats Adam has been pushing toward transparency for many steps;
    // keeping them is wasted budget. Empty grad_acc accumulator tracks
    // per-splat L1(d_position) for the future densify pass (A5.2).
    const PRUNE_EVERY: u32 = 100;
    const PRUNE_WARMUP: u32 = 50;
    const PRUNE_OPACITY_LOGIT: f32 = -5.3;
    let mut grad_acc: Vec<f32> = vec![0.0; splats.len()];

    let width = cfg.reference.width;
    let height = cfg.reference.height;
    let tiling = TilingParams {
        width,
        height,
        tile_size: 16,
        // Depth range for quantisation: anything past `orbit_radius * 2`
        // is clipped into the last 16-bit depth bucket. Scenes whose
        // bounding box spans further than this will need a tweaked value.
        depth_max: cfg.reference.orbit_radius * 2.0,
    };
    let predicted = rasterizer.alloc_image(&ctx, width, height);
    let projected = ctx.storage_buffer_zeroed(
        "projected",
        (gpu_splats.n as u64) * std::mem::size_of::<crate::raster::ProjectedSplat>() as u64,
    );
    // Backward-pass scratch buffers (reused every iteration).
    let projected_grad = rasterizer.alloc_projected_grad(&ctx, gpu_splats.n);
    let grads = GradSplatBuffers::new(&ctx, gpu_splats.n);
    let dl_dc_buf = ctx.storage_buffer_zeroed(
        "dl_dc",
        (width as u64) * (height as u64) * 16,
    );

    eprintln!("[train] forward rasterising {} iterations...", cfg.iterations);
    for iter in 0..cfg.iterations {
        let view = &refs[(iter as usize) % refs.len()];
        let camera = CameraUniform::from_pose(
            view.camera_pos,
            view.target,
            view.up,
            view.fov_y,
            view.width,
            view.height,
            gpu_splats.n,
        );

        // Forward pass: project → bin → composite.
        rasterizer.project(&ctx, &gpu_splats, &camera, &projected);
        let bins = binner.bin(&ctx, &projected, gpu_splats.n, &tiling);
        rasterizer.composite(
            &ctx,
            &projected,
            &bins.sorted_payloads,
            &bins.tile_ranges,
            &predicted,
            &tiling,
        );

        // Read back the predicted frame; compare to reference radiance.
        let pred: Vec<[f32; 4]> = ctx.readback(&predicted, (width * height) as usize);
        let pred_rgb: Vec<Vec3> = pred
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        let mse = mean_squared_error(&pred_rgb, &view.pixels);

        // Backward pass: MSE → dL/dC, then α-blend backward, then
        // project backward. Per-pixel loss gradient for MSE over N
        // pixels × 3 channels: dL/dC = 2·(predicted - target) / (3·W·H).
        let n_scalars = (width as f32) * (height as f32) * 3.0;
        let dl_dc_vec: Vec<[f32; 4]> = pred_rgb
            .iter()
            .zip(view.pixels.iter())
            .map(|(p, t)| {
                let d = (*p - *t) * (2.0 / n_scalars);
                [d.x, d.y, d.z, 0.0]
            })
            .collect();
        ctx.queue
            .write_buffer(&dl_dc_buf, 0, bytemuck::cast_slice(&dl_dc_vec));
        rasterizer.zero_projected_grad(&ctx, &projected_grad, gpu_splats.n);
        grads.zero(&ctx);
        rasterizer.composite_backward(
            &ctx,
            &projected,
            &bins.sorted_payloads,
            &bins.tile_ranges,
            &predicted,
            &dl_dc_buf,
            &projected_grad,
            &tiling,
        );
        rasterizer.project_backward(&ctx, &gpu_splats, &projected_grad, &camera, &grads);

        // First-iteration sanity dump — lets a human spot whether the
        // rasteriser is producing anything recognisable before sitting
        // through 30k iterations.
        if iter == 0 {
            save_image(&pred_rgb, width, height, "train_predicted.png", false)?;
            save_image(&view.pixels, width, height, "train_reference.png", false)?;
            eprintln!(
                "[train] iter 0: wrote train_predicted.png / train_reference.png ({}×{})",
                width, height
            );
        }

        // Adam update — readback gradients, flatten, step, write back.
        let n = gpu_splats.n as usize;
        let g_pos: Vec<[f32; 4]> = ctx.readback(&grads.d_positions, n);
        let g_rot: Vec<[f32; 4]> = ctx.readback(&grads.d_rotations, n);
        let g_scale: Vec<[f32; 4]> = ctx.readback(&grads.d_scales, n);
        let g_op: Vec<f32> = ctx.readback(&grads.d_opacities, n);
        let g_dc: Vec<[f32; 4]> = ctx.readback(&grads.d_sh_dc, n);
        let g_sh_rest: Vec<f32> = ctx.readback(&grads.d_sh_rest, n * 48);

        let mut pos_flat = flatten_vec3(&splats.positions);
        let mut rot_flat: Vec<f32> = splats.rotations.iter().flat_map(|q| q.iter().copied()).collect();
        let mut scale_flat: Vec<f32> = splats.scales.iter().flat_map(|s| s.iter().copied()).collect();
        let mut op_flat: Vec<f32> = splats.opacities.clone();
        let mut dc_flat: Vec<f32> = splats.sh_dc.iter().flat_map(|c| c.iter().copied()).collect();
        let mut rest_flat: Vec<f32> = splats.sh_rest.clone();

        let g_pos_flat = flatten_vec4_strip_w(&g_pos);
        let g_rot_flat: Vec<f32> = g_rot.iter().flat_map(|q| q.iter().copied()).collect();
        let g_scale_flat = flatten_vec4_strip_w(&g_scale);
        let g_op_flat = g_op.clone();
        let g_dc_flat = flatten_vec4_strip_w(&g_dc);
        let g_rest_flat = strip_sh_rest_padding(&g_sh_rest, n);

        adam_pos.step(&mut pos_flat, &g_pos_flat);
        adam_rot.step(&mut rot_flat, &g_rot_flat);
        adam_scale.step(&mut scale_flat, &g_scale_flat);
        adam_op.step(&mut op_flat, &g_op_flat);
        adam_dc.step(&mut dc_flat, &g_dc_flat);
        adam_rest.step(&mut rest_flat, &g_rest_flat);

        // Write updated params back to SplatBuffer, applying physical
        // constraints (quat re-norm, log-σ clamp, opacity-logit clamp).
        for i in 0..n {
            splats.positions[i] = Vec3::new(pos_flat[i * 3], pos_flat[i * 3 + 1], pos_flat[i * 3 + 2]);
            let mut q = [rot_flat[i * 4], rot_flat[i * 4 + 1], rot_flat[i * 4 + 2], rot_flat[i * 4 + 3]];
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-8);
            for c in &mut q { *c /= norm; }
            splats.rotations[i] = q;
            splats.scales[i] = [
                scale_flat[i * 3].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX),
                scale_flat[i * 3 + 1].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX),
                scale_flat[i * 3 + 2].clamp(LOG_SCALE_MIN, LOG_SCALE_MAX),
            ];
            splats.opacities[i] = op_flat[i].clamp(-OPACITY_LOGIT_MAX, OPACITY_LOGIT_MAX);
            splats.sh_dc[i] = [dc_flat[i * 3], dc_flat[i * 3 + 1], dc_flat[i * 3 + 2]];
        }
        splats.sh_rest = rest_flat;

        // Accumulate per-splat |d_position| for the future densify gate.
        for i in 0..n {
            grad_acc[i] += g_pos_flat[i * 3].abs()
                         + g_pos_flat[i * 3 + 1].abs()
                         + g_pos_flat[i * 3 + 2].abs();
        }

        // Periodic prune: drop splats Adam has pushed to near-zero
        // opacity. swap_remove is O(1) per attribute and AdamState's
        // moments + grad_acc must follow in lockstep so per-splat
        // state stays consistent.
        let do_prune = iter >= PRUNE_WARMUP && iter % PRUNE_EVERY == 0;
        if do_prune {
            let removed = prune_low_opacity(
                &mut splats,
                PRUNE_OPACITY_LOGIT,
                &mut adam_pos,
                &mut adam_rot,
                &mut adam_scale,
                &mut adam_op,
                &mut adam_dc,
                &mut adam_rest,
                &mut grad_acc,
            );
            if removed > 0 {
                eprintln!(
                    "[train] iter {iter}: pruned {removed} low-opacity splats ({}→{})",
                    splats.len() + removed,
                    splats.len()
                );
                gpu_splats.n = splats.len() as u32;
            }
        }
        gpu_splats.sync_from(&ctx, &splats);

        if iter % 100 == 0 || iter + 1 == cfg.iterations {
            // Gradient-norm diagnostics — pos / opacity / sh_dc / scale.
            // Reuse the readbacks done above for Adam.
            let norm_pos = grad_norm_vec4(&g_pos);
            let norm_op = grad_norm_scalar(&g_op);
            let norm_dc = grad_norm_vec4(&g_dc);
            let norm_scale = grad_norm_vec4(&g_scale);
            eprintln!(
                "[train] iter {iter:>5} | view {:>3}/{} | mse {mse:.4} | grad‖pos‖={norm_pos:.2e} ‖op‖={norm_op:.2e} ‖dc‖={norm_dc:.2e} ‖scale‖={norm_scale:.2e} | {} splats",
                iter as usize % refs.len(),
                refs.len(),
                splats.len(),
            );
        }
    }

    eprintln!(
        "[train] complete. final splat count: {} (Adam updates applied)",
        splats.len()
    );
    Ok(splats)
}

/// Mean squared error over `Vec3` framebuffers — sums squared per-
/// channel differences, divides by total scalar count.
fn mean_squared_error(a: &[Vec3], b: &[Vec3]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = *av - *bv;
        sum += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
    }
    (sum / (a.len() as f64 * 3.0)) as f32
}

/// Drop splats whose opacity logit is below `threshold` (effectively
/// invisible). `swap_remove` is applied in lockstep to the
/// `SplatBuffer`, every `AdamState` slab (sized to each attribute's
/// per-splat stride), and the per-splat `grad_acc` accumulator.
/// Returns the count removed.
#[allow(clippy::too_many_arguments)]
fn prune_low_opacity(
    splats: &mut SplatBuffer,
    threshold: f32,
    adam_pos: &mut AdamState,
    adam_rot: &mut AdamState,
    adam_scale: &mut AdamState,
    adam_op: &mut AdamState,
    adam_dc: &mut AdamState,
    adam_rest: &mut AdamState,
    grad_acc: &mut Vec<f32>,
) -> usize {
    let mut to_remove: Vec<usize> = (0..splats.len())
        .filter(|&i| splats.opacities[i] < threshold)
        .collect();
    // Apply high-to-low so earlier indices remain valid as we shrink.
    to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for &i in &to_remove {
        let last = splats.len() - 1;
        splats.swap_remove(i);
        adam_pos.swap_remove_range(i * 3, 3);
        adam_rot.swap_remove_range(i * 4, 4);
        adam_scale.swap_remove_range(i * 3, 3);
        adam_op.swap_remove_range(i, 1);
        adam_dc.swap_remove_range(i * 3, 3);
        adam_rest.swap_remove_range(i * 45, 45);
        // grad_acc swap matches the SplatBuffer swap_remove semantics.
        grad_acc.swap(i, last);
        grad_acc.pop();
    }
    to_remove.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_drops_low_opacity_splats() {
        let mut splats = SplatBuffer::default();
        for i in 0..5 {
            splats.push_splat(
                Vec3::splat(i as f32),
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0],
                if i % 2 == 0 { 2.0 } else { -10.0 },  // alternating: 2, -10, 2, -10, 2
                [0.0, 0.0, 0.0],
                &[0.0; 45],
            );
        }
        let mut a_pos = AdamState::new(AdamConfig::default(), 5 * 3);
        let mut a_rot = AdamState::new(AdamConfig::default(), 5 * 4);
        let mut a_scale = AdamState::new(AdamConfig::default(), 5 * 3);
        let mut a_op = AdamState::new(AdamConfig::default(), 5);
        let mut a_dc = AdamState::new(AdamConfig::default(), 5 * 3);
        let mut a_rest = AdamState::new(AdamConfig::default(), 5 * 45);
        let mut grad_acc = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let removed = prune_low_opacity(
            &mut splats,
            -5.3,
            &mut a_pos, &mut a_rot, &mut a_scale, &mut a_op, &mut a_dc, &mut a_rest,
            &mut grad_acc,
        );
        assert_eq!(removed, 2, "should drop indices 1 and 3 (opacity = -10)");
        assert_eq!(splats.len(), 3);
        assert_eq!(grad_acc.len(), 3);
        assert_eq!(a_pos.m.len(), 3 * 3);
        assert_eq!(a_rot.m.len(), 3 * 4);
        assert_eq!(a_scale.m.len(), 3 * 3);
        assert_eq!(a_op.m.len(), 3);
        assert_eq!(a_dc.m.len(), 3 * 3);
        assert_eq!(a_rest.m.len(), 3 * 45);
        // Survivors carry opacity = 2.0.
        for o in &splats.opacities {
            assert!((*o - 2.0).abs() < 1e-6, "opacity should be 2.0, got {o}");
        }
    }
}

fn flatten_vec3(v: &[Vec3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.push(p.x);
        out.push(p.y);
        out.push(p.z);
    }
    out
}

fn flatten_vec4_strip_w(v: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.push(p[0]);
        out.push(p[1]);
        out.push(p[2]);
    }
    out
}

/// `sh_rest` on the GPU is padded from 45 to 48 floats per splat (3
/// trailing zeros). Strip the padding so AdamState (sized for 45·n)
/// gets the matching slab.
fn strip_sh_rest_padding(padded: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 45);
    for i in 0..n {
        let base = i * 48;
        out.extend_from_slice(&padded[base..base + 45]);
    }
    out
}

/// L2 norm over a flat vec4-padded gradient buffer (w channel is padding).
fn grad_norm_vec4(g: &[[f32; 4]]) -> f32 {
    let mut s = 0.0_f64;
    for v in g {
        s += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64;
    }
    s.sqrt() as f32
}

/// L2 norm over a flat scalar gradient buffer.
fn grad_norm_scalar(g: &[f32]) -> f32 {
    let mut s = 0.0_f64;
    for v in g {
        s += (*v as f64) * (*v as f64);
    }
    s.sqrt() as f32
}
