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
use crate::adam_gpu::AdamGpu;
use crate::gpu::WgpuCtx;
use crate::raster::{CameraUniform, Rasterizer};
use crate::reference::{BakeConfig, bake_references};
use crate::splat_gpu::{AdamMomentBuffers, GpuSplatBuffer, GradSplatBuffers};
use crate::splat_store::SplatBuffer;
use crate::tile_binner::{TileBinner, TilingParams};

// Quaternion re-normalisation after each step keeps rotations on the
// unit hyper-sphere; Adam happily drifts off otherwise. Hard cap on
// log-σ and opacity-logit prevents runaway exponentials.
const LOG_SCALE_MIN: f32 = -10.0;
const LOG_SCALE_MAX: f32 = 5.0;
const OPACITY_LOGIT_MAX: f32 = 10.0;

/// Top-level training configuration.
pub struct TrainConfig {
    pub iterations: u32,
    pub max_splats: usize,
    pub reference: BakeConfig,
    pub seed: SplatConfigGpu,
    pub adam_pos: AdamConfig,
    pub adam_attr: AdamConfig,
    /// Blend factor for the combined MSE + SSIM loss:
    /// `L = (1 − λ) · MSE + λ · (1 − SSIM)`. `λ = 0` reproduces the
    /// pure-MSE behaviour bit-for-bit; the Inria 3DGS default is `0.2`.
    pub ssim_lambda: f32,
}

/// Run the optimisation loop and return the final splat buffer.
///
/// `on_iter` is invoked at the end of every iteration with
/// `(iter_index, &updated_splats, combined_loss_for_this_iter)`. The
/// scalar is the full objective `(1 − λ) · MSE + λ · (1 − SSIM)` the
/// optimiser is descending — *not* MSE alone, so the viewer's loss
/// plot reflects what training is actually minimising. Pass a no-op
/// closure for the headless training path; the viewer wires this up
/// to publish a snapshot into a shared `RwLock` so the live preview
/// stays in sync with the training thread.
pub fn train<F: FnMut(u32, &SplatBuffer, f32)>(
    scene: &Scene,
    cfg: &TrainConfig,
    mut on_iter: F,
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
    const DENSIFY_EVERY: u32 = 200;
    const DENSIFY_WARMUP: u32 = 100;
    /// Average per-splat gradient magnitude above which the splat is
    /// considered a densification candidate. The Inria default is
    /// ~2e-4 on the screen-space 2D mean grad; we use L1 of the 3D
    /// position grad which sits in a similar range.
    const DENSIFY_GRAD_THRESHOLD: f32 = 2e-4;
    /// log σ above which a candidate is "large" → SPLIT instead of CLONE.
    const SPLIT_SCALE_THRESHOLD: f32 = -2.3; // σ > 0.1
    // Per-splat L1(d_position) accumulator lives on the GPU now
    // (`grad_acc_buf`); CPU `grad_acc` is allocated lazily at the
    // densify checkpoint when the data has to round-trip through the
    // CPU mirror anyway. `grad_count` still lives here because the
    // densify gate normalises the accumulator by it.
    let mut grad_count: u32 = 0;

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
    let mut projected = ctx.storage_buffer_zeroed(
        "projected",
        (gpu_splats.n as u64) * std::mem::size_of::<crate::raster::ProjectedSplat>() as u64,
    );
    // Backward-pass scratch buffers (reused every iteration).
    let mut projected_grad = rasterizer.alloc_projected_grad(&ctx, gpu_splats.n);
    let mut grads = GradSplatBuffers::new(&ctx, gpu_splats.n);
    let dl_dc_buf = ctx.storage_buffer_zeroed("dl_dc", (width as u64) * (height as u64) * 16);

    // GPU-resident Adam moments + per-splat L1(d_position) accumulator.
    // After this point CPU AdamState only tracks `t` for bias correction
    // and is mirrored from GPU at densify/prune checkpoints.
    let adam_gpu = AdamGpu::new(&ctx);
    let mut moments = AdamMomentBuffers::new(&ctx, gpu_splats.n);
    let mut grad_acc_buf =
        ctx.storage_buffer_zeroed("grad_acc_pos", (gpu_splats.n as u64).max(1) * 4);

    eprintln!(
        "[train] forward rasterising {} iterations...",
        cfg.iterations
    );
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
        let pred_rgb: Vec<Vec3> = pred.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();

        // Combined loss `L = (1 − λ) · MSE + λ · (1 − SSIM)`. The
        // returned `dl_dc_vec3` already carries the λ-weighted sum of
        // both component gradients (linearity of derivatives) so the
        // GPU backward pass needs no further blending.
        let (loss_total, loss_mse, loss_ssim, dl_dc_vec3) = crate::loss::combined_loss_grad(
            &pred_rgb,
            &view.pixels,
            width,
            height,
            cfg.ssim_lambda,
        );
        let dl_dc_vec: Vec<[f32; 4]> = dl_dc_vec3.iter().map(|d| [d.x, d.y, d.z, 0.0]).collect();
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

        // Pure-GPU Adam update. The 6 step dispatches operate in-place
        // on `gpu_splats` SSBOs; padded vec4 lanes (W slot) and sh_rest
        // pads carry zero gradient → moments stay zero → update is 0.
        // After stepping, a single `apply_constraints` dispatch fixes
        // up the physical state (quat norm, log-σ + opacity clamps).
        adam_gpu.accumulate_grad(&ctx, &grads.d_positions, &grad_acc_buf, gpu_splats.n);
        let t = adam_pos.t + 1;
        adam_gpu.step_all(
            &ctx,
            &gpu_splats,
            &grads,
            &moments,
            t,
            &cfg.adam_pos,
            &cfg.adam_attr,
        );
        adam_gpu.apply_constraints(
            &ctx,
            &gpu_splats,
            LOG_SCALE_MIN,
            LOG_SCALE_MAX,
            OPACITY_LOGIT_MAX,
        );
        // `adam_pos.t` is the shared step counter for all six attributes:
        // `step_all` reads it once for bias correction, and densify/prune
        // resume on it. The other adam_* `t` fields are dead weight — left
        // at 0 — and intentionally not bumped here.
        adam_pos.t += 1;
        grad_count += 1;

        // Periodic prune / densify checkpoint. The hot path is fully
        // GPU-resident, but prune + densify still run on CPU because
        // they do attribute-wide swap_remove / push_splat that's far
        // simpler in `SplatBuffer` than in WGSL. We pay one round-trip
        // (readback splats + moments + grad_acc → mutate → re-upload)
        // every PRUNE_EVERY / DENSIFY_EVERY iterations.
        let do_prune = iter >= PRUNE_WARMUP && iter % PRUNE_EVERY == 0;
        let do_densify = iter >= DENSIFY_WARMUP && iter % DENSIFY_EVERY == 0;

        if do_prune || do_densify {
            // Pull current GPU state into the CPU mirror so prune / densify
            // helpers can mutate in lockstep with the existing AdamState
            // bookkeeping.
            splats = gpu_splats.readback(&ctx);
            moments.download_to_cpu(
                &ctx,
                &mut adam_pos,
                &mut adam_rot,
                &mut adam_scale,
                &mut adam_op,
                &mut adam_dc,
                &mut adam_rest,
            );
            let mut grad_acc: Vec<f32> = ctx.readback(&grad_acc_buf, splats.len());

            let count_before = splats.len();
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
                        count_before,
                        splats.len()
                    );
                }
            }
            let mut reset_grad_acc = false;
            if do_densify && grad_count > 0 {
                let (cloned, split, removed) = densify(
                    &mut splats,
                    grad_count,
                    DENSIFY_GRAD_THRESHOLD,
                    SPLIT_SCALE_THRESHOLD,
                    cfg.max_splats,
                    iter,
                    &mut adam_pos,
                    &mut adam_rot,
                    &mut adam_scale,
                    &mut adam_op,
                    &mut adam_dc,
                    &mut adam_rest,
                    &mut grad_acc,
                );
                if cloned + split + removed > 0 {
                    eprintln!(
                        "[train] iter {iter}: densify cloned {cloned} split {split} (removed {removed} parents) → {} splats",
                        splats.len()
                    );
                }
                for g in grad_acc.iter_mut() {
                    *g = 0.0;
                }
                grad_count = 0;
                reset_grad_acc = true;
            }

            // Re-upload mutated CPU state to GPU. If splat count
            // changed, every GPU-side scratch buffer must be
            // reallocated because its size depends on `n`.
            let n_changed = splats.len() as u32 != gpu_splats.n;
            gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
            moments = AdamMomentBuffers::upload_from_cpu(
                &ctx,
                &adam_pos,
                &adam_rot,
                &adam_scale,
                &adam_op,
                &adam_dc,
                &adam_rest,
            );
            if n_changed {
                projected = ctx.storage_buffer_zeroed(
                    "projected",
                    (gpu_splats.n as u64)
                        * std::mem::size_of::<crate::raster::ProjectedSplat>() as u64,
                );
                projected_grad = rasterizer.alloc_projected_grad(&ctx, gpu_splats.n);
                grads = GradSplatBuffers::new(&ctx, gpu_splats.n);
                grad_acc_buf =
                    ctx.storage_buffer_zeroed("grad_acc_pos", (gpu_splats.n as u64).max(1) * 4);
                // Upload (already-resized) grad_acc back to GPU.
                ctx.queue
                    .write_buffer(&grad_acc_buf, 0, bytemuck::cast_slice(&grad_acc));
            } else if reset_grad_acc {
                adam_gpu.zero_grad_acc(&ctx, &grad_acc_buf, gpu_splats.n);
            } else {
                ctx.queue
                    .write_buffer(&grad_acc_buf, 0, bytemuck::cast_slice(&grad_acc));
            }
        }

        // Notify the caller (live viewer / metrics collector) with a
        // snapshot of MSE. The splat buffer is only round-tripped at
        // prune/densify cadence; intermediate iters publish the last
        // fresh CPU mirror (or the initial seed buffer otherwise).
        on_iter(iter, &splats, loss_total);

        if iter % 100 == 0 || iter + 1 == cfg.iterations {
            // Gradient-norm diagnostics — pos / opacity / sh_dc / scale.
            // These are diagnostic-only, so we eat the readback cost
            // once per 100 iters instead of doing it every iter.
            let n = gpu_splats.n as usize;
            let g_pos: Vec<[f32; 4]> = ctx.readback(&grads.d_positions, n);
            let g_op: Vec<f32> = ctx.readback(&grads.d_opacities, n);
            let g_dc: Vec<[f32; 4]> = ctx.readback(&grads.d_sh_dc, n);
            let g_scale: Vec<[f32; 4]> = ctx.readback(&grads.d_scales, n);
            let norm_pos = grad_norm_vec4(&g_pos);
            let norm_op = grad_norm_scalar(&g_op);
            let norm_dc = grad_norm_vec4(&g_dc);
            let norm_scale = grad_norm_vec4(&g_scale);
            eprintln!(
                "[train] iter {iter:>5} | view {:>3}/{} | mse {loss_mse:.4} ssim {loss_ssim:.4} total {loss_total:.4} | grad‖pos‖={norm_pos:.2e} ‖op‖={norm_op:.2e} ‖dc‖={norm_dc:.2e} ‖scale‖={norm_scale:.2e} | {} splats",
                iter as usize % refs.len(),
                refs.len(),
                gpu_splats.n,
            );
        }
    }

    // Pull the final GPU state into a CPU SplatBuffer for the caller —
    // intermediate iters skip this round-trip, so the in-loop `splats`
    // mirror is only fresh as of the last densify/prune checkpoint.
    let final_splats = gpu_splats.readback(&ctx);
    eprintln!(
        "[train] complete. final splat count: {} (Adam updates applied)",
        final_splats.len()
    );
    Ok(final_splats)
}

/// Inria-style densify: high-gradient splats either clone (small
/// "under-reconstructed") or split into two children (large
/// "over-reconstructed"). The original splat is removed for splits;
/// clones append a duplicate with halved opacity (so the pair's net
/// emission matches the parent's, letting Adam settle each toward its
/// own footprint).
///
/// Returns `(cloned, split_children, removed_parents)`. Caps the total
/// splat count at `max_splats` — candidates exceeding the budget are
/// skipped silently rather than partially applied.
#[allow(clippy::too_many_arguments)]
fn densify(
    splats: &mut SplatBuffer,
    grad_count: u32,
    grad_threshold: f32,
    split_log_scale_threshold: f32,
    max_splats: usize,
    iter: u32,
    adam_pos: &mut AdamState,
    adam_rot: &mut AdamState,
    adam_scale: &mut AdamState,
    adam_op: &mut AdamState,
    adam_dc: &mut AdamState,
    adam_rest: &mut AdamState,
    grad_acc_mut: &mut Vec<f32>,
) -> (usize, usize, usize) {
    let n_before = splats.len();
    let count_f = grad_count.max(1) as f32;

    // Classify candidates without mutating yet — keeps index math simple.
    let mut to_clone: Vec<usize> = Vec::new();
    let mut to_split: Vec<usize> = Vec::new();
    let mut projected_count = n_before;
    for (i, &acc) in grad_acc_mut.iter().enumerate().take(n_before) {
        let avg = acc / count_f;
        if avg <= grad_threshold {
            continue;
        }
        let max_log_scale = splats.scales[i]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if max_log_scale > split_log_scale_threshold {
            // Split adds 2 children and removes the parent (net +1).
            if projected_count < max_splats {
                to_split.push(i);
                projected_count += 1;
            }
        } else if projected_count < max_splats {
            // Clone adds 1 child.
            to_clone.push(i);
            projected_count += 1;
        }
    }

    // Deterministic LCG seeded by iter — same scene + same seed makes
    // a training run reproducible for debug runs.
    let mut rng_state: u32 = iter.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);

    let half = -std::f32::consts::LN_2; // ln(0.5)
    let log_split_scale = -(1.6_f32).ln(); // -ln(1.6) ≈ -0.47

    // Apply clones: append duplicate with halved opacity, halve parent.
    for &i in &to_clone {
        let pos = splats.positions[i];
        let rot = splats.rotations[i];
        let scale = splats.scales[i];
        let opacity_new = splats.opacities[i] + half;
        let sh_dc = splats.sh_dc[i];
        let sh_rest_slice: Vec<f32> = splats.sh_rest[i * 45..(i + 1) * 45].to_vec();
        splats.push_splat(pos, rot, scale, opacity_new, sh_dc, &sh_rest_slice);
        splats.opacities[i] = opacity_new;
        adam_pos.grow(3);
        adam_rot.grow(4);
        adam_scale.grow(3);
        adam_op.grow(1);
        adam_dc.grow(3);
        adam_rest.grow(45);
        grad_acc_mut.push(0.0);
    }

    // Apply splits: append 2 children sampled from parent's Gaussian,
    // mark parent for removal.
    let mut parents_to_remove: Vec<usize> = Vec::with_capacity(to_split.len());
    for &i in &to_split {
        let parent_pos = splats.positions[i];
        let parent_rot = splats.rotations[i];
        let parent_scale = splats.scales[i];
        let parent_opacity = splats.opacities[i];
        let parent_sh_dc = splats.sh_dc[i];
        let parent_sh_rest: Vec<f32> = splats.sh_rest[i * 45..(i + 1) * 45].to_vec();
        let new_log_scale = [
            parent_scale[0] + log_split_scale,
            parent_scale[1] + log_split_scale,
            parent_scale[2] + log_split_scale,
        ];
        let sigma = [
            parent_scale[0].exp(),
            parent_scale[1].exp(),
            parent_scale[2].exp(),
        ];
        // rotations are stored (w, x, y, z); glam Quat::from_xyzw expects (x, y, z, w).
        let q = glam::Quat::from_xyzw(parent_rot[1], parent_rot[2], parent_rot[3], parent_rot[0]);
        for _child in 0..2 {
            let off = Vec3::new(
                box_muller(&mut rng_state) * sigma[0],
                box_muller(&mut rng_state) * sigma[1],
                box_muller(&mut rng_state) * sigma[2],
            );
            let new_pos = parent_pos + q * off;
            splats.push_splat(
                new_pos,
                parent_rot,
                new_log_scale,
                parent_opacity,
                parent_sh_dc,
                &parent_sh_rest,
            );
            adam_pos.grow(3);
            adam_rot.grow(4);
            adam_scale.grow(3);
            adam_op.grow(1);
            adam_dc.grow(3);
            adam_rest.grow(45);
            grad_acc_mut.push(0.0);
        }
        parents_to_remove.push(i);
    }

    // Remove parents high-to-low so earlier indices stay valid as we
    // shrink. Lock-step swap_remove on all attribute / moment slabs.
    parents_to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for &i in &parents_to_remove {
        let last = splats.len() - 1;
        splats.swap_remove(i);
        adam_pos.swap_remove_range(i * 3, 3);
        adam_rot.swap_remove_range(i * 4, 4);
        adam_scale.swap_remove_range(i * 3, 3);
        adam_op.swap_remove_range(i, 1);
        adam_dc.swap_remove_range(i * 3, 3);
        adam_rest.swap_remove_range(i * 45, 45);
        grad_acc_mut.swap(i, last);
        grad_acc_mut.pop();
    }

    (to_clone.len(), to_split.len() * 2, parents_to_remove.len())
}

/// Deterministic 32-bit LCG yielding `[0, 1)` floats.
fn lcg01(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state as f32) / (u32::MAX as f32 + 1.0)
}

/// One-sample Box-Muller — converts two uniform draws into a unit
/// standard-normal sample. Caller multiplies by `σ` for the splat's
/// per-axis scale before adding to the parent position.
fn box_muller(state: &mut u32) -> f32 {
    let u1 = lcg01(state).max(1e-8);
    let u2 = lcg01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn densify_clones_small_and_splits_large() {
        let mut splats = SplatBuffer::default();
        // Splat 0: small (log σ = -3 → σ ≈ 0.05), HIGH grad → CLONE.
        splats.push_splat(
            Vec3::new(0.0, 0.0, 0.0),
            [1.0, 0.0, 0.0, 0.0],
            [-3.0, -3.0, -3.0],
            2.0,
            [0.5, 0.5, 0.5],
            &[0.0; 45],
        );
        // Splat 1: large (log σ = 0 → σ = 1), HIGH grad → SPLIT.
        splats.push_splat(
            Vec3::new(5.0, 5.0, 5.0),
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            2.0,
            [0.3, 0.3, 0.3],
            &[0.0; 45],
        );
        // Splat 2: any scale, LOW grad → untouched.
        splats.push_splat(
            Vec3::new(-1.0, -1.0, -1.0),
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
            2.0,
            [0.1, 0.2, 0.3],
            &[0.0; 45],
        );

        let mut a_pos = AdamState::new(AdamConfig::default(), 3 * 3);
        let mut a_rot = AdamState::new(AdamConfig::default(), 3 * 4);
        let mut a_scale = AdamState::new(AdamConfig::default(), 3 * 3);
        let mut a_op = AdamState::new(AdamConfig::default(), 3);
        let mut a_dc = AdamState::new(AdamConfig::default(), 3 * 3);
        let mut a_rest = AdamState::new(AdamConfig::default(), 3 * 45);
        // Grad accumulator: first two have high grad, third low.
        let mut grad_acc = vec![1.0, 1.0, 0.0001];

        let (cloned, split_children, removed) = densify(
            &mut splats,
            /* grad_count */ 1,
            /* threshold */ 0.5,
            /* split_log_scale_threshold */ -2.3,
            /* max_splats */ 100,
            /* iter */ 42,
            &mut a_pos,
            &mut a_rot,
            &mut a_scale,
            &mut a_op,
            &mut a_dc,
            &mut a_rest,
            &mut grad_acc,
        );

        assert_eq!(cloned, 1, "should clone splat 0");
        assert_eq!(split_children, 2, "should produce 2 children from splat 1");
        assert_eq!(removed, 1, "should remove the original split parent");

        // Net count: 3 - 1 (parent removed) + 1 (clone) + 2 (split children) = 5.
        assert_eq!(splats.len(), 5);
        assert_eq!(grad_acc.len(), 5);
        // Adam slabs sized for 5 splats.
        assert_eq!(a_pos.m.len(), 5 * 3);
        assert_eq!(a_rot.m.len(), 5 * 4);
        assert_eq!(a_scale.m.len(), 5 * 3);
        assert_eq!(a_op.m.len(), 5);
        assert_eq!(a_dc.m.len(), 5 * 3);
        assert_eq!(a_rest.m.len(), 5 * 45);
    }

    #[test]
    fn prune_drops_low_opacity_splats() {
        let mut splats = SplatBuffer::default();
        for i in 0..5 {
            splats.push_splat(
                Vec3::splat(i as f32),
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0],
                if i % 2 == 0 { 2.0 } else { -10.0 }, // alternating: 2, -10, 2, -10, 2
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
            &mut a_pos,
            &mut a_rot,
            &mut a_scale,
            &mut a_op,
            &mut a_dc,
            &mut a_rest,
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
