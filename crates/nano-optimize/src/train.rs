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
    let gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
    let rasterizer = Rasterizer::new(&ctx);
    let binner = TileBinner::new(&ctx);

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

        adam_pos.t += 1;
        adam_rot.t += 1;
        adam_scale.t += 1;
        adam_op.t += 1;
        adam_dc.t += 1;
        adam_rest.t += 1;

        if iter % 100 == 0 || iter + 1 == cfg.iterations {
            // Gradient-norm diagnostics — pos / opacity / sh_dc / scale.
            // Cheap to read (small buffers); only on log iterations.
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
                "[train] iter {iter:>5} | view {:>3}/{} | mse {mse:.4} | grad‖pos‖={norm_pos:.2e} ‖op‖={norm_op:.2e} ‖dc‖={norm_dc:.2e} ‖scale‖={norm_scale:.2e} | {} splats",
                iter as usize % refs.len(),
                refs.len(),
                splats.len(),
            );
        }
    }

    eprintln!(
        "[train] complete. final splat count: {} (forward-only, no gradients applied yet)",
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
