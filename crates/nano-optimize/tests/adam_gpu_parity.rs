//! Parity tests for the Pure-GPU Adam optimiser.
//!
//! Runs the CPU `AdamState::step` and the GPU `AdamGpu::step` on the
//! same synthetic params + grads sequence and asserts the final
//! parameters / moments agree to within `1e-4`. Skips cleanly when no
//! GPU adapter is available (headless CI), like other tests in this
//! crate.

use glam::Vec3;
use nano_optimize::adam::{AdamConfig, AdamState};
use nano_optimize::adam_gpu::AdamGpu;
use nano_optimize::gpu::WgpuCtx;
use nano_optimize::splat_gpu::{AdamMomentBuffers, GpuSplatBuffer, GradSplatBuffers};
use nano_optimize::splat_store::SplatBuffer;

/// Run `steps` iterations of Adam on both CPU and GPU; assert outputs
/// match. `grad_fn(t, i)` produces the gradient for parameter `i` at
/// iteration `t` (1-indexed). Returns `Ok(())` on success and the
/// diagnostic string when parity fails.
fn run_parity(ctx: &WgpuCtx, n: usize, steps: u32, grad_fn: impl Fn(u32, usize) -> f32) {
    let cfg = AdamConfig::default();
    let mut cpu_params = vec![0.5_f32; n];
    let mut cpu_state = AdamState::new(cfg, n);

    let gpu = AdamGpu::new(ctx);
    let mut gpu_params_host = cpu_params.clone();
    let m_host = vec![0.0_f32; n];
    let v_host = vec![0.0_f32; n];
    let params_buf = ctx.storage_buffer("parity-params", &gpu_params_host);
    let grads_buf = ctx.storage_buffer("parity-grads", &vec![0.0_f32; n]);
    let m_buf = ctx.storage_buffer("parity-m", &m_host);
    let v_buf = ctx.storage_buffer("parity-v", &v_host);

    for t in 1..=steps {
        let g: Vec<f32> = (0..n).map(|i| grad_fn(t, i)).collect();
        // CPU step.
        cpu_state.step(&mut cpu_params, &g);
        // GPU step: re-upload grads, then dispatch.
        ctx.queue
            .write_buffer(&grads_buf, 0, bytemuck::cast_slice(&g));
        gpu.step(
            ctx,
            &params_buf,
            &grads_buf,
            &m_buf,
            &v_buf,
            n as u32,
            cfg.lr,
            t as u64,
            &cfg,
        );
    }

    gpu_params_host = ctx.readback(&params_buf, n);
    let m_gpu: Vec<f32> = ctx.readback(&m_buf, n);
    let v_gpu: Vec<f32> = ctx.readback(&v_buf, n);

    for i in 0..n {
        let dp = (cpu_params[i] - gpu_params_host[i]).abs();
        assert!(
            dp < 1e-4,
            "params[{}] diverge: cpu={} gpu={} (|Δ|={})",
            i,
            cpu_params[i],
            gpu_params_host[i],
            dp
        );
        let dm = (cpu_state.m[i] - m_gpu[i]).abs();
        let dv = (cpu_state.v[i] - v_gpu[i]).abs();
        assert!(
            dm < 1e-5 && dv < 1e-5,
            "moments[{}] diverge: m cpu={} gpu={} (Δ={}), v cpu={} gpu={} (Δ={})",
            i,
            cpu_state.m[i],
            m_gpu[i],
            dm,
            cpu_state.v[i],
            v_gpu[i],
            dv
        );
    }
}

#[test]
fn adam_parity_positive_grad() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    run_parity(&ctx, 256, 10, |_, _| 0.1);
}

#[test]
fn adam_parity_negative_grad() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    run_parity(&ctx, 256, 10, |_, _| -0.2);
}

#[test]
fn adam_parity_mixed_signs() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    // Per-element alternating sign + slight time-varying magnitude.
    run_parity(&ctx, 1024, 20, |t, i| {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        sign * (0.05 + 0.01 * (t as f32))
    });
}

#[test]
fn adam_parity_single_step_matches_moments() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    run_parity(&ctx, 64, 1, |_, i| 0.01 * (i as f32 + 1.0));
}

#[test]
fn adam_parity_large_n() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };
    // 8192 lanes — exercises multi-workgroup dispatch (workgroup_size 256).
    run_parity(&ctx, 8192, 5, |t, i| {
        let phase = ((i as f32) * 0.01 + (t as f32) * 0.1).sin();
        0.05 * phase
    });
}

/// End-to-end parity check for [`AdamGpu::step_all`] against six per-attribute
/// CPU `AdamState::step` calls. Constructs a real [`SplatBuffer`] and
/// [`AdamMomentBuffers`], injects synthetic gradients (different magnitudes
/// per attribute so a confusion across attrs is detectable), runs one step,
/// then asserts params and moments agree to tight tolerances.
///
/// This is the only test in the suite that exercises the actual training-loop
/// path: vec4 padding for pos/scale/dc, the 48-stride sh_rest layout, and the
/// scalar opacity dispatch.
#[test]
fn step_all_matches_per_attribute_cpu() {
    let ctx = match WgpuCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };

    // --- Build a non-trivial 4-splat buffer (non-unit rotations, non-zero
    //     scales/opacities, distinct sh_rest patterns) -------------------
    let n: usize = 4;
    let mut splats = SplatBuffer::default();
    for i in 0..n {
        let f = i as f32;
        let mut rest = [0.0f32; 45];
        for (k, slot) in rest.iter_mut().enumerate() {
            *slot = 0.1 * f + 0.01 * (k as f32);
        }
        splats.push_splat(
            Vec3::new(f, f * 0.5 - 1.0, f * 2.0 + 0.25),
            // Non-unit quaternion — apply_constraints would renorm it,
            // but step_all doesn't, so Adam sees it as-is.
            [0.7 + 0.1 * f, 0.2 * f, -0.1 * f, 0.3 - 0.05 * f],
            [-1.0 - 0.1 * f, -1.2 + 0.05 * f, -0.9 * (f + 1.0).ln()],
            -0.5 + 0.3 * f,
            [0.1 * f, 0.2 - 0.05 * f, 0.05 * f + 0.3],
            &rest,
        );
    }

    // --- Six CPU AdamState slabs sized per the training-loop convention.
    let cfg_pos = AdamConfig {
        lr: 5e-4,
        ..AdamConfig::default()
    };
    let cfg_attr = AdamConfig::default();
    let mut adam_pos = AdamState::new(cfg_pos, n * 3);
    let mut adam_rot = AdamState::new(cfg_attr, n * 4);
    let mut adam_scale = AdamState::new(cfg_attr, n * 3);
    let mut adam_op = AdamState::new(cfg_attr, n);
    let mut adam_dc = AdamState::new(cfg_attr, n * 3);
    let mut adam_rest = AdamState::new(cfg_attr, n * 45);

    // --- Upload to GPU + allocate moments/grads. ----------------------
    let mut gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
    let moments = AdamMomentBuffers::new(&ctx, gpu_splats.n);
    let grads = GradSplatBuffers::new(&ctx, gpu_splats.n);

    // --- Synthetic gradients. Distinct magnitudes per attribute so a
    //     bind-group confusion (e.g. opacities updated by sh_rest grads)
    //     would surface as a large delta. Pad vec3-shaped attrs to vec4
    //     with W=0; pad sh_rest's 45 → 48 with three trailing zeros, all
    //     matching the upload convention in splat_gpu.rs. ----------------
    let mut g_pos_cpu = vec![0.0f32; n * 3];
    let mut g_rot_cpu = vec![0.0f32; n * 4];
    let mut g_scale_cpu = vec![0.0f32; n * 3];
    let mut g_op_cpu = vec![0.0f32; n];
    let mut g_dc_cpu = vec![0.0f32; n * 3];
    let mut g_rest_cpu = vec![0.0f32; n * 45];
    for i in 0..n {
        let f = i as f32;
        g_pos_cpu[i * 3] = 0.10 + 0.001 * f;
        g_pos_cpu[i * 3 + 1] = -0.07 - 0.002 * f;
        g_pos_cpu[i * 3 + 2] = 0.03 + 0.0015 * f;
        g_rot_cpu[i * 4] = -0.05 + 0.001 * f;
        g_rot_cpu[i * 4 + 1] = 0.04 - 0.002 * f;
        g_rot_cpu[i * 4 + 2] = 0.06 + 0.0005 * f;
        g_rot_cpu[i * 4 + 3] = -0.02 + 0.003 * f;
        g_scale_cpu[i * 3] = 0.08 - 0.001 * f;
        g_scale_cpu[i * 3 + 1] = -0.09 + 0.002 * f;
        g_scale_cpu[i * 3 + 2] = 0.11 + 0.0005 * f;
        g_op_cpu[i] = -0.12 + 0.004 * f;
        g_dc_cpu[i * 3] = 0.02 + 0.0007 * f;
        g_dc_cpu[i * 3 + 1] = -0.03 - 0.0009 * f;
        g_dc_cpu[i * 3 + 2] = 0.04 + 0.0011 * f;
        for k in 0..45 {
            g_rest_cpu[i * 45 + k] = 0.001 * ((k as f32) + 1.0) + 0.0001 * f;
        }
    }
    let g_pos_padded: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            [
                g_pos_cpu[i * 3],
                g_pos_cpu[i * 3 + 1],
                g_pos_cpu[i * 3 + 2],
                0.0,
            ]
        })
        .collect();
    let g_scale_padded: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            [
                g_scale_cpu[i * 3],
                g_scale_cpu[i * 3 + 1],
                g_scale_cpu[i * 3 + 2],
                0.0,
            ]
        })
        .collect();
    let g_dc_padded: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            [
                g_dc_cpu[i * 3],
                g_dc_cpu[i * 3 + 1],
                g_dc_cpu[i * 3 + 2],
                0.0,
            ]
        })
        .collect();
    let mut g_rest_padded = vec![0.0f32; n * 48];
    for i in 0..n {
        for k in 0..45 {
            g_rest_padded[i * 48 + k] = g_rest_cpu[i * 45 + k];
        }
    }
    ctx.queue
        .write_buffer(&grads.d_positions, 0, bytemuck::cast_slice(&g_pos_padded));
    ctx.queue
        .write_buffer(&grads.d_rotations, 0, bytemuck::cast_slice(&g_rot_cpu));
    ctx.queue
        .write_buffer(&grads.d_scales, 0, bytemuck::cast_slice(&g_scale_padded));
    ctx.queue
        .write_buffer(&grads.d_opacities, 0, bytemuck::cast_slice(&g_op_cpu));
    ctx.queue
        .write_buffer(&grads.d_sh_dc, 0, bytemuck::cast_slice(&g_dc_padded));
    ctx.queue
        .write_buffer(&grads.d_sh_rest, 0, bytemuck::cast_slice(&g_rest_padded));

    // --- GPU step_all (single encoder + single submit under the hood). -
    let adam_gpu = AdamGpu::new(&ctx);
    adam_gpu.step_all(&ctx, &gpu_splats, &grads, &moments, 1, &cfg_pos, &cfg_attr);

    // --- Mirror on CPU: flatten the SplatBuffer to per-attribute slabs,
    //     run AdamState::step on each, then compare element-wise. ------
    let mut cpu_pos: Vec<f32> = splats
        .positions
        .iter()
        .flat_map(|p| [p.x, p.y, p.z])
        .collect();
    let mut cpu_rot: Vec<f32> = splats.rotations.iter().flat_map(|r| *r).collect();
    let mut cpu_scale: Vec<f32> = splats.scales.iter().flat_map(|s| *s).collect();
    let mut cpu_op: Vec<f32> = splats.opacities.clone();
    let mut cpu_dc: Vec<f32> = splats.sh_dc.iter().flat_map(|c| *c).collect();
    let mut cpu_rest: Vec<f32> = splats.sh_rest.clone();
    adam_pos.step(&mut cpu_pos, &g_pos_cpu);
    adam_rot.step(&mut cpu_rot, &g_rot_cpu);
    adam_scale.step(&mut cpu_scale, &g_scale_cpu);
    adam_op.step(&mut cpu_op, &g_op_cpu);
    adam_dc.step(&mut cpu_dc, &g_dc_cpu);
    adam_rest.step(&mut cpu_rest, &g_rest_cpu);

    // --- Readback GPU state (auto-unpads to the same flat layout). ----
    gpu_splats.n = n as u32; // (unchanged, but explicit)
    let back = gpu_splats.readback(&ctx);
    let gpu_pos: Vec<f32> = back
        .positions
        .iter()
        .flat_map(|p| [p.x, p.y, p.z])
        .collect();
    let gpu_rot: Vec<f32> = back.rotations.iter().flat_map(|r| *r).collect();
    let gpu_scale: Vec<f32> = back.scales.iter().flat_map(|s| *s).collect();
    let gpu_op: Vec<f32> = back.opacities.clone();
    let gpu_dc: Vec<f32> = back.sh_dc.iter().flat_map(|c| *c).collect();
    let gpu_rest: Vec<f32> = back.sh_rest.clone();

    // Params: tighter than 1e-4 (Adam updates here are O(1e-4..1e-3)).
    assert_close(&cpu_pos, &gpu_pos, 1e-4, "pos");
    assert_close(&cpu_rot, &gpu_rot, 1e-4, "rot");
    assert_close(&cpu_scale, &gpu_scale, 1e-4, "scale");
    assert_close(&cpu_op, &gpu_op, 1e-4, "opacity");
    assert_close(&cpu_dc, &gpu_dc, 1e-4, "sh_dc");
    assert_close(&cpu_rest, &gpu_rest, 1e-4, "sh_rest");

    // Moments: unpad the GPU buffers and compare. Tolerance is tighter
    // (1e-5) because moments are linear in the gradient and don't go
    // through the sqrt/divide that amplifies fp error in the params.
    let m_pos_padded: Vec<[f32; 4]> = ctx.readback(&moments.m_pos, n);
    let v_pos_padded: Vec<[f32; 4]> = ctx.readback(&moments.v_pos, n);
    let m_pos_gpu = unpad_vec4(&m_pos_padded);
    let v_pos_gpu = unpad_vec4(&v_pos_padded);
    assert_close(&adam_pos.m, &m_pos_gpu, 1e-5, "m_pos");
    assert_close(&adam_pos.v, &v_pos_gpu, 1e-5, "v_pos");

    let m_rot_gpu: Vec<f32> = ctx.readback(&moments.m_rot, n * 4);
    let v_rot_gpu: Vec<f32> = ctx.readback(&moments.v_rot, n * 4);
    assert_close(&adam_rot.m, &m_rot_gpu, 1e-5, "m_rot");
    assert_close(&adam_rot.v, &v_rot_gpu, 1e-5, "v_rot");

    let m_scale_padded: Vec<[f32; 4]> = ctx.readback(&moments.m_scale, n);
    let v_scale_padded: Vec<[f32; 4]> = ctx.readback(&moments.v_scale, n);
    let m_scale_gpu = unpad_vec4(&m_scale_padded);
    let v_scale_gpu = unpad_vec4(&v_scale_padded);
    assert_close(&adam_scale.m, &m_scale_gpu, 1e-5, "m_scale");
    assert_close(&adam_scale.v, &v_scale_gpu, 1e-5, "v_scale");

    let m_op_gpu: Vec<f32> = ctx.readback(&moments.m_op, n);
    let v_op_gpu: Vec<f32> = ctx.readback(&moments.v_op, n);
    assert_close(&adam_op.m, &m_op_gpu, 1e-5, "m_op");
    assert_close(&adam_op.v, &v_op_gpu, 1e-5, "v_op");

    let m_dc_padded: Vec<[f32; 4]> = ctx.readback(&moments.m_dc, n);
    let v_dc_padded: Vec<[f32; 4]> = ctx.readback(&moments.v_dc, n);
    let m_dc_gpu = unpad_vec4(&m_dc_padded);
    let v_dc_gpu = unpad_vec4(&v_dc_padded);
    assert_close(&adam_dc.m, &m_dc_gpu, 1e-5, "m_dc");
    assert_close(&adam_dc.v, &v_dc_gpu, 1e-5, "v_dc");

    let m_rest_padded: Vec<f32> = ctx.readback(&moments.m_rest, n * 48);
    let v_rest_padded: Vec<f32> = ctx.readback(&moments.v_rest, n * 48);
    let m_rest_gpu = unpad_rest(&m_rest_padded, n);
    let v_rest_gpu = unpad_rest(&v_rest_padded, n);
    assert_close(&adam_rest.m, &m_rest_gpu, 1e-5, "m_rest");
    assert_close(&adam_rest.v, &v_rest_gpu, 1e-5, "v_rest");
}

/// Element-wise absolute-tolerance compare; on failure dumps the first
/// few mismatched indices to make a diff easy to read.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        assert!(
            d < tol,
            "{label}[{i}] diverge: cpu={} gpu={} (|Δ|={} > {})",
            a[i],
            b[i],
            d,
            tol
        );
    }
}

fn unpad_vec4(padded: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(padded.len() * 3);
    for p in padded {
        out.push(p[0]);
        out.push(p[1]);
        out.push(p[2]);
    }
    out
}

fn unpad_rest(padded: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 45);
    for i in 0..n {
        let base = i * 48;
        out.extend_from_slice(&padded[base..base + 45]);
    }
    out
}
