//! Per-frame loss functions for the 3DGS training pipeline.
//!
//! Two ingredients combine into the canonical Inria 3DGS objective:
//!
//! - **MSE** — L2 over per-pixel RGB. Sharp where pixels disagree.
//!   Drives the bulk of early training but flattens out once the splat
//!   cloud is locally correct.
//! - **SSIM** — Wang 2004 structural similarity over an 11×11 Gaussian
//!   window (σ = 1.5, C1 = 0.01², C2 = 0.03²). Captures local
//!   luminance / contrast / structure correlation; gives the perceptual
//!   nudge that polishes the last ~20% of convergence.
//!
//! The combined loss is `L = (1 − λ) · MSE + λ · (1 − SSIM)` with
//! `λ ≈ 0.2`. (Inria uses `λ · (1 − SSIM)/2` — a constant scale we fold
//! into the user-facing λ; the actual gradient direction is identical.)
//!
//! All gradients are with respect to the per-pixel predicted RGB and
//! match the existing `dL/dC` contract: a `Vec<Vec3>` indexed
//! `y * width + x`, ready to be uploaded to the GPU backward pass.

use glam::Vec3;

/// 11×11 separable Gaussian window with σ = 1.5 — Inria default.
const SSIM_WINDOW_SIZE: usize = 11;
const SSIM_SIGMA: f32 = 1.5;
/// `(0.01)²` and `(0.03)²` — the SSIM stabiliser constants for
/// 8-bit-range images (we treat HDR `[0, 1]`-ish radiance the same way;
/// raising the constants slightly damps gradient noise on bright
/// pixels).
const SSIM_C1: f32 = 0.0001;
const SSIM_C2: f32 = 0.0009;

/// Build the 1D Gaussian kernel `g[k] = exp(-(k - c)² / (2σ²))`
/// normalised to sum 1. Called once per loss invocation — the 11-tap
/// kernel is so cheap that caching it would be premature.
fn gauss_kernel_1d() -> [f32; SSIM_WINDOW_SIZE] {
    let mut k = [0.0_f32; SSIM_WINDOW_SIZE];
    let centre = (SSIM_WINDOW_SIZE as f32 - 1.0) * 0.5;
    let two_sigma_sq = 2.0 * SSIM_SIGMA * SSIM_SIGMA;
    let mut sum = 0.0_f32;
    for (i, slot) in k.iter_mut().enumerate() {
        let d = i as f32 - centre;
        let v = (-d * d / two_sigma_sq).exp();
        *slot = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Reflect an out-of-range index back into `[0, n)`. This matches the
/// "reflect-101" padding used by Inria's `ssim.py` (the most common
/// convention for SSIM windows on bounded images).
#[inline]
fn reflect_index(i: i32, n: i32) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut x = i.rem_euclid(period);
    if x >= n {
        x = period - x;
    }
    x as usize
}

/// Separable 2D Gaussian blur with reflection padding. Per-channel
/// scalar `f32` input → blurred `f32` output (caller invokes per
/// channel). One horizontal pass + one vertical pass = O(W·H·K) work
/// for an 11-tap kernel.
fn gauss_blur(src: &[f32], width: u32, height: u32, kernel: &[f32; SSIM_WINDOW_SIZE]) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let radius = (SSIM_WINDOW_SIZE / 2) as i32;
    let mut tmp = vec![0.0_f32; w * h];
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let xi = reflect_index(x as i32 + k as i32 - radius, w as i32);
                acc += src[y * w + xi] * kw;
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass.
    let mut dst = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let yi = reflect_index(y as i32 + k as i32 - radius, h as i32);
                acc += tmp[yi * w + x] * kw;
            }
            dst[y * w + x] = acc;
        }
    }
    dst
}

/// Adjoint of [`gauss_blur`] (the transpose of the linear convolution
/// operator). Required for backprop through reflection-padded blurs:
/// reflection makes the conv *non*-self-adjoint near the boundary (two
/// taps can map to the same source pixel), so naively re-using
/// [`gauss_blur`] in the backward pass overcounts edge contributions by
/// up to ~2×. The fix is a scatter formulation — for each (output, tap)
/// pair, push `out[y] · kernel[k]` to the source pixel via the same
/// `reflect_index` map the forward used.
///
/// Forward composition is `H_pass ∘ V_pass` (horizontal-then-vertical
/// in the implementation order), so the adjoint must be `Vᵀ ∘ Hᵀ` —
/// vertical-adjoint first, then horizontal-adjoint. Composition order
/// reverses under transpose; we exploit that here.
///
/// Mathematically: forward `μ[j] = Σ_i W[j,i] · x[i]` with
/// `W[j,i] = Σ_{k: reflect(j+k-radius)=i} G[k]` (sum of taps that hit
/// `i`). The adjoint `(Wᵀ·f)[i] = Σ_j f[j]·W[j,i]` is exactly the
/// scatter below.
fn gauss_blur_adjoint(
    src: &[f32],
    width: u32,
    height: u32,
    kernel: &[f32; SSIM_WINDOW_SIZE],
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let radius = (SSIM_WINDOW_SIZE / 2) as i32;
    // Adjoint of the vertical pass — scatter rather than gather.
    let mut tmp = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let s = src[y * w + x];
            for (k, &kw) in kernel.iter().enumerate() {
                let yi = reflect_index(y as i32 + k as i32 - radius, h as i32);
                tmp[yi * w + x] += s * kw;
            }
        }
    }
    // Adjoint of the horizontal pass.
    let mut dst = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let s = tmp[y * w + x];
            for (k, &kw) in kernel.iter().enumerate() {
                let xi = reflect_index(x as i32 + k as i32 - radius, w as i32);
                dst[y * w + xi] += s * kw;
            }
        }
    }
    dst
}

/// Mean-squared error over `Vec3` framebuffers — sums squared
/// per-channel differences, divides by total scalar count
/// `W · H · 3`. Returns the scalar loss and the per-pixel
/// `dL/dC = 2(pred − target) / (W·H·3)` gradient.
///
/// Bit-equivalent to the inline MSE that lived in `train.rs` before the
/// loss module was extracted.
pub fn mse_loss_grad(pred: &[Vec3], target: &[Vec3], width: u32, height: u32) -> (f32, Vec<Vec3>) {
    assert_eq!(pred.len(), target.len());
    assert_eq!(pred.len(), (width as usize) * (height as usize));
    let n_scalars = (width as f32) * (height as f32) * 3.0;
    // `f64` accumulator avoids catastrophic cancellation as W·H grows;
    // the per-pixel squared diffs are tiny but there are millions of them.
    let mut sum = 0.0_f64;
    let mut grad = Vec::with_capacity(pred.len());
    let inv_n = 2.0 / n_scalars;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = *p - *t;
        sum += (d.x * d.x + d.y * d.y + d.z * d.z) as f64;
        grad.push(d * inv_n);
    }
    let loss = (sum / (pred.len() as f64 * 3.0)) as f32;
    (loss, grad)
}

/// Per-channel SSIM forward + analytic gradient.
///
/// Returns `(loss = 1 - mean(SSIM_map), per-pixel dL/dx)` for a single
/// channel. The gradient simplifies to a cascade of three Gaussian
/// convolutions over precomputed quotient-rule coefficients — derived
/// from the standard SSIM-map quotient:
///
/// ```text
/// SSIM = (A · B) / (C · D)
/// A = 2·μ_x·μ_y + C1,      C = μ_x² + μ_y² + C1
/// B = 2·σ_xy + C2,         D = σ_x² + σ_y² + C2
/// ```
///
/// Taking `d/dx_i` and grouping per intermediate (μ_x, σ_x², σ_xy):
///
/// ```text
/// dSSIM/dx_i = G * [ 2·μ_y · (B − A · D/C) / (C · D) ]      ← μ_x term
///            + 2·x_i · G * [ −A·B / (C · D²) + ?? ]
/// ```
///
/// Rearranging into the canonical pytorch-ssim form gives:
///
/// ```text
/// dSSIM/dx = 2 · G * (P · (μ_y − μ_x · 0)
///                   + Q · (μ_x · μ_x − μ_x²)
///                   + R · (μ_y − μ_x))
/// ```
///
/// — which is messy in prose. The implementation below derives it
/// term-by-term: see the per-buffer comments. Validated against finite
/// differences in `tests/loss_tests.rs` (tolerance 5e-3).
fn ssim_loss_grad_channel(
    pred: &[f32],
    target: &[f32],
    width: u32,
    height: u32,
    kernel: &[f32; SSIM_WINDOW_SIZE],
) -> (f32, Vec<f32>) {
    let n = pred.len();
    debug_assert_eq!(n, target.len());

    // Pointwise products needed by the SSIM expansion.
    let xx: Vec<f32> = pred.iter().map(|v| v * v).collect();
    let yy: Vec<f32> = target.iter().map(|v| v * v).collect();
    let xy: Vec<f32> = pred.iter().zip(target.iter()).map(|(a, b)| a * b).collect();

    // Gaussian-smoothed primaries.
    let mu_x = gauss_blur(pred, width, height, kernel);
    let mu_y = gauss_blur(target, width, height, kernel);
    let g_xx = gauss_blur(&xx, width, height, kernel);
    let g_yy = gauss_blur(&yy, width, height, kernel);
    let g_xy = gauss_blur(&xy, width, height, kernel);

    // SSIM intermediates per pixel.
    let mut a = vec![0.0_f32; n]; // 2·μ_x·μ_y + C1
    let mut b = vec![0.0_f32; n]; // 2·σ_xy + C2
    let mut c_ = vec![0.0_f32; n]; // μ_x² + μ_y² + C1
    let mut d = vec![0.0_f32; n]; // σ_x² + σ_y² + C2
    let mut ssim_map = vec![0.0_f32; n];
    let mut sum_map = 0.0_f64;
    for i in 0..n {
        let mux = mu_x[i];
        let muy = mu_y[i];
        let mux2 = mux * mux;
        let muy2 = muy * muy;
        let sxx = g_xx[i] - mux2;
        let syy = g_yy[i] - muy2;
        let sxy = g_xy[i] - mux * muy;
        let av = 2.0 * mux * muy + SSIM_C1;
        let bv = 2.0 * sxy + SSIM_C2;
        let cv = mux2 + muy2 + SSIM_C1;
        let dv = sxx + syy + SSIM_C2;
        a[i] = av;
        b[i] = bv;
        c_[i] = cv;
        d[i] = dv;
        let s = (av * bv) / (cv * dv);
        ssim_map[i] = s;
        sum_map += s as f64;
    }
    let mean_ssim = (sum_map / n as f64) as f32;
    let loss = 1.0 - mean_ssim;

    // Analytic gradient of `loss = 1 - mean(SSIM_map)` w.r.t. each `x_i`.
    //
    // SSIM_map = (A·B) / (C·D). Let
    //   P = ∂SSIM/∂μ_x  = 2·μ_y·B / (C·D) − 2·μ_x·A·B / (C²·D)
    //                   = 2·[ μ_y·B·C − μ_x·A·B ] / (C²·D)
    //                   = 2·B·[ μ_y·C − μ_x·A ] / (C²·D)
    //   Q = ∂SSIM/∂σ_x² = −A·B / (C·D²)
    //   R = ∂SSIM/∂σ_xy = 2·A / (C·D)         (B = 2·σ_xy + C2 ⇒ dB/dσ_xy = 2)
    //
    // The dependence on `x_i` flows through:
    //   ∂μ_x/∂x_i  = G(i, .) — a 1-hot conv response
    //   ∂σ_x²/∂x_i = 2·(x_i · G(i,.) − μ_x · G(i,.))   ← from g_xx − μ_x²
    //   ∂σ_xy/∂x_i = G(i,.) · y − G(i,.) · μ_y
    //
    // So per-pixel:
    //   dSSIM/dx_i = G * (P)(i)
    //              + 2·x_i · G * (Q)(i) − 2·μ_x(i) · G * (Q)(i)
    //              + y_i · G * (R)(i) − μ_y(i) · G * (R)(i)
    //
    // Equivalently, after collecting "multiplied by x_i / y_i" vs.
    // "smoothed alone" terms, the canonical form below uses three
    // Gaussian smoothings of `P_term`, `Q_term`, `R_term`. The
    // intermediate `_term` buffers fold in the `−μ_x` / `−μ_y` mirror
    // contributions so the final per-pixel expression is linear in
    // `x_i` / `y_i`.
    let mut p_term = vec![0.0_f32; n];
    let mut q_term = vec![0.0_f32; n];
    let mut r_term = vec![0.0_f32; n];
    for i in 0..n {
        let av = a[i];
        let bv = b[i];
        let cv = c_[i];
        let dv = d[i];
        let cd = cv * dv;
        // ∂SSIM/∂μ_x   = 2·B·(μ_y·C − μ_x·A) / (C² · D)
        // ∂SSIM/∂σ_x²  = −A·B / (C · D²)
        // ∂SSIM/∂σ_xy  =  2·A / (C · D)              ← B = 2·σ_xy + C2 ⇒ dB/dσ_xy = 2
        let p = 2.0 * bv * (mu_y[i] * cv - mu_x[i] * av) / (cv * cd);
        let q = -av * bv / (cv * dv * dv);
        let r = 2.0 * av / cd;
        p_term[i] = p;
        q_term[i] = q;
        r_term[i] = r;
    }
    // Backward smoothings must use the conv *adjoint* — not the forward
    // conv — because reflection padding makes the forward conv non-self-
    // adjoint near the image boundary. See `gauss_blur_adjoint` docs.
    let smooth_p = gauss_blur_adjoint(&p_term, width, height, kernel);
    let smooth_q = gauss_blur_adjoint(&q_term, width, height, kernel);
    let smooth_r = gauss_blur_adjoint(&r_term, width, height, kernel);

    // (μ_x · Q) and (μ_y · R) propagate the centring contributions back
    // through the adjoint. The −μ_x term comes from
    // `∂σ_x²/∂x_i = 2·(x_i − μ_x)·G(i,·)` — the `σ_x² = g_xx − μ_x²`
    // unroll subtracts μ_x's own derivative, leaving the `(x_i − μ_x)`
    // residual that splits into the `pred[i]·smooth_q − smooth(μ_x·q)`
    // pair below. Same shape for σ_xy giving `y_i · ... − smooth(μ_y·r)`.
    let mux_q: Vec<f32> = (0..n).map(|i| mu_x[i] * q_term[i]).collect();
    let muy_r: Vec<f32> = (0..n).map(|i| mu_y[i] * r_term[i]).collect();
    let smooth_mux_q = gauss_blur_adjoint(&mux_q, width, height, kernel);
    let smooth_muy_r = gauss_blur_adjoint(&muy_r, width, height, kernel);

    let inv_n = 1.0 / (n as f32);
    let mut grad = vec![0.0_f32; n];
    for i in 0..n {
        // dSSIM/dx_i (per pixel) — pieced together from the three
        // intermediate gradients above. The factor 2 on the Q-row
        // matches `∂σ_x²/∂x = 2·(x − μ_x)`.
        let dssim = smooth_p[i]
            + 2.0 * (pred[i] * smooth_q[i] - smooth_mux_q[i])
            + (target[i] * smooth_r[i] - smooth_muy_r[i]);
        // loss = 1 − mean(SSIM) ⇒ dLoss/dx = −(1/N) · dSSIM/dx.
        grad[i] = -inv_n * dssim;
    }

    (loss, grad)
}

/// SSIM loss over `Vec3` framebuffers — averages SSIM over R, G, B
/// channels. Returns `(loss, dL/dC)` ready for the backward rasteriser.
///
/// `loss = 1 - (SSIM_R + SSIM_G + SSIM_B) / 3` ∈ `[0, 1]` (0 = perfect).
pub fn ssim_loss_grad(pred: &[Vec3], target: &[Vec3], width: u32, height: u32) -> (f32, Vec<Vec3>) {
    assert_eq!(pred.len(), target.len());
    assert_eq!(pred.len(), (width as usize) * (height as usize));
    let kernel = gauss_kernel_1d();
    let n = pred.len();

    // De-interleave channels — the per-channel helper expects flat
    // scalar planes for the convolution to be straightforward.
    let mut px = vec![0.0_f32; n];
    let mut py = vec![0.0_f32; n];
    let mut pz = vec![0.0_f32; n];
    let mut tx = vec![0.0_f32; n];
    let mut ty = vec![0.0_f32; n];
    let mut tz = vec![0.0_f32; n];
    for i in 0..n {
        px[i] = pred[i].x;
        py[i] = pred[i].y;
        pz[i] = pred[i].z;
        tx[i] = target[i].x;
        ty[i] = target[i].y;
        tz[i] = target[i].z;
    }

    let (lx, gx) = ssim_loss_grad_channel(&px, &tx, width, height, &kernel);
    let (ly, gy) = ssim_loss_grad_channel(&py, &ty, width, height, &kernel);
    let (lz, gz) = ssim_loss_grad_channel(&pz, &tz, width, height, &kernel);

    // Channel-average — the per-channel losses each carry the (1 − SSIM)
    // shape and the per-channel gradients each carry the −(1/N) factor,
    // so a simple mean here gives the correct combined-channel result.
    let loss = (lx + ly + lz) / 3.0;
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        grad.push(Vec3::new(gx[i] / 3.0, gy[i] / 3.0, gz[i] / 3.0));
    }
    (loss, grad)
}

/// Combined `(1 − λ) · MSE + λ · (1 − SSIM)` loss with analytic
/// gradient. Returns `(total, mse_component, ssim_component, dL/dC)`.
///
/// The per-pixel gradient is the λ-weighted sum of the two component
/// gradients — by linearity of derivatives, no further work is needed
/// beyond computing each loss once.
pub fn combined_loss_grad(
    pred: &[Vec3],
    target: &[Vec3],
    width: u32,
    height: u32,
    lambda: f32,
) -> (f32, f32, f32, Vec<Vec3>) {
    let lambda = lambda.clamp(0.0, 1.0);
    let (mse, mse_grad) = mse_loss_grad(pred, target, width, height);
    // Skip the relatively expensive SSIM pass entirely when λ = 0 —
    // matches the historical pure-MSE behaviour bit-for-bit.
    if lambda == 0.0 {
        return (mse, mse, 0.0, mse_grad);
    }
    let (ssim, ssim_grad) = ssim_loss_grad(pred, target, width, height);
    if lambda == 1.0 {
        return (ssim, 0.0, ssim, ssim_grad);
    }
    let one_minus = 1.0 - lambda;
    let total = one_minus * mse + lambda * ssim;
    let mut grad = Vec::with_capacity(pred.len());
    for (m, s) in mse_grad.iter().zip(ssim_grad.iter()) {
        grad.push(*m * one_minus + *s * lambda);
    }
    (total, mse, ssim, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(width: u32, height: u32, scale: f32) -> Vec<Vec3> {
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = if (x + y) % 2 == 0 { scale } else { 0.0 };
                out.push(Vec3::new(v, v * 0.5, v * 0.25));
            }
        }
        out
    }

    #[test]
    fn ssim_is_one_for_identical_images() {
        let img = checkerboard(8, 8, 0.7);
        let (loss, grad) = ssim_loss_grad(&img, &img, 8, 8);
        assert!(loss.abs() < 1e-5, "ssim_loss(x,x) should be 0, got {loss}");
        // Gradient at the global minimum should be near zero.
        for g in &grad {
            assert!(
                g.length() < 1e-4,
                "ssim grad at minimum should be ~0, got {g:?}"
            );
        }
    }

    #[test]
    fn ssim_grows_with_noise() {
        let img = checkerboard(8, 8, 0.7);
        let mut noisy = img.clone();
        // Deterministic pseudo-noise — simple LCG.
        let mut s: u32 = 1;
        for px in &mut noisy {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let n = ((s >> 16) as f32 / 65535.0 - 0.5) * 0.4;
            *px += Vec3::splat(n);
        }
        let (loss_clean, _) = ssim_loss_grad(&img, &img, 8, 8);
        let (loss_noisy, _) = ssim_loss_grad(&noisy, &img, 8, 8);
        assert!(
            loss_noisy > loss_clean,
            "noisy image should have larger SSIM loss ({loss_noisy} vs {loss_clean})"
        );
    }

    /// Per-channel finite-difference check against the analytic SSIM
    /// gradient. 8×8 keeps the O(W·H) perturbation cost manageable
    /// (~192 forward passes — sub-second).
    #[test]
    fn ssim_gradient_matches_finite_diff() {
        let width = 8;
        let height = 8;
        let n = (width * height) as usize;
        // Two distinct images so the gradient is non-trivial.
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32 / n as f32;
            pred.push(Vec3::new(0.3 + 0.4 * f, 0.5 - 0.3 * f, 0.2 + 0.6 * f));
            target.push(Vec3::new(0.4 + 0.3 * f, 0.6 - 0.4 * f, 0.3 + 0.5 * f));
        }
        let (_, analytic) = ssim_loss_grad(&pred, &target, width, height);

        let eps = 1e-3_f32;
        for i in 0..n {
            for ch in 0..3 {
                let mut p_plus = pred.clone();
                let mut p_minus = pred.clone();
                let bump = |v: &mut Vec3, c: usize, e: f32| match c {
                    0 => v.x += e,
                    1 => v.y += e,
                    _ => v.z += e,
                };
                bump(&mut p_plus[i], ch, eps);
                bump(&mut p_minus[i], ch, -eps);
                let (l_plus, _) = ssim_loss_grad(&p_plus, &target, width, height);
                let (l_minus, _) = ssim_loss_grad(&p_minus, &target, width, height);
                let fd = (l_plus - l_minus) / (2.0 * eps);
                let an = match ch {
                    0 => analytic[i].x,
                    1 => analytic[i].y,
                    _ => analytic[i].z,
                };
                let diff = (fd - an).abs();
                // SSIM gradients are small in magnitude; absolute
                // tolerance is more stable than relative here.
                assert!(
                    diff < 5e-3,
                    "ssim grad mismatch at pixel {i} ch {ch}: analytic={an}, fd={fd}, |diff|={diff}"
                );
            }
        }
    }

    #[test]
    fn combined_lambda_zero_equals_mse() {
        let width = 8;
        let height = 8;
        let n = (width * height) as usize;
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32 / n as f32;
            pred.push(Vec3::new(f, 1.0 - f, 0.3));
            target.push(Vec3::new(f * 0.9, 1.0 - f * 1.1, 0.4));
        }
        let (mse, mse_grad) = mse_loss_grad(&pred, &target, width, height);
        let (total, mse_c, ssim_c, grad) = combined_loss_grad(&pred, &target, width, height, 0.0);
        assert_eq!(total, mse);
        assert_eq!(mse_c, mse);
        assert_eq!(ssim_c, 0.0);
        for (a, b) in grad.iter().zip(mse_grad.iter()) {
            assert!((a.x - b.x).abs() < 1e-9);
            assert!((a.y - b.y).abs() < 1e-9);
            assert!((a.z - b.z).abs() < 1e-9);
        }
    }

    #[test]
    fn combined_lambda_one_equals_ssim() {
        let width = 8;
        let height = 8;
        let n = (width * height) as usize;
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32 / n as f32;
            pred.push(Vec3::new(f, 1.0 - f, 0.3));
            target.push(Vec3::new(f * 0.9, 1.0 - f * 1.1, 0.4));
        }
        let (ssim, ssim_grad) = ssim_loss_grad(&pred, &target, width, height);
        let (total, mse_c, ssim_c, grad) = combined_loss_grad(&pred, &target, width, height, 1.0);
        assert_eq!(total, ssim);
        assert_eq!(mse_c, 0.0);
        assert_eq!(ssim_c, ssim);
        for (a, b) in grad.iter().zip(ssim_grad.iter()) {
            assert!((a.x - b.x).abs() < 1e-9);
            assert!((a.y - b.y).abs() < 1e-9);
            assert!((a.z - b.z).abs() < 1e-9);
        }
    }

    #[test]
    fn combined_gradient_is_weighted_sum() {
        let width = 8;
        let height = 8;
        let n = (width * height) as usize;
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32 / n as f32;
            pred.push(Vec3::new(0.2 + f, 0.4, 0.5 - f * 0.3));
            target.push(Vec3::new(0.25 + f * 0.9, 0.5, 0.55 - f * 0.2));
        }
        let lambda = 0.2_f32;
        let (_, mse_grad) = mse_loss_grad(&pred, &target, width, height);
        let (_, ssim_grad) = ssim_loss_grad(&pred, &target, width, height);
        let (_, _, _, grad) = combined_loss_grad(&pred, &target, width, height, lambda);
        for i in 0..n {
            let expected = mse_grad[i] * (1.0 - lambda) + ssim_grad[i] * lambda;
            let diff = (grad[i] - expected).length();
            assert!(diff < 1e-6, "combined grad mismatch at {i}: |diff|={diff}");
        }
    }
}
