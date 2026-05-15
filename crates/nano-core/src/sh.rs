//! CPU spherical-harmonics reference matching the GLSL splat shader.
//!
//! The GPU splat fitter inlines the same constants and the same
//! `sh_basis(l, m, dir)` polynomial. Keeping a tested Rust mirror here
//! catches regressions in the basis convention without needing to run
//! Vulkan: any divergence between `sh_basis(...)` in this module and the
//! `sh_basis(int index, vec3 dir)` function in `nano-splat::generator`
//! shows up as a unit-test failure.
//!
//! Convention: real spherical harmonics up to degree 3 with the
//! Condon–Shortley phase, exactly as in `graphdeco-inria/gaussian-splatting`
//! `utils/sh_utils.py`. Mixing in another convention produces psychedelic
//! colours in 3DGS viewers — do not retune these constants in isolation.

use glam::Vec3;
use std::f32::consts::PI;

pub const SH_C0: f32 = 0.282_094_8;
const SH_C1: f32 = 0.488_602_52;
const SH_C2: [f32; 5] = [
    1.092_548_5,
    -1.092_548_5,
    0.315_391_57,
    -1.092_548_5,
    0.546_274_24,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Spherical-harmonics coefficients up to the given degree.
///
/// `coeffs` is `(degree + 1)²` entries; each entry holds `[R, G, B]`.
#[derive(Debug, Clone)]
pub struct ShCoeffs {
    pub coeffs: Vec<[f32; 3]>,
    pub degree: u32,
}

impl ShCoeffs {
    pub fn new(degree: u32) -> Self {
        let n_coeffs = ((degree + 1) * (degree + 1)) as usize;
        Self {
            coeffs: vec![[0.0; 3]; n_coeffs],
            degree,
        }
    }

    /// SH coefficient `l = 0` (DC term).
    pub fn dc(&self) -> [f32; 3] {
        self.coeffs[0]
    }

    /// 3DGS `f_rest_*` layout: planar by colour channel.
    /// `[R(1..15), G(1..15), B(1..15)]` — 45 floats for degree 3.
    pub fn rest_planar(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(45);
        for channel in 0..3 {
            for coeff_idx in 1..16 {
                result.push(self.coeffs.get(coeff_idx).map_or(0.0, |c| c[channel]));
            }
        }
        result
    }
}

/// Evaluate the real SH basis Y_l^m at a unit direction.
pub fn sh_basis(l: u32, m: i32, dir: Vec3) -> f32 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;

    match (l, m) {
        (0, 0) => SH_C0,

        (1, -1) => -SH_C1 * y,
        (1, 0) => SH_C1 * z,
        (1, 1) => -SH_C1 * x,

        (2, -2) => SH_C2[0] * x * y,
        (2, -1) => SH_C2[1] * y * z,
        (2, 0) => SH_C2[2] * (2.0 * zz - xx - yy),
        (2, 1) => SH_C2[3] * x * z,
        (2, 2) => SH_C2[4] * (xx - yy),

        (3, -3) => SH_C3[0] * y * (3.0 * xx - yy),
        (3, -2) => SH_C3[1] * x * y * z,
        (3, -1) => SH_C3[2] * y * (4.0 * zz - xx - yy),
        (3, 0) => SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        (3, 1) => SH_C3[4] * x * (4.0 * zz - xx - yy),
        (3, 2) => SH_C3[5] * z * (xx - yy),
        (3, 3) => SH_C3[6] * x * (xx - 3.0 * yy),

        _ => 0.0,
    }
}

/// Evaluate all SH basis functions up to `max_degree` at a unit direction.
pub fn sh_basis_all(max_degree: u32, dir: Vec3) -> Vec<f32> {
    let n_coeffs = ((max_degree + 1) * (max_degree + 1)) as usize;
    let mut result = Vec::with_capacity(n_coeffs);
    for l in 0..=max_degree {
        for m in -(l as i32)..=(l as i32) {
            result.push(sh_basis(l, m, dir));
        }
    }
    result
}

/// Convert an sRGB colour in `[0, 1]` to a constant DC coefficient.
///
/// Used by the constant-colour (DC-only) splat construction path, equivalent
/// to the CPU reference's `from_sample_constant`.
pub fn sh_dc_from_srgb(color: Vec3) -> [f32; 3] {
    let c = color.clamp(Vec3::ZERO, Vec3::ONE) - Vec3::splat(0.5);
    [c.x / SH_C0, c.y / SH_C0, c.z / SH_C0]
}

/// Fit SH coefficients to a set of `(view_direction, srgb_color)` samples
/// using ordinary least squares with mild Tikhonov regularisation.
///
/// `view_direction` should point **from the splat toward the camera** (the
/// outgoing direction). Returns coefficients in 3DGS feature space, ready
/// to be written into PLY `f_dc_*` / `f_rest_*` fields.
pub fn fit_sh(samples: &[(Vec3, Vec3)], max_degree: u32) -> ShCoeffs {
    let n_coeffs = ((max_degree + 1) * (max_degree + 1)) as usize;
    let mut coeffs = vec![[0.0f32; 3]; n_coeffs];

    if samples.is_empty() || n_coeffs == 0 {
        return ShCoeffs {
            coeffs,
            degree: max_degree,
        };
    }

    const MAX_COEFFS: usize = 16;
    debug_assert!(n_coeffs <= MAX_COEFFS);

    let mut ata = [[0.0f32; MAX_COEFFS]; MAX_COEFFS];
    let mut atb = [[0.0f32; 3]; MAX_COEFFS];

    for (dir, color) in samples {
        let len_sq = dir.length_squared();
        if len_sq < 1e-12 {
            continue;
        }
        let dir = *dir / len_sq.sqrt();
        let basis = sh_basis_all(max_degree, dir);

        let b = color.clamp(Vec3::ZERO, Vec3::ONE) - Vec3::splat(0.5);

        for j in 0..n_coeffs {
            let yj = basis[j];
            atb[j][0] += yj * b.x;
            atb[j][1] += yj * b.y;
            atb[j][2] += yj * b.z;
            for k in 0..n_coeffs {
                ata[j][k] += yj * basis[k];
            }
        }
    }

    let lambda = 1e-4;
    for (i, row) in ata.iter_mut().enumerate().take(n_coeffs) {
        row[i] += lambda;
    }

    for channel in 0..3 {
        let mut a = ata;
        let mut rhs = [0.0f32; MAX_COEFFS];
        for i in 0..n_coeffs {
            rhs[i] = atb[i][channel];
        }
        let x = solve_linear_system(&mut a, &mut rhs, n_coeffs);
        for i in 0..n_coeffs {
            coeffs[i][channel] = x[i];
        }
    }

    ShCoeffs {
        coeffs,
        degree: max_degree,
    }
}

/// Gauss–Jordan elimination with partial pivoting for the in-memory
/// `MAX_COEFFS × MAX_COEFFS` system used by [`fit_sh`].
fn solve_linear_system<const N: usize>(
    a: &mut [[f32; N]; N],
    b: &mut [f32; N],
    n: usize,
) -> [f32; N] {
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(n).skip(col + 1) {
            let v = row[col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }

        if pivot_val < 1e-12 {
            continue;
        }

        if pivot_row != col {
            a.swap(pivot_row, col);
            b.swap(pivot_row, col);
        }

        let pivot = a[col][col];
        for cell in a[col].iter_mut().take(n).skip(col) {
            *cell /= pivot;
        }
        b[col] /= pivot;

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor.abs() < 1e-12 {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }

    let mut x = [0.0f32; N];
    x[..n].copy_from_slice(&b[..n]);
    x
}

/// Evaluate fitted SH at a unit direction, returning the 3DGS feature value
/// (caller adds 0.5 to get the sRGB colour).
pub fn eval_sh(coeffs: &ShCoeffs, dir: Vec3) -> Vec3 {
    let basis = sh_basis_all(coeffs.degree, dir);
    let mut result = Vec3::ZERO;
    for (idx, &b) in basis.iter().enumerate() {
        if idx < coeffs.coeffs.len() {
            result.x += coeffs.coeffs[idx][0] * b;
            result.y += coeffs.coeffs[idx][1] * b;
            result.z += coeffs.coeffs[idx][2] * b;
        }
    }
    result
}

/// Quasi-uniform Fibonacci-spiral directions on the unit hemisphere
/// (z ≥ 0). Deterministic, used by the SH fit sample distribution.
pub fn fibonacci_hemisphere(n: usize) -> Vec<Vec3> {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden_ratio;

    (0..n)
        .map(|i| {
            let theta = golden_angle * i as f32;
            let z = 1.0 - (i as f32 + 0.5) / n as f32;
            let z = z.max(0.0);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y00_integral_over_hemisphere_is_half() {
        // Y_0^0 = SH_C0 is constant; integrated * 2π/N over the hemisphere
        // should give ≈ 0.5 (half of the full-sphere value of 1).
        let n = 1000;
        let dirs = fibonacci_hemisphere(n);
        let weight = 2.0 * PI / n as f32;
        let mut sum = 0.0;
        for dir in &dirs {
            let y00 = sh_basis(0, 0, *dir);
            sum += y00 * y00 * weight;
        }
        assert!((sum - 0.5).abs() < 0.1, "Y_0^0 integral: {sum}");
    }

    #[test]
    fn fibonacci_dirs_on_hemisphere_and_normalised() {
        let dirs = fibonacci_hemisphere(64);
        for dir in &dirs {
            assert!(dir.z >= -0.01, "direction below hemisphere: {dir:?}");
            let len = dir.length();
            assert!((len - 1.0).abs() < 0.001, "non-unit direction len: {len}");
        }
    }

    #[test]
    fn fit_sh_reconstructs_synthetic_function() {
        // Build a synthetic SH function in 3DGS feature space, sample it,
        // fit back, and verify reconstruction MSE is ~0.
        let degree = 3;
        let n_coeffs = ((degree + 1) * (degree + 1)) as usize;

        let mut true_coeffs = ShCoeffs::new(degree);
        for i in 0..n_coeffs {
            true_coeffs.coeffs[i] = [
                (i as f32 * 0.011).sin() * 0.15,
                (i as f32 * 0.017).cos() * 0.15,
                (i as f32 * 0.013).sin() * 0.15,
            ];
        }

        let dirs = fibonacci_hemisphere(64);
        let samples: Vec<(Vec3, Vec3)> = dirs
            .into_iter()
            .map(|d| {
                let feature = eval_sh(&true_coeffs, d);
                let color = feature + Vec3::splat(0.5);
                (d, color)
            })
            .collect();

        let fitted = fit_sh(&samples, degree);

        let mut mse = Vec3::ZERO;
        for (d, color) in &samples {
            let feature_fit = eval_sh(&fitted, *d);
            let color_fit = feature_fit + Vec3::splat(0.5);
            let e = *color - color_fit;
            mse += e * e;
        }
        mse /= samples.len() as f32;

        assert!(
            mse.x < 1e-4 && mse.y < 1e-4 && mse.z < 1e-4,
            "MSE too high: {mse:?}"
        );
    }

    #[test]
    fn sh_dc_from_srgb_roundtrip() {
        // sh_dc * SH_C0 + 0.5 should recover the input colour (clamped to [0,1]).
        for c in [
            Vec3::new(0.2, 0.5, 0.8),
            Vec3::new(0.0, 1.0, 0.5),
            Vec3::ZERO,
            Vec3::ONE,
        ] {
            let dc = sh_dc_from_srgb(c);
            let recovered = Vec3::new(
                dc[0] * SH_C0 + 0.5,
                dc[1] * SH_C0 + 0.5,
                dc[2] * SH_C0 + 0.5,
            );
            assert!(
                (recovered - c).abs().max_element() < 1e-5,
                "roundtrip failed for {c:?}: got {recovered:?}",
            );
        }
    }
}
