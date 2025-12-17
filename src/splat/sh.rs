//! Spherical Harmonics computation
//!
//! Real spherical harmonics up to degree 3 with Condon-Shortley phase convention.
//! Used for encoding view-dependent radiance in Gaussian splats.

use glam::Vec3;
use std::f32::consts::PI;

use super::ShCoeffs;

// IMPORTANT: Use the exact polynomial SH basis and constants from the reference
// 3DGS implementation (graphdeco-inria/gaussian-splatting, utils/sh_utils.py).
// Many viewers expect this exact convention; mixing SH conventions yields
// psychedelic/incorrect colors.
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
const SH_C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];

/// Evaluate real spherical harmonic basis function Y_l^m at direction (x, y, z)
///
/// Uses Condon-Shortley phase convention.
/// Direction must be normalized.
pub fn sh_basis(l: u32, m: i32, dir: Vec3) -> f32 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;

    match (l, m) {
        // Degree 0 (1 function)
        (0, 0) => SH_C0,

        // Degree 1 (3 functions)
        (1, -1) => -SH_C1 * y,
        (1, 0) => SH_C1 * z,
        (1, 1) => -SH_C1 * x,

        // Degree 2 (5 functions)
        (2, -2) => SH_C2[0] * x * y,
        (2, -1) => SH_C2[1] * y * z,
        (2, 0) => SH_C2[2] * (2.0 * zz - xx - yy),
        (2, 1) => SH_C2[3] * x * z,
        (2, 2) => SH_C2[4] * (xx - yy),

        // Degree 3 (7 functions)
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

/// Evaluate all SH basis functions up to given degree at direction
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

/// Fit SH coefficients to radiance samples using least squares
///
/// # Arguments
/// * `samples` - Pairs of (view_direction, radiance) where view_direction points FROM splat TO camera
/// * `max_degree` - Maximum SH degree (0-3)
///
/// # Returns
/// Fitted SH coefficients in 3DGS feature space:
/// `color(view) ~= 0.5 + sum_i(coeff_i * Y_i(view))`.
pub fn fit_sh(samples: &[(Vec3, Vec3)], max_degree: u32) -> ShCoeffs {
    let n_coeffs = ((max_degree + 1) * (max_degree + 1)) as usize;
    let mut coeffs = vec![[0.0f32; 3]; n_coeffs];

    if samples.is_empty() || n_coeffs == 0 {
        return ShCoeffs {
            coeffs,
            degree: max_degree,
        };
    }

    // Least-squares fit against the same basis used by common 3DGS viewers.
    // This is much more stable than attempting Monte-Carlo projection when:
    // - sampling isn't truly uniform over the sphere
    // - we only sample a hemisphere
    const MAX_COEFFS: usize = 16; // degree 3
    debug_assert!(n_coeffs <= MAX_COEFFS);

    let mut ata = [[0.0f32; MAX_COEFFS]; MAX_COEFFS];
    let mut atb = [[0.0f32; 3]; MAX_COEFFS];

    for (dir, radiance) in samples {
        let len_sq = dir.length_squared();
        if len_sq < 1e-12 {
            continue;
        }
        let dir = *dir / len_sq.sqrt();
        let basis = sh_basis_all(max_degree, dir);

        // Input is expected in sRGB space in [0, 1], matching bevy_gaussian_splatting.
        // Map to 3DGS feature space (shift by 0.5).
        let b = radiance.clamp(Vec3::ZERO, Vec3::ONE) - Vec3::splat(0.5);

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

    // Small Tikhonov regularization to avoid singular matrices for low sample counts.
    let lambda = 1e-4;
    for i in 0..n_coeffs {
        ata[i][i] += lambda;
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

fn solve_linear_system<const N: usize>(
    a: &mut [[f32; N]; N],
    b: &mut [f32; N],
    n: usize,
) -> [f32; N] {
    // Gauss-Jordan elimination with partial pivoting.
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for r in (col + 1)..n {
            let v = a[r][col].abs();
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

        // Normalize pivot row
        let pivot = a[col][col];
        for c in col..n {
            a[col][c] /= pivot;
        }
        b[col] /= pivot;

        // Eliminate other rows
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor.abs() < 1e-12 {
                continue;
            }
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }

    let mut x = [0.0f32; N];
    for i in 0..n {
        x[i] = b[i];
    }
    x
}

/// Evaluate SH at given direction to reconstruct 3DGS feature value.
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

/// Generate Fibonacci spiral directions on hemisphere
///
/// Returns deterministic, quasi-uniform distribution of directions
/// in local space where Z is up (normal direction).
pub fn fibonacci_hemisphere(n: usize) -> Vec<Vec3> {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden_ratio;

    (0..n)
        .map(|i| {
            // Longitude: golden angle spiral
            let theta = golden_angle * i as f32;

            // Latitude: uniform distribution on hemisphere
            // For hemisphere: z in [0, 1], mapped from index
            let z = 1.0 - (i as f32 + 0.5) / n as f32;
            let z = z.max(0.0); // Ensure hemisphere (z >= 0)

            let r = (1.0 - z * z).sqrt();

            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

/// Generate stratified random directions on hemisphere
pub fn stratified_hemisphere(n_theta: usize, n_phi: usize, jitter: bool) -> Vec<Vec3> {
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    let mut rng = fastrand::Rng::new();

    for i in 0..n_theta {
        for j in 0..n_phi {
            let u = if jitter {
                (i as f32 + rng.f32()) / n_theta as f32
            } else {
                (i as f32 + 0.5) / n_theta as f32
            };

            let v = if jitter {
                (j as f32 + rng.f32()) / n_phi as f32
            } else {
                (j as f32 + 0.5) / n_phi as f32
            };

            // Cosine-weighted hemisphere sampling
            let theta = 2.0 * PI * v;
            let r = u.sqrt();
            let z = (1.0 - u).sqrt();

            dirs.push(Vec3::new(r * theta.cos(), r * theta.sin(), z));
        }
    }

    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_orthonormality() {
        // SH basis functions should be orthonormal
        let n = 1000;
        let dirs = fibonacci_hemisphere(n);
        let weight = 2.0 * PI / n as f32;

        // Test Y_0^0 normalization (should integrate to sqrt(4*pi) over sphere)
        let mut sum = 0.0;
        for dir in &dirs {
            let y00 = sh_basis(0, 0, *dir);
            sum += y00 * y00 * weight;
        }
        // For hemisphere, expect ~0.5 of full sphere integral
        assert!((sum - 0.5).abs() < 0.1, "Y_0^0 integral: {}", sum);
    }

    #[test]
    fn test_fibonacci_coverage() {
        let dirs = fibonacci_hemisphere(64);

        // All directions should be on hemisphere (z >= 0)
        for dir in &dirs {
            assert!(dir.z >= -0.01, "Direction below hemisphere: {:?}", dir);
            let len = dir.length();
            assert!((len - 1.0).abs() < 0.001, "Non-unit direction: {}", len);
        }
    }

    #[test]
    fn test_fit_sh_reconstructs_function() {
        // Build a synthetic SH function in feature space, sample it, and fit back.
        let degree = 3;
        let n_coeffs = ((degree + 1) * (degree + 1)) as usize;

        // Small coefficients to keep the resulting color within [0, 1] after +0.5.
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

        // Compare reconstruction error on the same samples.
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
}
