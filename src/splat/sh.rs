//! Spherical Harmonics computation
//!
//! Real spherical harmonics up to degree 3 with Condon-Shortley phase convention.
//! Used for encoding view-dependent radiance in Gaussian splats.

use glam::Vec3;
use std::f32::consts::PI;

use super::ShCoeffs;

// Pre-computed SH normalization constants
// Y_l^m = K_l^m * P_l^|m|(cos(theta)) * e^(im*phi)
// For real SH: Y_l^m = K_l^m * P_l^|m|(z) * {cos(m*phi) or sin(|m|*phi)}

const SH_C0: f32 = 0.28209479177387814;      // 1 / (2*sqrt(pi))

const SH_C1: f32 = 0.4886025119029199;       // sqrt(3) / (2*sqrt(pi))

const SH_C2_0: f32 = 0.31539156525252005;    // sqrt(5) / (4*sqrt(pi))
const SH_C2_1: f32 = 1.0925484305920792;     // sqrt(15) / (2*sqrt(pi))
const SH_C2_2: f32 = 0.5462742152960396;     // sqrt(15) / (4*sqrt(pi))

const SH_C3_0: f32 = 0.37317633259011539;    // sqrt(7) / (4*sqrt(pi))
const SH_C3_1: f32 = 0.45704579946446572;    // sqrt(42) / (8*sqrt(pi))
const SH_C3_2: f32 = 1.4453057213202769;     // sqrt(105) / (4*sqrt(pi))
const SH_C3_3: f32 = 0.5900435899266435;     // sqrt(70) / (8*sqrt(pi))

/// Evaluate real spherical harmonic basis function Y_l^m at direction (x, y, z)
/// 
/// Uses Condon-Shortley phase convention.
/// Direction must be normalized.
pub fn sh_basis(l: u32, m: i32, dir: Vec3) -> f32 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    
    match (l, m) {
        // Degree 0 (1 function)
        (0, 0) => SH_C0,
        
        // Degree 1 (3 functions)
        (1, -1) => SH_C1 * y,
        (1, 0) => SH_C1 * z,
        (1, 1) => SH_C1 * x,
        
        // Degree 2 (5 functions)
        (2, -2) => SH_C2_1 * x * y,
        (2, -1) => SH_C2_1 * y * z,
        (2, 0) => SH_C2_0 * (3.0 * z * z - 1.0),
        (2, 1) => SH_C2_1 * x * z,
        (2, 2) => SH_C2_2 * (x * x - y * y),
        
        // Degree 3 (7 functions)
        (3, -3) => SH_C3_3 * y * (3.0 * x * x - y * y),
        (3, -2) => SH_C3_2 * x * y * z,
        (3, -1) => SH_C3_1 * y * (5.0 * z * z - 1.0),
        (3, 0) => SH_C3_0 * z * (5.0 * z * z - 3.0),
        (3, 1) => SH_C3_1 * x * (5.0 * z * z - 1.0),
        (3, 2) => SH_C3_2 * z * (x * x - y * y),
        (3, 3) => SH_C3_3 * x * (x * x - 3.0 * y * y),
        
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

/// Project radiance samples onto SH basis using Monte Carlo integration
///
/// # Arguments
/// * `samples` - Pairs of (view_direction, radiance) where view_direction points FROM camera TO point
/// * `max_degree` - Maximum SH degree (0-3)
///
/// # Returns
/// Fitted SH coefficients
pub fn fit_sh(samples: &[(Vec3, Vec3)], max_degree: u32) -> ShCoeffs {
    let n_coeffs = ((max_degree + 1) * (max_degree + 1)) as usize;
    let mut coeffs = vec![[0.0f32; 3]; n_coeffs];
    
    if samples.is_empty() {
        return ShCoeffs { coeffs, degree: max_degree };
    }
    
    let n = samples.len() as f32;
    
    // For uniform hemisphere sampling:
    // Integral approximation: (2*pi / N) * sum(f(w) * Y_lm(w))
    // But we need to be careful about the normalization
    
    // Simple approach: just average the radiance weighted by SH basis
    // This gives us coefficients that reconstruct the function
    for (dir, radiance) in samples {
        let basis = sh_basis_all(max_degree, *dir);
        
        for (idx, &b) in basis.iter().enumerate() {
            // Weight by solid angle of hemisphere / N samples
            // For hemisphere: 2*pi steradians
            let weight = 2.0 * PI / n;
            coeffs[idx][0] += radiance.x * b * weight;
            coeffs[idx][1] += radiance.y * b * weight;
            coeffs[idx][2] += radiance.z * b * weight;
        }
    }
    
    ShCoeffs { coeffs, degree: max_degree }
}

/// Evaluate SH at given direction to reconstruct radiance
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
    
    // Clamp negative values (can happen due to ringing)
    result.max(Vec3::ZERO)
}

/// Generate Fibonacci spiral directions on hemisphere
/// 
/// Returns deterministic, quasi-uniform distribution of directions
/// in local space where Z is up (normal direction).
pub fn fibonacci_hemisphere(n: usize) -> Vec<Vec3> {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden_ratio;
    
    (0..n).map(|i| {
        // Longitude: golden angle spiral
        let theta = golden_angle * i as f32;
        
        // Latitude: uniform distribution on hemisphere
        // For hemisphere: z in [0, 1], mapped from index
        let z = 1.0 - (i as f32 + 0.5) / n as f32;
        let z = z.max(0.0); // Ensure hemisphere (z >= 0)
        
        let r = (1.0 - z * z).sqrt();
        
        Vec3::new(
            r * theta.cos(),
            r * theta.sin(),
            z,
        )
    }).collect()
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
            
            dirs.push(Vec3::new(
                r * theta.cos(),
                r * theta.sin(),
                z,
            ));
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
}
