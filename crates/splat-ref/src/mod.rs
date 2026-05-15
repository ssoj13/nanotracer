//! Gaussian Splatting generation module
//!
//! Converts scene geometry directly to Gaussian splats with full SH coefficients.
//! This is a forward approach (geometry -> splats) rather than inverse (images -> splats).

pub mod ply;
pub mod sampler;
pub mod sh;

use glam::Vec3;

use crate::material::Material;
use crate::splat::sh::sh_dc_from_srgb;

/// Surface sample from geometry
#[derive(Debug, Clone, Copy)]
pub struct SurfaceSample {
    pub pos: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

/// Spherical harmonics coefficients up to degree 3
/// Total: 16 coefficients per channel, 48 for RGB
#[derive(Debug, Clone)]
pub struct ShCoeffs {
    /// Coefficients ordered by degree: [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, ...]
    /// Each entry is [R, G, B]
    pub coeffs: Vec<[f32; 3]>,
    pub degree: u32,
}

impl ShCoeffs {
    /// Create empty SH coefficients for given degree
    pub fn new(degree: u32) -> Self {
        let n_coeffs = ((degree + 1) * (degree + 1)) as usize;
        Self {
            coeffs: vec![[0.0; 3]; n_coeffs],
            degree,
        }
    }

    /// Get DC component (degree 0, index 0)
    pub fn dc(&self) -> [f32; 3] {
        self.coeffs[0]
    }

    /// Get rest coefficients (degrees 1-3, indices 1-15)
    /// Returns 45 floats in PLANAR format for bevy_gaussian_splatting:
    /// f_rest_0..14 = R coeffs, f_rest_15..29 = G coeffs, f_rest_30..44 = B coeffs
    pub fn rest_interleaved(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(45);

        // Planar format: all R coeffs (1-15), then all G, then all B
        for channel in 0..3 {
            for coeff_idx in 1..16 {
                if coeff_idx < self.coeffs.len() {
                    result.push(self.coeffs[coeff_idx][channel]);
                } else {
                    result.push(0.0);
                }
            }
        }

        result
    }
}

/// Single Gaussian splat ready for PLY export
#[derive(Debug, Clone)]
pub struct Gaussian {
    /// World position
    pub pos: Vec3,
    /// Surface normal
    pub normal: Vec3,
    /// SH degree 0 (base RGB color)
    pub sh_dc: [f32; 3],
    /// SH degrees 1-3 (45 floats: 15 coeffs * 3 channels)
    pub sh_rest: Vec<f32>,
    /// Opacity in logit-space: logit(p) = ln(p / (1-p))
    pub opacity: f32,
    /// Scale in log-space (ln of actual scale)
    pub scale: [f32; 3],
    /// Rotation quaternion (w, x, y, z)
    pub rotation: [f32; 4],
}

impl Gaussian {
    /// Create gaussian from surface sample and fitted SH coefficients
    pub fn from_sample(
        sample: &SurfaceSample,
        sh: &ShCoeffs,
        splat_scale: f32,
        opacity_prob: f32,
    ) -> Self {
        // SH coefficients in 3DGS feature space: color = 0.5 + sum_i(coeff_i * Y_i(view_dir)).
        // `fit_sh` already solves for these coefficients, so store them directly.
        let sh_dc = sh.dc();
        let sh_rest = sh.rest_interleaved();

        // Quaternion aligning local Z to surface normal
        let rotation = quat_from_normal(sample.normal);

        // Log-space scale: thin along the normal to behave like a surface (disc-like),
        // otherwise isotropic gaussians tend to look "foggy"/blurry.
        let tangent_sigma = splat_scale.max(1e-6);
        let normal_sigma = (splat_scale * 0.3).max(1e-6);
        let scale = [tangent_sigma.ln(), tangent_sigma.ln(), normal_sigma.ln()];

        // Logit-space opacity
        let opacity_p = opacity_prob.clamp(1e-4, 1.0 - 1e-4);
        let opacity = logit(opacity_p);

        Self {
            pos: sample.pos,
            normal: sample.normal,
            sh_dc,
            sh_rest,
            opacity,
            scale,
            rotation,
        }
    }

    /// Create gaussian from a constant sRGB color (view-independent).
    pub fn from_sample_constant(
        sample: &SurfaceSample,
        color_srgb: Vec3,
        splat_scale: f32,
        opacity_prob: f32,
    ) -> Self {
        let sh_dc = sh_dc_from_srgb(color_srgb);
        let sh_rest = vec![0.0; 45];

        let rotation = quat_from_normal(sample.normal);

        let tangent_sigma = splat_scale.max(1e-6);
        let normal_sigma = (splat_scale * 0.3).max(1e-6);
        let scale = [tangent_sigma.ln(), tangent_sigma.ln(), normal_sigma.ln()];

        let opacity_p = opacity_prob.clamp(1e-4, 1.0 - 1e-4);
        let opacity = logit(opacity_p);

        Self {
            pos: sample.pos,
            normal: sample.normal,
            sh_dc,
            sh_rest,
            opacity,
            scale,
            rotation,
        }
    }
}

/// Compute quaternion that rotates Z-axis to given normal
fn quat_from_normal(normal: Vec3) -> [f32; 4] {
    let z = Vec3::Z;
    let n = normal.normalize();

    // Handle case when normal is parallel to Z
    let dot = z.dot(n);
    if dot > 0.99999 {
        return [1.0, 0.0, 0.0, 0.0]; // Identity
    }
    if dot < -0.99999 {
        // 180 degree rotation around X axis
        return [0.0, 1.0, 0.0, 0.0];
    }

    // Cross product gives rotation axis
    let axis = z.cross(n).normalize();
    let angle = dot.acos();
    let half_angle = angle * 0.5;

    let s = half_angle.sin();
    let c = half_angle.cos();

    // Quaternion (w, x, y, z)
    [c, axis.x * s, axis.y * s, axis.z * s]
}

/// Logit function: ln(p / (1 - p))
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// Splat generation configuration
#[derive(Debug, Clone)]
pub struct SplatConfig {
    /// Samples per unit surface area
    pub density: f32,
    /// Number of directions to sample for SH fitting
    pub sh_samples: usize,
    /// Maximum ray depth for SH sampling
    pub max_depth: i32,
    /// Reflection depth limit
    pub reflection_depth: i32,
    /// Refraction depth limit
    pub refraction_depth: i32,
    /// Override splat scale (if None, auto-calculated from density)
    pub scale_override: Option<f32>,
    /// Apply tonemap before writing SH/DC
    pub tonemap: bool,
}

impl Default for SplatConfig {
    fn default() -> Self {
        Self {
            density: 100.0,
            sh_samples: 64,
            max_depth: 32,
            reflection_depth: 6,
            refraction_depth: 16,
            scale_override: None,
            tonemap: true,
        }
    }
}
