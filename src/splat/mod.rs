//! Gaussian Splatting generation module
//!
//! Converts scene geometry directly to Gaussian splats with full SH coefficients.
//! This is a forward approach (geometry -> splats) rather than inverse (images -> splats).

pub mod sh;
pub mod sampler;
pub mod ply;

use glam::Vec3;

/// Surface sample from geometry
#[derive(Debug, Clone, Copy)]
pub struct SurfaceSample {
    pub pos: Vec3,
    pub normal: Vec3,
    pub material_color: Vec3,
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
    /// Returns 45 floats: 15 coeffs * 3 channels, ordered by channel then coefficient
    pub fn rest_interleaved(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(45);
        
        // 3DGS format: all R coeffs (1-15), then all G, then all B
        for channel in 0..3 {
            for coeff_idx in 1..16.min(self.coeffs.len()) {
                result.push(self.coeffs[coeff_idx][channel]);
            }
            // Pad with zeros if degree < 3
            for _ in self.coeffs.len()..16 {
                result.push(0.0);
            }
        }
        
        result
    }
    
    /// Get normalized rest coefficients for 3DGS format
    /// Applies scaling to keep values in reasonable range
    pub fn rest_normalized(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(45);
        
        // Normalization: scale down higher order coeffs to prevent artifacts
        // 3DGS expects small values for f_rest (typically -1 to 1 range)
        let scale = 0.5;  // Conservative scaling
        
        // 3DGS format: all R coeffs (1-15), then all G, then all B
        for channel in 0..3 {
            for coeff_idx in 1..16 {
                if coeff_idx < self.coeffs.len() {
                    // Apply scaling to higher orders
                    result.push(self.coeffs[coeff_idx][channel] * scale);
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

// SH normalization constant for DC component
const SH_C0: f32 = 0.28209479177387814;  // Y_0^0 = 1/(2*sqrt(pi))

impl Gaussian {
    /// Create gaussian from surface sample and fitted SH coefficients
    pub fn from_sample(sample: &SurfaceSample, sh: &ShCoeffs, splat_scale: f32) -> Self {
        // Convert RGB radiance to SH DC format
        // In 3DGS: RGB = SH_DC * C0 + 0.5, so SH_DC = (RGB - 0.5) / C0
        let dc = sh.dc();
        let sh_dc = [
            (dc[0].clamp(0.0, 1.0) - 0.5) / SH_C0,
            (dc[1].clamp(0.0, 1.0) - 0.5) / SH_C0,
            (dc[2].clamp(0.0, 1.0) - 0.5) / SH_C0,
        ];
        
        // Normalize higher order SH coefficients
        // Scale factor to keep values in reasonable range
        let sh_rest = sh.rest_normalized();
        
        // Quaternion aligning local Z to surface normal
        let rotation = quat_from_normal(sample.normal);
        
        // Log-space scale (isotropic for now)
        let log_scale = splat_scale.ln();
        let scale = [log_scale, log_scale, log_scale];
        
        // Logit-space opacity (0.99 -> ~4.6)
        let opacity = logit(0.99);
        
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
    /// Maximum SH degree (0-3)
    pub sh_degree: u32,
    /// Maximum ray depth for SH sampling
    pub max_depth: i32,
    /// Reflection depth limit
    pub reflection_depth: i32,
    /// Refraction depth limit
    pub refraction_depth: i32,
    /// Override splat scale (if None, auto-calculated from density)
    pub scale_override: Option<f32>,
}

impl Default for SplatConfig {
    fn default() -> Self {
        Self {
            density: 100.0,
            sh_samples: 64,
            sh_degree: 3,
            max_depth: 32,
            reflection_depth: 6,
            refraction_depth: 16,
            scale_override: None,
        }
    }
}
