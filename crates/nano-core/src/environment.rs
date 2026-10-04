//! HDR environment map for image-based lighting and reflections.
//!
//! CPU side holds raw pixel data (or a "procedural sky" flag) plus a
//! pre-convolved degree-2 SH for diffuse IBL. GPU sampling uses the
//! equirectangular path in the shaders; the SH irradiance is uploaded as
//! a uniform array and evaluated per surface normal at shade time.

use glam::Vec3;
use std::f32::consts::PI;

use crate::sh::sh_basis_all;

/// HDR environment map for realistic lighting.
pub struct EnvironmentMap {
    /// HDR pixel data in linear RGB. Empty when `use_sky` is true.
    data: Vec<Vec3>,
    width: u32,
    height: u32,
    /// Use the procedural gradient sky instead of the HDR data.
    use_sky: bool,
    /// Exposure multiplier applied on the GPU side.
    exposure: f32,
}

/// GPU-ready copy of the environment data.
///
/// `data` is uploaded as a 2D texture and `irradiance_sh` is uploaded as a
/// `vec4[9]` uniform array used by the IBL diffuse term in both shaders.
pub struct EnvGpuData {
    pub data: Vec<[f32; 4]>,
    pub width: u32,
    pub height: u32,
    pub exposure: f32,
    pub use_sky: bool,
    /// Lambertian-convolved env irradiance (degree-2 SH, 9 coefficients × RGB).
    /// xyz holds RGB, w unused — laid out for direct upload as `vec4[9]`.
    pub irradiance_sh: [[f32; 4]; 9],
}

impl EnvironmentMap {
    /// Load an EXR HDR environment map: the R, G, B channels of a single-part
    /// flat file through exr-core (our OpenEXR port), any channel type widened
    /// to f32. A file without all three is an error. Exposure is applied later
    /// on the GPU when sampling.
    pub fn from_exr(path: &str, exposure: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let image = exr_core::Image::read(path)?;
        let (width, height) = (u32::try_from(image.width())?, u32::try_from(image.height())?);
        let plane = |name: &str| -> Result<Vec<f32>, String> {
            if image.sampling(name).is_some_and(|s| s != (1, 1)) {
                return Err(format!("EXR {path}: channel {name} is subsampled"));
            }
            image
                .channel(name)
                .map(exr_core::ChannelData::to_f32)
                .ok_or_else(|| format!("EXR {path}: no {name} channel (an RGB image is required)"))
        };
        let (r, g, b) = (plane("R")?, plane("G")?, plane("B")?);
        let data: Vec<Vec3> = r.iter().zip(&g).zip(&b).map(|((r, g), b)| Vec3::new(*r, *g, *b)).collect();

        let max_luminance = data
            .iter()
            .map(|v| v.x.max(v.y).max(v.z))
            .fold(0.0f32, f32::max);
        let min_luminance = data
            .iter()
            .map(|v| v.x.min(v.y).min(v.z))
            .fold(f32::INFINITY, f32::min);
        println!("HDR stats: min={:.3}, max={:.3}", min_luminance, max_luminance);

        Ok(Self {
            data,
            width,
            height,
            use_sky: false,
            exposure,
        })
    }

    /// Procedural blue-to-warm gradient sky (no texture upload required).
    pub fn procedural_sky() -> Self {
        Self {
            data: vec![],
            width: 0,
            height: 0,
            use_sky: true,
            exposure: 1.0,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Sample radiance in direction `dir`. Mirrors the GPU shader's
    /// `sample_environment` so CPU pre-convolution stays consistent with
    /// runtime sampling. Returns linear RGB times `exposure`.
    pub fn sample_dir(&self, dir: Vec3) -> Vec3 {
        let n = dir.normalize_or_zero();
        if self.use_sky {
            let sky_blue = Vec3::new(0.5, 0.7, 1.0);
            let horizon = Vec3::new(1.0, 0.9, 0.7);
            let t = (n.y + 1.0) * 0.5;
            return sky_blue * t + horizon * (1.0 - t);
        }
        if self.width == 0 || self.height == 0 || self.data.is_empty() {
            return Vec3::ZERO;
        }
        let phi = n.z.atan2(n.x);
        let theta = (-n.y).acos();
        let u = (phi / (2.0 * PI) + 0.5).rem_euclid(1.0);
        let v = (theta / PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        let idx = (y * self.width + x) as usize;
        self.data.get(idx).copied().unwrap_or(Vec3::ZERO) * self.exposure
    }

    /// Lambertian-convolved env irradiance projected onto degree-2 SH.
    ///
    /// The result is the 9-coefficient SH such that
    /// `E(N) = Σ_lm A_l L_l^m Y_l^m(N)` reconstructs the cosine-integrated
    /// irradiance at any surface normal `N`, using the Ramamoorthi–Hanrahan
    /// band factors `A_0 = π`, `A_1 = 2π/3`, `A_2 = π/4`. Bands `l ≥ 3`
    /// vanish for Lambertian convolution, so degree 2 is sufficient.
    pub fn irradiance_sh(&self) -> [[f32; 4]; 9] {
        const N_SAMPLES: usize = 1024;
        let dirs = fibonacci_sphere(N_SAMPLES);
        let sphere_weight = 4.0 * PI / N_SAMPLES as f32;

        let mut sh = [[0.0f32; 3]; 9];
        for dir in &dirs {
            let radiance = self.sample_dir(*dir);
            let basis = sh_basis_all(2, *dir);
            for (i, &b) in basis.iter().enumerate().take(9) {
                sh[i][0] += radiance.x * b * sphere_weight;
                sh[i][1] += radiance.y * b * sphere_weight;
                sh[i][2] += radiance.z * b * sphere_weight;
            }
        }

        // Cosine convolution coefficients (Lambertian BRDF).
        let band = [
            PI,
            2.0 * PI / 3.0,
            2.0 * PI / 3.0,
            2.0 * PI / 3.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
            PI / 4.0,
        ];
        let mut out = [[0.0f32; 4]; 9];
        for i in 0..9 {
            out[i][0] = sh[i][0] * band[i];
            out[i][1] = sh[i][1] * band[i];
            out[i][2] = sh[i][2] * band[i];
            out[i][3] = 0.0;
        }
        out
    }

    /// Convert to the GPU-side layout. Always emits a 1×1 fallback texture
    /// when `use_sky` is true so the binding stays valid.
    pub fn gpu_data(&self) -> EnvGpuData {
        let (width, height) = if self.use_sky {
            (1, 1)
        } else {
            (self.width.max(1), self.height.max(1))
        };

        let data = if self.use_sky {
            vec![[0.0, 0.0, 0.0, 1.0]]
        } else {
            self.data.iter().map(|v| [v.x, v.y, v.z, 1.0]).collect()
        };

        EnvGpuData {
            data,
            width,
            height,
            exposure: self.exposure,
            use_sky: self.use_sky,
            irradiance_sh: self.irradiance_sh(),
        }
    }
}

/// Quasi-uniform Fibonacci-spiral directions on the full unit sphere.
/// Used by [`EnvironmentMap::irradiance_sh`] for SH projection.
fn fibonacci_sphere(n: usize) -> Vec<Vec3> {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden_ratio;
    (0..n)
        .map(|i| {
            let theta = golden_angle * i as f32;
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_sky_irradiance_is_finite_and_nonzero() {
        let env = EnvironmentMap::procedural_sky();
        let sh = env.irradiance_sh();
        // DC term (Y_0^0) should be roughly the average sky colour times π.
        // For the sky_blue × horizon mix, that's ~ (0.75, 0.8, 0.85) on the
        // hemisphere; sphere-averaged the DC contribution is moderately
        // bright. Just check it's finite, positive, and not nuked.
        for (ch, &v) in sh[0].iter().enumerate().take(3) {
            assert!(v.is_finite(), "sh[0][{ch}] non-finite");
            assert!(v > 0.0, "sh[0][{ch}] = {v} not positive");
        }
    }

    #[test]
    fn fibonacci_sphere_is_unit_length_and_covers_full_sphere() {
        let dirs = fibonacci_sphere(128);
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for d in &dirs {
            let len = d.length();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "non-unit direction len {len}: {d:?}"
            );
            min_z = min_z.min(d.z);
            max_z = max_z.max(d.z);
        }
        assert!(min_z < -0.9, "min z = {min_z} (lower hemisphere missing)");
        assert!(max_z > 0.9, "max z = {max_z} (upper hemisphere missing)");
    }

    #[test]
    fn exr_environment_keeps_hdr_radiance_through_exr_core() {
        use exr_core::attr::Compression;
        use exr_core::{ChannelData, Image};
        use imath_rs::{Box2i, V2i};

        let path = std::env::temp_dir().join(format!("nano-env-{}.exr", std::process::id()));
        let window = Box2i { min: V2i { x: 0, y: 0 }, max: V2i { x: 1, y: 0 } };
        Image::new(window)
            .with_channel("R", ChannelData::Float(vec![19.43, 0.0]))
            .with_channel("G", ChannelData::Float(vec![2.0, 0.5]))
            .with_channel("B", ChannelData::Float(vec![0.125, 1e-6]))
            .write(&path, Compression::Zip)
            .unwrap();
        let env = EnvironmentMap::from_exr(path.to_str().unwrap(), 1.0).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!((env.width(), env.height()), (2, 1));
        assert_eq!(env.data, vec![Vec3::new(19.43, 2.0, 0.125), Vec3::new(0.0, 0.5, 1e-6)]);
    }
}
