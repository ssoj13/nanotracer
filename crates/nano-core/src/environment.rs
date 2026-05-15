//! HDR environment map for image-based lighting and reflections.
//!
//! The CPU side just holds raw pixel data and metadata; sampling happens on
//! the GPU via the equirectangular path inside the shaders. The `sample()`
//! / `tone_map()` / `direction_to_uv()` helpers that used to live here were
//! only consumed by the now-removed CPU intersection path and have been
//! dropped — see `plan1.md §3.1`.

use glam::Vec3;

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

/// GPU-ready copy of the environment data, ready to upload as a 2D texture.
pub struct EnvGpuData {
    pub data: Vec<[f32; 4]>,
    pub width: u32,
    pub height: u32,
    pub exposure: f32,
    pub use_sky: bool,
}

impl EnvironmentMap {
    /// Load an EXR HDR environment map. Loads the first RGBA layer; exposure
    /// is applied later on the GPU when sampling.
    pub fn from_exr(path: &str, exposure: f32) -> Result<Self, Box<dyn std::error::Error>> {
        use exr::prelude::*;

        let image = read_first_rgba_layer_from_file(
            path,
            |resolution, _| {
                let width = resolution.width() as u32;
                let height = resolution.height() as u32;
                vec![Vec3::ZERO; (width * height) as usize]
            },
            |pixel_vector, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                let width = position.width();
                let index = position.y() * width + position.x();
                if index < pixel_vector.len() {
                    pixel_vector[index] = Vec3::new(r, g, b);
                }
            },
        )?;

        let width = image.layer_data.size.width() as u32;
        let height = image.layer_data.size.height() as u32;
        let data = image.layer_data.channel_data.pixels;

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

    /// Convert to the GPU-side layout (always at least 1×1 so the texture
    /// binding stays valid even in procedural-sky mode).
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
        }
    }
}
