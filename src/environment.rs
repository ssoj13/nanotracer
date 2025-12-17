//! HDR environment map support for realistic lighting and reflections

use crate::vec3::Vector3;
use std::f32::consts::PI;

/// HDR environment map for realistic lighting
pub struct EnvironmentMap {
    /// HDR pixel data in linear RGB
    data: Vec<Vector3>,
    /// Width of the environment map
    width: u32,
    /// Height of the environment map
    height: u32,
    /// Use procedural sky instead of HDR data
    use_sky: bool,
    /// Exposure adjustment for HDR tone mapping
    exposure: f32,
}

impl EnvironmentMap {
    /// Load HDR environment map from EXR file
    pub fn from_exr(path: &str, exposure: f32) -> Result<Self, Box<dyn std::error::Error>> {
        use exr::prelude::*;

        // Read the EXR file
        let image = read_first_rgba_layer_from_file(
            path,
            |resolution, _| {
                let width = resolution.width() as u32;
                let height = resolution.height() as u32;
                vec![Vector3::ZERO; (width * height) as usize]
            },
            |pixel_vector, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                let width = position.width();
                let index = position.y() * width + position.x();
                if index < pixel_vector.len() {
                    pixel_vector[index] = Vector3::new(r, g, b);
                }
            },
        )?;

        let width = image.layer_data.size.width() as u32;
        let height = image.layer_data.size.height() as u32;
        let data = image.layer_data.channel_data.pixels;

        // Debug: Print some statistics about the HDR data
        let max_luminance = data
            .iter()
            .map(|v| v.x.max(v.y).max(v.z))
            .fold(0.0f32, f32::max);
        let min_luminance = data
            .iter()
            .map(|v| v.x.min(v.y).min(v.z))
            .fold(f32::INFINITY, f32::min);
        println!(
            "HDR stats: min={:.3}, max={:.3}",
            min_luminance, max_luminance
        );

        Ok(Self {
            data,
            width,
            height,
            use_sky: false,
            exposure,
        })
    }

    /// Create a procedural sky environment
    pub fn procedural_sky() -> Self {
        Self {
            data: vec![],
            width: 0,
            height: 0,
            use_sky: true,
            exposure: 1.0, // Not used for procedural sky
        }
    }

    /// Convert 3D direction vector to UV coordinates (equirectangular projection)
    fn direction_to_uv(&self, dir: Vector3) -> (f32, f32) {
        // Normalize direction vector
        let dir = dir.normalize();

        // Convert to spherical coordinates
        let phi = dir.z.atan2(dir.x); // Azimuth angle [-π, π]
        let theta = (-dir.y).acos(); // Polar angle [0, π] (from +Y down)

        // Convert to UV coordinates [0,1]
        let u = (phi / (2.0 * PI) + 0.5) % 1.0;
        let v = theta / PI;

        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    }

    /// Sample environment map with bilinear filtering and tone mapping
    pub fn sample(&self, direction: Vector3) -> Vector3 {
        // Use procedural sky if enabled
        if self.use_sky {
            let dir = direction.normalize();
            let sky_blue = Vector3::new(0.5, 0.7, 1.0);
            let horizon = Vector3::new(1.0, 0.9, 0.7);
            let t = (dir.y + 1.0) * 0.5; // Blend from horizon to sky
            return sky_blue * t + horizon * (1.0 - t);
        }

        let (u, v) = self.direction_to_uv(direction);

        // Simple nearest neighbor sampling to avoid interpolation issues
        let x = (u * self.width as f32) as u32;
        let y = (v * self.height as f32) as u32;
        let hdr_color = self.get_pixel(x, y);

        // Simple tone mapping to avoid blown out highlights
        self.tone_map(hdr_color)
    }

    /// Simple tone mapping using Reinhard operator with exposure adjustment
    fn tone_map(&self, hdr_color: Vector3) -> Vector3 {
        // Apply user-specified exposure adjustment
        let exposed = hdr_color * self.exposure;

        // Reinhard tone mapping: color / (1 + color)
        Vector3::new(
            exposed.x / (1.0 + exposed.x),
            exposed.y / (1.0 + exposed.y),
            exposed.z / (1.0 + exposed.z),
        )
    }

    /// Get pixel at coordinates with bounds checking
    fn get_pixel(&self, x: u32, y: u32) -> Vector3 {
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        let index = (y * self.width + x) as usize;
        self.data.get(index).copied().unwrap_or(Vector3::ZERO)
    }

    /// Get width of environment map
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get height of environment map
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Default sky color fallback
pub const DEFAULT_SKY_COLOR: Vector3 = Vector3::new(0.2, 0.7, 0.8);
