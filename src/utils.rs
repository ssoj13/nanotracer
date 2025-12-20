//! Utility functions for the raytracer

use crate::color::apply_tonemap_srgb;
use crate::vec3::Vector3;
use image::{ImageBuffer, Rgb};

/// Convert a Vector3 color to a clamped RGB pixel value
pub fn vec3_to_rgb(color: Vector3, tonemap: bool) -> Rgb<u8> {
    let srgb = apply_tonemap_srgb(color, tonemap);
    Rgb([
        (255.0 * srgb.x.clamp(0.0, 1.0)) as u8,
        (255.0 * srgb.y.clamp(0.0, 1.0)) as u8,
        (255.0 * srgb.z.clamp(0.0, 1.0)) as u8,
    ])
}

/// Save framebuffer as PNG image
pub fn save_image(
    framebuffer: &[Vector3],
    width: u32,
    height: u32,
    filename: &str,
    tonemap: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = ImageBuffer::new(width, height);

    for (i, color) in framebuffer.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, vec3_to_rgb(*color, tonemap));
    }

    img.save(filename)?;
    Ok(())
}
