//! Utility functions for the raytracer

use crate::vec3::Vector3;
use image::{ImageBuffer, Rgb};

/// Convert a Vector3 color to a clamped RGB pixel value
pub fn vec3_to_rgb(color: Vector3) -> Rgb<u8> {
    // Clamp the color values and normalize if necessary
    let max_value = 1.0_f32.max(color.x.max(color.y.max(color.z)));
    
    Rgb([
        (255.0 * (color.x / max_value)) as u8,
        (255.0 * (color.y / max_value)) as u8,
        (255.0 * (color.z / max_value)) as u8,
    ])
}

/// Save framebuffer as PNG image
pub fn save_image(
    framebuffer: &[Vector3],
    width: u32,
    height: u32,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = ImageBuffer::new(width, height);
    
    for (i, color) in framebuffer.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, vec3_to_rgb(*color));
    }
    
    img.save(filename)?;
    Ok(())
}