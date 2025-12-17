//! PLY file writer for Gaussian splats
//!
//! Writes binary little-endian PLY files compatible with standard 3DGS viewers:
//! - SuperSplat
//! - Luma Labs viewer
//! - bevy_gaussian_splatting
//! - web-splat

use std::fs::File;
use std::io::{BufWriter, Write, Result};
use std::path::Path;

use super::Gaussian;

/// Write gaussians to binary PLY file
/// 
/// Format follows the original 3DGS implementation:
/// - Position: x, y, z (float32)
/// - Normal: nx, ny, nz (float32)
/// - SH DC: f_dc_0, f_dc_1, f_dc_2 (float32)
/// - SH rest: f_rest_0 .. f_rest_44 (float32) - 15 coeffs * 3 channels
/// - Opacity: opacity (float32, logit-space)
/// - Scale: scale_0, scale_1, scale_2 (float32, log-space)
/// - Rotation: rot_0, rot_1, rot_2, rot_3 (float32, quaternion wxyz)
pub fn write_ply(path: &Path, gaussians: &[Gaussian]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write ASCII header
    write_header(&mut writer, gaussians.len())?;
    
    // Write binary data
    for g in gaussians {
        write_gaussian(&mut writer, g)?;
    }
    
    writer.flush()?;
    Ok(())
}

/// Write PLY header
fn write_header(w: &mut impl Write, vertex_count: usize) -> Result<()> {
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {}", vertex_count)?;
    
    // Position
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    
    // Normal
    writeln!(w, "property float nx")?;
    writeln!(w, "property float ny")?;
    writeln!(w, "property float nz")?;
    
    // SH DC (degree 0)
    writeln!(w, "property float f_dc_0")?;
    writeln!(w, "property float f_dc_1")?;
    writeln!(w, "property float f_dc_2")?;
    
    // SH rest (degrees 1-3: 45 coefficients)
    for i in 0..45 {
        writeln!(w, "property float f_rest_{}", i)?;
    }
    
    // Opacity (logit-space)
    writeln!(w, "property float opacity")?;
    
    // Scale (log-space)
    writeln!(w, "property float scale_0")?;
    writeln!(w, "property float scale_1")?;
    writeln!(w, "property float scale_2")?;
    
    // Rotation (quaternion w, x, y, z)
    writeln!(w, "property float rot_0")?;
    writeln!(w, "property float rot_1")?;
    writeln!(w, "property float rot_2")?;
    writeln!(w, "property float rot_3")?;
    
    writeln!(w, "end_header")?;
    
    Ok(())
}

/// Write single gaussian in binary format
fn write_gaussian(w: &mut impl Write, g: &Gaussian) -> Result<()> {
    // Position
    write_f32(w, g.pos.x)?;
    write_f32(w, g.pos.y)?;
    write_f32(w, g.pos.z)?;
    
    // Normal
    write_f32(w, g.normal.x)?;
    write_f32(w, g.normal.y)?;
    write_f32(w, g.normal.z)?;
    
    // SH DC
    write_f32(w, g.sh_dc[0])?;
    write_f32(w, g.sh_dc[1])?;
    write_f32(w, g.sh_dc[2])?;
    
    // SH rest (45 coefficients)
    for i in 0..45 {
        let val = g.sh_rest.get(i).copied().unwrap_or(0.0);
        write_f32(w, val)?;
    }
    
    // Opacity
    write_f32(w, g.opacity)?;
    
    // Scale
    write_f32(w, g.scale[0])?;
    write_f32(w, g.scale[1])?;
    write_f32(w, g.scale[2])?;
    
    // Rotation
    write_f32(w, g.rotation[0])?;
    write_f32(w, g.rotation[1])?;
    write_f32(w, g.rotation[2])?;
    write_f32(w, g.rotation[3])?;
    
    Ok(())
}

/// Write f32 in little-endian format
#[inline]
fn write_f32(w: &mut impl Write, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Calculate total file size for given number of gaussians
pub fn estimate_file_size(n_gaussians: usize) -> usize {
    // Header is approximately 1KB
    let header_size = 1024;
    
    // Each gaussian: 3 + 3 + 3 + 45 + 1 + 3 + 4 = 62 floats = 248 bytes
    let gaussian_size = 62 * 4;
    
    header_size + n_gaussians * gaussian_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    
    #[test]
    fn test_write_simple_ply() {
        let gaussians = vec![
            Gaussian {
                pos: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                sh_dc: [0.5, 0.5, 0.5],
                sh_rest: vec![0.0; 45],
                opacity: 4.6, // logit(0.99)
                scale: [-2.0, -2.0, -2.0], // ln(0.135)
                rotation: [1.0, 0.0, 0.0, 0.0],
            }
        ];
        
        let path = std::env::temp_dir().join("test_splat.ply");
        write_ply(&path, &gaussians).unwrap();
        
        // Verify file was created
        assert!(path.exists());
        
        // Clean up
        std::fs::remove_file(path).ok();
    }
}
