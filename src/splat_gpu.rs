use glam::Vec3;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Gaussian {
    pub pos: Vec3,
    pub normal: Vec3,
    pub sh_dc: [f32; 3],
    pub sh_rest: Vec<f32>,
    pub opacity: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
}

pub fn write_ply(path: &Path, gaussians: &[Gaussian]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write_header(&mut writer, gaussians.len())?;
    for g in gaussians {
        write_gaussian(&mut writer, g)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_header(w: &mut impl Write, vertex_count: usize) -> Result<()> {
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {}", vertex_count)?;

    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;

    writeln!(w, "property float nx")?;
    writeln!(w, "property float ny")?;
    writeln!(w, "property float nz")?;

    writeln!(w, "property float f_dc_0")?;
    writeln!(w, "property float f_dc_1")?;
    writeln!(w, "property float f_dc_2")?;

    for i in 0..45 {
        writeln!(w, "property float f_rest_{}", i)?;
    }

    writeln!(w, "property float opacity")?;

    writeln!(w, "property float scale_0")?;
    writeln!(w, "property float scale_1")?;
    writeln!(w, "property float scale_2")?;

    writeln!(w, "property float rot_0")?;
    writeln!(w, "property float rot_1")?;
    writeln!(w, "property float rot_2")?;
    writeln!(w, "property float rot_3")?;

    writeln!(w, "end_header")?;

    Ok(())
}

fn write_gaussian(w: &mut impl Write, g: &Gaussian) -> Result<()> {
    write_f32(w, g.pos.x)?;
    write_f32(w, g.pos.y)?;
    write_f32(w, g.pos.z)?;

    write_f32(w, g.normal.x)?;
    write_f32(w, g.normal.y)?;
    write_f32(w, g.normal.z)?;

    write_f32(w, g.sh_dc[0])?;
    write_f32(w, g.sh_dc[1])?;
    write_f32(w, g.sh_dc[2])?;

    for i in 0..45 {
        let val = g.sh_rest.get(i).copied().unwrap_or(0.0);
        write_f32(w, val)?;
    }

    write_f32(w, g.opacity)?;

    write_f32(w, g.scale[0])?;
    write_f32(w, g.scale[1])?;
    write_f32(w, g.scale[2])?;

    write_f32(w, g.rotation[0])?;
    write_f32(w, g.rotation[1])?;
    write_f32(w, g.rotation[2])?;
    write_f32(w, g.rotation[3])?;

    Ok(())
}

#[inline]
fn write_f32(w: &mut impl Write, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}
