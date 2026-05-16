use glam::Vec3;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Write};
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

/// Read a 3DGS PLY file (binary little-endian) into a `Vec<Gaussian>`.
///
/// Tolerates ordering differences in the property list — the header is
/// parsed first to map each known property name to its index in the
/// binary record, then each vertex is read field-by-field according to
/// that map. Unknown properties are skipped (still consumed from the
/// stream so the cursor stays aligned).
///
/// Supports the Inria 3DGS schema (x/y/z, nx/ny/nz, f_dc_{0..2},
/// f_rest_{0..44}, opacity, scale_{0..2}, rot_{0..3}). Other PLY
/// variants will likely fail at the property-mapping step with a
/// clear error.
pub fn read_ply(path: &Path) -> Result<Vec<Gaussian>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader)?;
    if !header.binary_le {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "only binary_little_endian PLY is supported",
        ));
    }

    let mut gaussians = Vec::with_capacity(header.vertex_count);
    let mut record_buf = vec![0u8; header.properties.len() * 4];
    for _ in 0..header.vertex_count {
        reader.read_exact(&mut record_buf)?;
        gaussians.push(parse_record(&record_buf, &header)?);
    }
    Ok(gaussians)
}

#[derive(Debug)]
struct PlyHeader {
    binary_le: bool,
    vertex_count: usize,
    /// Each entry is the byte offset of that property within one
    /// record (all properties are `float` in our schema → 4 bytes each).
    properties: Vec<String>,
}

impl PlyHeader {
    fn offset(&self, name: &str) -> Option<usize> {
        self.properties.iter().position(|n| n == name).map(|i| i * 4)
    }
}

fn read_header(reader: &mut impl std::io::BufRead) -> Result<PlyHeader> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing 'ply' magic line",
        ));
    }
    let mut binary_le = false;
    let mut vertex_count: usize = 0;
    let mut properties: Vec<String> = Vec::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unterminated PLY header",
            ));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("comment ") {
            continue;
        }
        if trimmed == "end_header" {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("format ") {
            binary_le = rest.starts_with("binary_little_endian");
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("element vertex ") {
            vertex_count = rest.trim().parse::<usize>().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("element vertex: {e}"))
            })?;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("property float ") {
            properties.push(rest.trim().to_string());
            continue;
        }
        if trimmed.starts_with("element ") {
            // Other elements not supported in 3DGS PLY.
            continue;
        }
        if trimmed.starts_with("property ") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("only `property float ...` supported, got: {trimmed}"),
            ));
        }
    }
    Ok(PlyHeader {
        binary_le,
        vertex_count,
        properties,
    })
}

fn parse_record(record: &[u8], header: &PlyHeader) -> Result<Gaussian> {
    let get = |name: &str| -> Result<f32> {
        match header.offset(name) {
            Some(off) => Ok(f32::from_le_bytes(
                record[off..off + 4].try_into().expect("4-byte slice"),
            )),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("PLY missing required property: {name}"),
            )),
        }
    };

    let pos = Vec3::new(get("x")?, get("y")?, get("z")?);
    // Normals are nice-to-have; pre-trained PLYs sometimes omit them.
    let normal = match (header.offset("nx"), header.offset("ny"), header.offset("nz")) {
        (Some(_), Some(_), Some(_)) => Vec3::new(get("nx")?, get("ny")?, get("nz")?),
        _ => Vec3::Y,
    };
    let sh_dc = [get("f_dc_0")?, get("f_dc_1")?, get("f_dc_2")?];

    let mut sh_rest = Vec::with_capacity(45);
    for i in 0..45 {
        let name = format!("f_rest_{i}");
        sh_rest.push(header.offset(&name).map(|_| get(&name).unwrap_or(0.0)).unwrap_or(0.0));
    }

    let opacity = get("opacity")?;
    let scale = [get("scale_0")?, get("scale_1")?, get("scale_2")?];
    let rotation = [
        get("rot_0")?,
        get("rot_1")?,
        get("rot_2")?,
        get("rot_3")?,
    ];

    Ok(Gaussian {
        pos,
        normal,
        sh_dc,
        sh_rest,
        opacity,
        scale,
        rotation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrip() {
        let g = Gaussian {
            pos: Vec3::new(1.0, 2.0, 3.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            sh_dc: [0.4, 0.5, 0.6],
            sh_rest: (0..45).map(|i| i as f32 * 0.1).collect(),
            opacity: 2.7,
            scale: [-0.5, -0.6, -0.7],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let tmp = std::env::temp_dir().join("nano-splat-roundtrip.ply");
        write_ply(&tmp, std::slice::from_ref(&g)).unwrap();
        let back = read_ply(&tmp).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pos, g.pos);
        assert_eq!(back[0].sh_dc, g.sh_dc);
        assert_eq!(back[0].opacity, g.opacity);
        assert_eq!(back[0].scale, g.scale);
        assert_eq!(back[0].rotation, g.rotation);
        assert_eq!(back[0].sh_rest.len(), 45);
        for (i, v) in back[0].sh_rest.iter().enumerate() {
            assert!((*v - i as f32 * 0.1).abs() < 1e-5);
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
