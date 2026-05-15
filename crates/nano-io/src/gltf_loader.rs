//! glTF/GLB mesh loader for static triangle geometry.

use std::path::Path;

use glam::{Mat3, Mat4, Vec3};

use nano_core::mesh::Mesh;

pub fn load_glb_mesh(path: &Path, scale: f32) -> Result<Mesh, Box<dyn std::error::Error>> {
    let (doc, buffers, _images) = gltf::import(path)?;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut all_normals = true;

    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let normal_mat = Mat3::from_mat4(transform).inverse().transpose();

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions = match reader.read_positions() {
                Some(p) => p.collect::<Vec<[f32; 3]>>(),
                None => continue,
            };
            let normal_data = reader.read_normals().map(|n| n.collect::<Vec<[f32; 3]>>());
            if normal_data.is_none() {
                all_normals = false;
            }

            let base_index = vertices.len() as u32;
            for (i, pos) in positions.iter().enumerate() {
                let p = Vec3::new(pos[0], pos[1], pos[2]) * scale;
                let p = transform.transform_point3(p);
                vertices.push(p);

                if let Some(nlist) = &normal_data {
                    let n = Vec3::new(nlist[i][0], nlist[i][1], nlist[i][2]);
                    normals.push((normal_mat * n).normalize_or_zero());
                } else {
                    normals.push(Vec3::ZERO);
                }
            }

            if let Some(idx_iter) = reader.read_indices() {
                let mut idx = Vec::new();
                idx.extend(idx_iter.into_u32());
                for tri in idx.chunks(3) {
                    if tri.len() == 3 {
                        indices.push([
                            base_index + tri[0],
                            base_index + tri[1],
                            base_index + tri[2],
                        ]);
                    }
                }
            } else {
                for i in (0..positions.len()).step_by(3) {
                    if i + 2 < positions.len() {
                        indices.push([
                            base_index + i as u32,
                            base_index + (i + 1) as u32,
                            base_index + (i + 2) as u32,
                        ]);
                    }
                }
            }
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err("GLB contains no triangle primitives".into());
    }

    if all_normals {
        Ok(Mesh::with_normals(vertices, indices, normals))
    } else {
        Ok(Mesh::new(vertices, indices))
    }
}
