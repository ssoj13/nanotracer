//! Triangle mesh data and primitive generators.
//!
//! Pre-GPU-refactor versions of this module shipped a CPU BVH (via `rtbvh`)
//! and `Mesh::intersect` helpers. After the refactor, intersection happens
//! entirely on the GPU through Vulkan ray queries, so the CPU side is now
//! just a data container.

use glam::Vec3;
use std::f32::consts::PI;

/// Triangle mesh — pure data container fed straight into `nano-gpu::gpu_scene`.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    /// Per-vertex normals; length matches `vertices` when present.
    pub normals: Vec<Vec3>,
}

impl Mesh {
    /// Create a mesh and auto-generate smooth normals from area-weighted face normals.
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        let normals = Self::compute_smooth_normals(&vertices, &indices);
        Self { vertices, indices, normals }
    }

    /// Create a mesh with explicit per-vertex normals (length must match
    /// `vertices`, otherwise GPU-side code falls back to face normals).
    pub fn with_normals(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>, normals: Vec<Vec3>) -> Self {
        Self { vertices, indices, normals }
    }

    /// Average area-weighted face normals at each vertex.
    fn compute_smooth_normals(vertices: &[Vec3], indices: &[[u32; 3]]) -> Vec<Vec3> {
        let mut normals = vec![Vec3::ZERO; vertices.len()];
        for tri in indices {
            let v0 = vertices[tri[0] as usize];
            let v1 = vertices[tri[1] as usize];
            let v2 = vertices[tri[2] as usize];
            let face_normal = (v1 - v0).cross(v2 - v0); // unnormalised — area-weighted
            normals[tri[0] as usize] += face_normal;
            normals[tri[1] as usize] += face_normal;
            normals[tri[2] as usize] += face_normal;
        }
        normals.iter().map(|n| n.normalize_or_zero()).collect()
    }
}

// ── Primitive generators ────────────────────────────────────────────────────

/// Cube of edge length `size`, centred at origin.
pub fn cube(size: f32) -> Mesh {
    let h = size * 0.5;
    let v = vec![
        Vec3::new(-h, -h, -h),
        Vec3::new( h, -h, -h),
        Vec3::new( h,  h, -h),
        Vec3::new(-h,  h, -h),
        Vec3::new(-h, -h,  h),
        Vec3::new( h, -h,  h),
        Vec3::new( h,  h,  h),
        Vec3::new(-h,  h,  h),
    ];
    let i = vec![
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 6, 2], [3, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 4, 7], [0, 7, 3],
    ];
    Mesh::new(v, i)
}

/// Square-base pyramid with side `base` and `height`; base sits at `y=0`.
pub fn pyramid(base: f32, height: f32) -> Mesh {
    let h = base * 0.5;
    let v = vec![
        Vec3::new(-h, 0.0, -h),
        Vec3::new( h, 0.0, -h),
        Vec3::new( h, 0.0,  h),
        Vec3::new(-h, 0.0,  h),
        Vec3::new(0.0, height, 0.0),
    ];
    let i = vec![
        [0, 2, 1], [0, 3, 2],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    Mesh::new(v, i)
}

/// Torus: major radius `r1`, minor radius `r2`, `seg` ring segments,
/// `sides` segments around the tube cross-section.
pub fn torus(r1: f32, r2: f32, seg: u32, sides: u32) -> Mesh {
    let mut vertices = Vec::with_capacity((seg * sides) as usize);
    let mut normals = Vec::with_capacity((seg * sides) as usize);
    for i in 0..seg {
        let u = (i as f32) / (seg as f32) * 2.0 * PI;
        let (su, cu) = u.sin_cos();
        for j in 0..sides {
            let v = (j as f32) / (sides as f32) * 2.0 * PI;
            let (sv, cv) = v.sin_cos();
            let x = (r1 + r2 * cv) * cu;
            let y = r2 * sv;
            let z = (r1 + r2 * cv) * su;
            vertices.push(Vec3::new(x, y, z));
            normals.push(Vec3::new(cv * cu, sv, cv * su));
        }
    }
    let mut indices = Vec::with_capacity((seg * sides * 2) as usize);
    for i in 0..seg {
        for j in 0..sides {
            let i1 = (i + 1) % seg;
            let j1 = (j + 1) % sides;
            let a = i * sides + j;
            let b = i1 * sides + j;
            let c = i1 * sides + j1;
            let d = i * sides + j1;
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }
    Mesh::with_normals(vertices, indices, normals)
}
