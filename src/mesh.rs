//! Mesh geometry with light-weight BVH acceleration using `rtbvh`.
//!
//! Supports triangle meshes with per-vertex normals, precomputed primitives, and
//! fast BVH traversal for ray intersections.

use glam::Vec3;
use rtbvh::{Aabb, Builder, Bvh, Primitive, Ray};
use std::f32::consts::PI;
use std::num::NonZeroUsize;

/// Triangle mesh with BVH acceleration
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    pub normals: Vec<Vec3>, // per-vertex normals
    bvh: Option<Bvh>,
    primitives: Vec<TrianglePrimitive>,
}

/// Triangle primitive wired into `rtbvh`
#[derive(Debug, Copy, Clone)]
struct TrianglePrimitive {
    tri_index: u32,
    center: Vec3,
    aabb: Aabb,
}

impl TrianglePrimitive {
    fn new(tri_index: u32, vertices: &[Vec3], tri: [u32; 3]) -> Self {
        let v0 = vertices[tri[0] as usize];
        let v1 = vertices[tri[1] as usize];
        let v2 = vertices[tri[2] as usize];
        let center = (v0 + v1 + v2) * (1.0 / 3.0);
        let aabb = Aabb::from_points(&[v0, v1, v2]);
        Self {
            tri_index,
            center,
            aabb,
        }
    }
}

impl Primitive<i32> for TrianglePrimitive {
    fn center(&self) -> Vec3 {
        self.center
    }

    fn aabb(&self) -> Aabb<i32> {
        self.aabb
    }
}

/// Result of ray-triangle intersection
#[derive(Debug, Clone, Copy)]
pub struct TriangleHit {
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub tri_idx: u32,
}

impl Mesh {
    /// Create mesh from vertices and indices, auto-generate normals
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        let normals = Self::compute_smooth_normals(&vertices, &indices);
        let mut mesh = Self {
            vertices,
            indices,
            normals,
            bvh: None,
            primitives: Vec::new(),
        };
        mesh.build_bvh();
        mesh
    }

    /// Create mesh with explicit normals
    pub fn with_normals(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>, normals: Vec<Vec3>) -> Self {
        let mut mesh = Self {
            vertices,
            indices,
            normals,
            bvh: None,
            primitives: Vec::new(),
        };
        mesh.build_bvh();
        mesh
    }

    /// Compute smooth normals by averaging face normals at each vertex
    fn compute_smooth_normals(vertices: &[Vec3], indices: &[[u32; 3]]) -> Vec<Vec3> {
        let mut normals = vec![Vec3::ZERO; vertices.len()];

        for tri in indices {
            let v0 = vertices[tri[0] as usize];
            let v1 = vertices[tri[1] as usize];
            let v2 = vertices[tri[2] as usize];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let face_normal = edge1.cross(edge2); // not normalized, weighted by area

            normals[tri[0] as usize] += face_normal;
            normals[tri[1] as usize] += face_normal;
            normals[tri[2] as usize] += face_normal;
        }

        normals.iter().map(|n| n.normalize_or_zero()).collect()
    }

    /// Build BVH for fast intersection
    fn build_bvh(&mut self) {
        if self.indices.is_empty() {
            self.primitives.clear();
            self.bvh = None;
            return;
        }

        let primitives: Vec<TrianglePrimitive> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, tri)| TrianglePrimitive::new(i as u32, &self.vertices, *tri))
            .collect();

        let builder = Builder {
            aabbs: None,
            primitives: primitives.as_slice(),
            primitives_per_leaf: NonZeroUsize::new(4),
        };

        let bvh = builder.construct_binned_sah().ok();
        self.primitives = primitives;
        self.bvh = bvh;
    }

    /// Ray-mesh intersection using BVH
    pub fn intersect(&self, origin: Vec3, dir: Vec3) -> Option<TriangleHit> {
        let bvh = self.bvh.as_ref()?;
        if self.primitives.is_empty() {
            return None;
        }

        let mut ray = Ray::from((origin, dir));
        let mut best_hit: Option<TriangleHit> = None;
        let mut best_t = f32::INFINITY;

        let iter = bvh.traverse_iter(&mut ray, self.primitives.as_slice());
        for (prim, bvh_ray) in iter {
            if let Some(hit) = self.intersect_triangle(origin, dir, prim.tri_index)
                && hit.t < best_t {
                    best_t = hit.t;
                    best_hit = Some(hit);
                    bvh_ray.t = best_t;
                }
        }

        best_hit
    }

    /// Moller-Trumbore ray-triangle intersection
    fn intersect_triangle(&self, origin: Vec3, dir: Vec3, tri_idx: u32) -> Option<TriangleHit> {
        let tri = &self.indices[tri_idx as usize];
        let v0 = self.vertices[tri[0] as usize];
        let v1 = self.vertices[tri[1] as usize];
        let v2 = self.vertices[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let h = dir.cross(edge2);
        let a = edge1.dot(h);

        // Parallel ray
        if a.abs() < 1e-8 {
            return None;
        }

        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * dir.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);

        if t > 0.001 {
            Some(TriangleHit { t, u, v, tri_idx })
        } else {
            None
        }
    }

    /// Get interpolated normal at hit point
    pub fn normal_at(&self, hit: &TriangleHit) -> Vec3 {
        let tri = &self.indices[hit.tri_idx as usize];
        let n0 = self.normals[tri[0] as usize];
        let n1 = self.normals[tri[1] as usize];
        let n2 = self.normals[tri[2] as usize];

        let w = 1.0 - hit.u - hit.v;
        (n0 * w + n1 * hit.u + n2 * hit.v).normalize()
    }

    /// Axis-aligned bounding box of the mesh
    pub fn bounding_box(&self) -> Aabb {
        if self.vertices.is_empty() {
            return Aabb::empty();
        }
        let mut bounds = Aabb::empty();
        for &vertex in &self.vertices {
            bounds.grow(vertex);
        }
        bounds
    }

    /// Total surface area
    pub fn surface_area(&self) -> f32 {
        self.indices
            .iter()
            .map(|tri| {
                let v0 = self.vertices[tri[0] as usize];
                let v1 = self.vertices[tri[1] as usize];
                let v2 = self.vertices[tri[2] as usize];
                (v1 - v0).cross(v2 - v0).length() * 0.5
            })
            .sum()
    }

    /// Triangle count
    pub fn tri_count(&self) -> usize {
        self.indices.len()
    }
}

// ============================================================================
// Procedural primitives
// ============================================================================

/// Create a unit cube centered at origin
pub fn cube(size: f32) -> Mesh {
    let h = size * 0.5;

    #[rustfmt::skip]
    let vertices = vec![
        // Front face
        Vec3::new(-h, -h,  h), Vec3::new( h, -h,  h), Vec3::new( h,  h,  h), Vec3::new(-h,  h,  h),
        // Back face
        Vec3::new(-h, -h, -h), Vec3::new(-h,  h, -h), Vec3::new( h,  h, -h), Vec3::new( h, -h, -h),
        // Top face
        Vec3::new(-h,  h, -h), Vec3::new(-h,  h,  h), Vec3::new( h,  h,  h), Vec3::new( h,  h, -h),
        // Bottom face
        Vec3::new(-h, -h, -h), Vec3::new( h, -h, -h), Vec3::new( h, -h,  h), Vec3::new(-h, -h,  h),
        // Right face
        Vec3::new( h, -h, -h), Vec3::new( h,  h, -h), Vec3::new( h,  h,  h), Vec3::new( h, -h,  h),
        // Left face
        Vec3::new(-h, -h, -h), Vec3::new(-h, -h,  h), Vec3::new(-h,  h,  h), Vec3::new(-h,  h, -h),
    ];

    #[rustfmt::skip]
    let normals = vec![
        // Front
        Vec3::Z, Vec3::Z, Vec3::Z, Vec3::Z,
        // Back
        Vec3::NEG_Z, Vec3::NEG_Z, Vec3::NEG_Z, Vec3::NEG_Z,
        // Top
        Vec3::Y, Vec3::Y, Vec3::Y, Vec3::Y,
        // Bottom
        Vec3::NEG_Y, Vec3::NEG_Y, Vec3::NEG_Y, Vec3::NEG_Y,
        // Right
        Vec3::X, Vec3::X, Vec3::X, Vec3::X,
        // Left
        Vec3::NEG_X, Vec3::NEG_X, Vec3::NEG_X, Vec3::NEG_X,
    ];

    #[rustfmt::skip]
    let indices = vec![
        [0, 1, 2], [0, 2, 3],       // Front
        [4, 5, 6], [4, 6, 7],       // Back
        [8, 9, 10], [8, 10, 11],    // Top
        [12, 13, 14], [12, 14, 15], // Bottom
        [16, 17, 18], [16, 18, 19], // Right
        [20, 21, 22], [20, 22, 23], // Left
    ];

    Mesh::with_normals(vertices, indices, normals)
}

/// Create a pyramid with square base
pub fn pyramid(base: f32, height: f32) -> Mesh {
    let h = base * 0.5;
    let apex = Vec3::new(0.0, height, 0.0);

    let v0 = Vec3::new(-h, 0.0, -h); // back-left
    let v1 = Vec3::new(h, 0.0, -h); // back-right
    let v2 = Vec3::new(h, 0.0, h); // front-right
    let v3 = Vec3::new(-h, 0.0, h); // front-left

    // Compute face normals (outward facing)
    let n_front = (v3 - apex).cross(v2 - apex).normalize();
    let n_right = (v2 - apex).cross(v1 - apex).normalize();
    let n_back = (v1 - apex).cross(v0 - apex).normalize();
    let n_left = (v0 - apex).cross(v3 - apex).normalize();
    let n_bottom = Vec3::NEG_Y;

    let vertices = vec![
        // Front face
        apex, v3, v2, // Back face
        apex, v1, v0, // Right face
        apex, v2, v1, // Left face
        apex, v0, v3, // Bottom face
        v0, v1, v2, v3,
    ];

    let normals = vec![
        // Front
        n_front, n_front, n_front, // Back
        n_back, n_back, n_back, // Right
        n_right, n_right, n_right, // Left
        n_left, n_left, n_left, // Bottom
        n_bottom, n_bottom, n_bottom, n_bottom,
    ];

    let indices = vec![
        [0, 1, 2],    // Front
        [3, 4, 5],    // Back
        [6, 7, 8],    // Right
        [9, 10, 11],  // Left
        [12, 13, 14], // Bottom tri 1
        [12, 14, 15], // Bottom tri 2
    ];

    Mesh::with_normals(vertices, indices, normals)
}

/// Create a torus with smooth normals
pub fn torus(
    major_radius: f32,
    minor_radius: f32,
    major_segments: u32,
    minor_segments: u32,
) -> Mesh {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=major_segments {
        let theta = 2.0 * PI * (i as f32) / (major_segments as f32);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Center of the tube at this major angle
        let tube_center = Vec3::new(major_radius * cos_theta, 0.0, major_radius * sin_theta);

        for j in 0..=minor_segments {
            let phi = 2.0 * PI * (j as f32) / (minor_segments as f32);
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            // Position on torus surface
            let x = (major_radius + minor_radius * cos_phi) * cos_theta;
            let y = minor_radius * sin_phi;
            let z = (major_radius + minor_radius * cos_phi) * sin_theta;

            let pos = Vec3::new(x, y, z);
            let normal = (pos - tube_center).normalize();

            vertices.push(pos);
            normals.push(normal);
        }
    }

    // Generate indices
    for i in 0..major_segments {
        for j in 0..minor_segments {
            let row1 = i * (minor_segments + 1);
            let row2 = (i + 1) * (minor_segments + 1);

            let a = row1 + j;
            let b = row2 + j;
            let c = row2 + j + 1;
            let d = row1 + j + 1;

            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }

    Mesh::with_normals(vertices, indices, normals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_creation() {
        let mesh = cube(2.0);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 12);
        assert!(mesh.bvh.is_some());
    }

    #[test]
    fn test_pyramid_creation() {
        let mesh = pyramid(2.0, 1.5);
        assert_eq!(mesh.indices.len(), 6);
        assert!(mesh.bvh.is_some());
    }

    #[test]
    fn test_torus_creation() {
        let mesh = torus(2.0, 0.5, 16, 8);
        assert!(mesh.tri_count() > 0);
        assert!(mesh.bvh.is_some());
    }

    #[test]
    fn test_ray_cube_hit() {
        let mesh = cube(2.0);

        // Ray from front, should hit
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = mesh.intersect(origin, dir);
        assert!(hit.is_some());

        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 0.01); // should hit at z=1, distance 4
    }

    #[test]
    fn test_ray_cube_miss() {
        let mesh = cube(2.0);

        // Ray missing the cube
        let origin = Vec3::new(10.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = mesh.intersect(origin, dir);
        assert!(hit.is_none());
    }
}
