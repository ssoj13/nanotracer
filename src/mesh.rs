//! Mesh geometry with BVH acceleration
//!
//! Supports triangle meshes with per-vertex normals and BVH for fast ray intersection.

use glam::Vec3;
use std::f32::consts::PI;

/// Triangle mesh with BVH acceleration
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    pub normals: Vec<Vec3>, // per-vertex normals
    bvh: Option<MeshBvh>,
}

/// Simple BVH node for mesh triangles
#[derive(Debug, Clone)]
struct BvhNode {
    aabb: Aabb,
    /// If leaf: triangle indices, else empty
    triangles: Vec<u32>,
    /// Child node indices (0 = invalid)
    left: usize,
    right: usize,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Ray-AABB intersection (slab method)
    pub fn intersect_ray(&self, origin: Vec3, dir_inv: Vec3, t_max: f32) -> bool {
        let t1 = (self.min - origin) * dir_inv;
        let t2 = (self.max - origin) * dir_inv;

        let t_min_v = t1.min(t2);
        let t_max_v = t1.max(t2);

        let t_enter = t_min_v.x.max(t_min_v.y).max(t_min_v.z).max(0.001);
        let t_exit = t_max_v.x.min(t_max_v.y).min(t_max_v.z).min(t_max);

        t_enter <= t_exit
    }
}

/// BVH for mesh triangles
#[derive(Debug, Clone)]
struct MeshBvh {
    nodes: Vec<BvhNode>,
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
            return;
        }

        let tri_count = self.indices.len();
        let mut tri_indices: Vec<u32> = (0..tri_count as u32).collect();
        let mut tri_aabbs: Vec<Aabb> = self
            .indices
            .iter()
            .map(|tri| self.triangle_aabb(tri))
            .collect();
        let mut tri_centers: Vec<Vec3> = tri_aabbs.iter().map(|aabb| aabb.center()).collect();

        let mut nodes = Vec::with_capacity(tri_count * 2);

        Self::build_bvh_recursive(
            &mut nodes,
            &mut tri_indices,
            &mut tri_aabbs,
            &mut tri_centers,
            0,
            tri_count,
        );

        self.bvh = Some(MeshBvh { nodes });
    }

    fn triangle_aabb(&self, tri: &[u32; 3]) -> Aabb {
        let v0 = self.vertices[tri[0] as usize];
        let v1 = self.vertices[tri[1] as usize];
        let v2 = self.vertices[tri[2] as usize];
        Aabb {
            min: v0.min(v1).min(v2),
            max: v0.max(v1).max(v2),
        }
    }

    fn build_bvh_recursive(
        nodes: &mut Vec<BvhNode>,
        tri_indices: &mut [u32],
        tri_aabbs: &mut [Aabb],
        tri_centers: &mut [Vec3],
        start: usize,
        end: usize,
    ) -> usize {
        let count = end - start;

        // Compute bounds
        let mut bounds = Aabb::empty();
        for i in start..end {
            bounds = bounds.merge(&tri_aabbs[i]);
        }

        // Leaf node if few triangles
        if count <= 4 {
            let node_idx = nodes.len();
            nodes.push(BvhNode {
                aabb: bounds,
                triangles: tri_indices[start..end].to_vec(),
                left: 0,
                right: 0,
            });
            return node_idx;
        }

        // Find best split axis (longest)
        let extent = bounds.max - bounds.min;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };

        // Sort by center along axis
        let slice = &mut tri_indices[start..end];
        let aabb_slice = &mut tri_aabbs[start..end];
        let center_slice = &mut tri_centers[start..end];

        // Simple sorting by center coordinate
        for i in 0..slice.len() {
            for j in i + 1..slice.len() {
                let ci = match axis {
                    0 => center_slice[i].x,
                    1 => center_slice[i].y,
                    _ => center_slice[i].z,
                };
                let cj = match axis {
                    0 => center_slice[j].x,
                    1 => center_slice[j].y,
                    _ => center_slice[j].z,
                };
                if cj < ci {
                    slice.swap(i, j);
                    aabb_slice.swap(i, j);
                    center_slice.swap(i, j);
                }
            }
        }

        let mid = start + count / 2;

        // Reserve node index
        let node_idx = nodes.len();
        nodes.push(BvhNode {
            aabb: bounds,
            triangles: vec![],
            left: 0,
            right: 0,
        });

        // Build children
        let left =
            Self::build_bvh_recursive(nodes, tri_indices, tri_aabbs, tri_centers, start, mid);
        let right = Self::build_bvh_recursive(nodes, tri_indices, tri_aabbs, tri_centers, mid, end);

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Ray-mesh intersection using BVH
    pub fn intersect(&self, origin: Vec3, dir: Vec3) -> Option<TriangleHit> {
        let bvh = self.bvh.as_ref()?;
        if bvh.nodes.is_empty() {
            return None;
        }

        let dir_inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        let mut best: Option<TriangleHit> = None;
        let mut t_max = f32::INFINITY;

        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            let node = &bvh.nodes[node_idx];

            if !node.aabb.intersect_ray(origin, dir_inv, t_max) {
                continue;
            }

            // Leaf node
            if !node.triangles.is_empty() {
                for &tri_idx in &node.triangles {
                    if let Some(hit) = self.intersect_triangle(origin, dir, tri_idx) {
                        if hit.t < t_max && hit.t > 0.001 {
                            t_max = hit.t;
                            best = Some(hit);
                        }
                    }
                }
            } else {
                // Internal node
                if node.left != 0 {
                    stack.push(node.left);
                }
                if node.right != 0 {
                    stack.push(node.right);
                }
            }
        }

        best
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
