//! Unified geometry types fed to the GPU acceleration structure.
//!
//! After the GPU refactor, the CPU side only holds *data*: vertices,
//! indices, material refs. Ray intersection lives on the GPU.

use glam::Vec3;

use crate::material::Material;
use crate::mesh::Mesh;

/// Discriminator over geometry primitives the scene can contain.
#[derive(Debug, Clone)]
pub enum Geometry {
    /// Analytic sphere — the GPU tessellates it to a UV mesh before BLAS build.
    Sphere { center: Vec3, radius: f32 },
    /// Triangle mesh.
    Mesh(Mesh),
}

/// Scene object: geometry + the material applied to all of its surface.
#[derive(Debug, Clone)]
pub struct Object {
    pub geometry: Geometry,
    pub material: Material,
}

impl Object {
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Self { geometry, material }
    }

    /// Convenience constructor for an analytic sphere.
    pub fn sphere(center: Vec3, radius: f32, material: Material) -> Self {
        Self {
            geometry: Geometry::Sphere { center, radius },
            material,
        }
    }

    /// Convenience constructor wrapping a mesh.
    pub fn mesh(mesh: Mesh, material: Material) -> Self {
        Self {
            geometry: Geometry::Mesh(mesh),
            material,
        }
    }
}
