//! Geometric primitives and unified geometry system

use glam::Vec3;

use crate::material::Material;
use crate::mesh::Mesh;

/// Unified geometry enum for all primitive types
#[derive(Debug, Clone)]
pub enum Geometry {
    /// Analytic sphere (fast, perfect)
    Sphere { center: Vec3, radius: f32 },
    /// Triangle mesh with BVH
    Mesh(Mesh),
}

/// Scene object combining geometry and material
#[derive(Debug, Clone)]
pub struct Object {
    pub geometry: Geometry,
    pub material: Material,
}

impl Object {
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Self { geometry, material }
    }

    /// Create sphere object
    pub fn sphere(center: Vec3, radius: f32, material: Material) -> Self {
        Self {
            geometry: Geometry::Sphere { center, radius },
            material,
        }
    }

    /// Create mesh object
    pub fn mesh(mesh: Mesh, material: Material) -> Self {
        Self {
            geometry: Geometry::Mesh(mesh),
            material,
        }
    }
}

/// Hit information from ray intersection
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

impl Geometry {
    /// Ray intersection, returns hit info if intersection found
    pub fn intersect(&self, origin: Vec3, dir: Vec3) -> Option<Hit> {
        match self {
            Geometry::Sphere { center, radius } => intersect_sphere(origin, dir, *center, *radius),
            Geometry::Mesh(mesh) => mesh.intersect(origin, dir).map(|tri_hit| {
                let point = origin + dir * tri_hit.t;
                let normal = mesh.normal_at(&tri_hit);
                Hit {
                    t: tri_hit.t,
                    point,
                    normal,
                }
            }),
        }
    }

    /// Surface area for sampling
    pub fn surface_area(&self) -> f32 {
        match self {
            Geometry::Sphere { radius, .. } => 4.0 * std::f32::consts::PI * radius * radius,
            Geometry::Mesh(mesh) => mesh.surface_area(),
        }
    }
}

/// Ray-sphere intersection
fn intersect_sphere(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<Hit> {
    let l = center - origin;
    let tca = l.dot(dir);
    let d2 = l.dot(l) - tca * tca;
    let r2 = radius * radius;

    if d2 > r2 {
        return None;
    }

    let thc = (r2 - d2).sqrt();
    let t0 = tca - thc;
    let t1 = tca + thc;

    // Find closest valid intersection (t > 0.001 to avoid self-intersection)
    let t = if t0 > 0.001 {
        t0
    } else if t1 > 0.001 {
        t1
    } else {
        return None;
    };

    let point = origin + dir * t;
    let normal = (point - center).normalize();

    Some(Hit { t, point, normal })
}

// Legacy compatibility: Sphere struct for existing code
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    /// Convert to Object
    pub fn to_object(self) -> Object {
        Object::sphere(self.center, self.radius, self.material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::cube;

    #[test]
    fn test_sphere_hit() {
        let geom = Geometry::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };

        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        let hit = geom.intersect(origin, dir);
        assert!(hit.is_some());

        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 0.01);
        assert!((h.normal.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mesh_hit() {
        let mesh = cube(2.0);
        let geom = Geometry::Mesh(mesh);

        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        let hit = geom.intersect(origin, dir);
        assert!(hit.is_some());
    }
}
