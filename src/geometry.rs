//! Geometric primitives for the raytracer

use crate::material::Material;
use crate::vec3::Vector3;

/// A sphere primitive with center, radius, and material
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vector3,
    pub radius: f32,
    pub material: Material,
}

impl Sphere {
    /// Create a new sphere
    pub fn new(center: Vector3, radius: f32, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

/// Calculate ray-sphere intersection
/// Returns (intersection_found, distance) tuple
pub fn ray_sphere_intersect(orig: Vector3, dir: Vector3, sphere: &Sphere) -> (bool, f32) {
    let l = sphere.center - orig;
    let tca = l.dot(dir);
    let d2 = l.dot(l) - tca * tca;

    if d2 > sphere.radius * sphere.radius {
        return (false, 0.0);
    }

    let thc = (sphere.radius * sphere.radius - d2).sqrt();
    let t0 = tca - thc;
    let t1 = tca + thc;

    // Offset the original point by 0.001 to avoid occlusion by the object itself
    if t0 > 0.001 {
        (true, t0)
    } else if t1 > 0.001 {
        (true, t1)
    } else {
        (false, 0.0)
    }
}
