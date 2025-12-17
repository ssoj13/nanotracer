//! Ray structure and operations

use crate::vec3::{Vec3Ext, Vector3};

/// A ray with origin and direction
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vector3, direction: Vector3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
        }
    }

    /// Get a point along the ray at distance t
    pub fn at(&self, t: f32) -> Vector3 {
        self.origin + self.direction * t
    }
}
