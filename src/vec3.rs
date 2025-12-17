//! Vector math utilities using glam crate

use glam::Vec3;

// Type alias for our 3D vector
pub type Vector3 = Vec3;

// Common vector constants
pub const ZERO: Vector3 = Vector3::ZERO;
pub const ONE: Vector3 = Vector3::ONE;
pub const X_AXIS: Vector3 = Vector3::X;
pub const Y_AXIS: Vector3 = Vector3::Y;
pub const Z_AXIS: Vector3 = Vector3::Z;

/// Extension traits for additional vector operations
pub trait Vec3Ext {
    /// Calculate the norm (length) of the vector
    fn norm(&self) -> f32;

    /// Return a normalized version of the vector
    fn normalized(&self) -> Vector3;
}

impl Vec3Ext for Vector3 {
    fn norm(&self) -> f32 {
        self.length()
    }

    fn normalized(&self) -> Vector3 {
        self.normalize()
    }
}
