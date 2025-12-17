//! Material definitions for the raytracer

use glam::Vec3;

/// Material properties for surfaces
#[derive(Debug, Clone, Copy)]
pub struct Material {
    /// Refractive index for refraction calculations
    pub refractive_index: f32,

    /// Albedo components: diffuse, specular, reflection, refraction
    pub albedo: [f32; 4],

    /// Diffuse color of the material
    pub diffuse_color: Vec3,

    /// Specular exponent for Phong shading
    pub specular_exponent: f32,
}

// Predefined materials
pub const IVORY: Material = Material {
    refractive_index: 1.0,
    albedo: [0.9, 0.5, 0.1, 0.0],
    diffuse_color: Vec3::new(0.4, 0.4, 0.3),
    specular_exponent: 50.0,
};

pub const GLASS: Material = Material {
    refractive_index: 1.1,
    albedo: [0.0, 0.9, 0.1, 0.8],
    diffuse_color: Vec3::new(0.0, 0.0, 0.0),
    specular_exponent: 125.0,
};

pub const RED_RUBBER: Material = Material {
    refractive_index: 1.0,
    albedo: [1.4, 0.3, 0.0, 0.0],
    diffuse_color: Vec3::new(0.3, 0.1, 0.1),
    specular_exponent: 10.0,
};

pub const MIRROR: Material = Material {
    refractive_index: 1.0,
    albedo: [0.0, 16.0, 0.8, 0.0],
    diffuse_color: Vec3::new(1.0, 1.0, 1.0),
    specular_exponent: 1425.0,
};

/// Matte diffuse material (for meshes)
pub const MATTE_WHITE: Material = Material {
    refractive_index: 1.0,
    albedo: [1.0, 0.1, 0.0, 0.0],
    diffuse_color: Vec3::new(0.8, 0.8, 0.8),
    specular_exponent: 10.0,
};

pub const MATTE_RED: Material = Material {
    refractive_index: 1.0,
    albedo: [1.0, 0.1, 0.0, 0.0],
    diffuse_color: Vec3::new(0.8, 0.2, 0.2),
    specular_exponent: 10.0,
};

pub const MATTE_BLUE: Material = Material {
    refractive_index: 1.0,
    albedo: [1.0, 0.1, 0.0, 0.0],
    diffuse_color: Vec3::new(0.2, 0.2, 0.8),
    specular_exponent: 10.0,
};

pub const MATTE_GREEN: Material = Material {
    refractive_index: 1.0,
    albedo: [1.0, 0.1, 0.0, 0.0],
    diffuse_color: Vec3::new(0.2, 0.8, 0.2),
    specular_exponent: 10.0,
};
