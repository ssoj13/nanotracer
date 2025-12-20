//! Scene definition and intersection logic

use glam::Vec3;

use crate::environment::{DEFAULT_SKY_COLOR, EnvironmentMap};
use crate::geometry::{Hit, Object, Sphere};
use crate::material::{Material, checkerboard_material};

/// Light source in the scene
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vec3,
}

/// Intersection result with material info
#[derive(Debug, Clone, Copy)]
pub struct Intersection {
    pub hit: bool,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: Material,
}

impl Intersection {
    pub fn new(hit: bool, point: Vec3, normal: Vec3, material: Material) -> Self {
        Self {
            hit,
            point,
            normal,
            material,
        }
    }

    pub fn empty() -> Self {
        Self {
            hit: false,
            point: Vec3::ZERO,
            normal: Vec3::ZERO,
            material: Material {
                refractive_index: 1.0,
                albedo: [0.0; 4],
                diffuse_color: Vec3::ZERO,
                specular_exponent: 0.0,
            },
        }
    }

    pub fn from_hit(hit: &Hit, material: Material) -> Self {
        Self {
            hit: true,
            point: hit.point,
            normal: hit.normal,
            material,
        }
    }
}

/// Scene containing all objects and lights
pub struct Scene {
    /// All scene objects (unified geometry + material)
    pub objects: Vec<Object>,
    /// Light sources
    pub lights: Vec<Light>,
    /// Environment map
    pub environment: Option<EnvironmentMap>,
    /// Enable checkerboard plane
    pub checkerboard_enabled: bool,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
            environment: None,
            checkerboard_enabled: true,
        }
    }

    pub fn set_environment(&mut self, environment: EnvironmentMap) {
        self.environment = Some(environment);
    }

    /// Sample environment map or return default sky color
    pub fn sample_environment(&self, direction: Vec3) -> Vec3 {
        match &self.environment {
            Some(env) => env.sample(direction),
            None => DEFAULT_SKY_COLOR,
        }
    }

    /// Add a sphere (converts to unified Object internally)
    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.objects.push(sphere.to_object());
    }

    /// Add an object (new API)
    pub fn add_object(&mut self, object: Object) {
        self.objects.push(object);
    }

    /// Add a light
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    /// Intersect a ray with the entire scene
    pub fn intersect(&self, orig: Vec3, dir: Vec3) -> Intersection {
        let mut best_t = f32::MAX;
        let mut result = Intersection::empty();

        // Check all objects
        for object in &self.objects {
            if let Some(hit) = object.geometry.intersect(orig, dir) {
                if hit.t < best_t {
                    best_t = hit.t;
                    result = Intersection::from_hit(&hit, object.material);
                }
            }
        }

        // Check checkerboard plane (y = -4)
        if self.checkerboard_enabled {
            if let Some((t, point, material)) = self.intersect_checkerboard(orig, dir) {
                if t < best_t {
                    best_t = t;
                    result = Intersection::new(true, point, Vec3::Y, material);
                }
            }
        }

        result.hit = best_t < 1000.0;
        result
    }

    /// Checkerboard plane intersection (legacy)
    fn intersect_checkerboard(&self, orig: Vec3, dir: Vec3) -> Option<(f32, Vec3, Material)> {
        if dir.y.abs() < 0.001 {
            return None;
        }

        let t = -(orig.y + 4.0) / dir.y;
        if t < 0.001 {
            return None;
        }

        let point = orig + dir * t;

        // Bounds check
        if point.x.abs() >= 10.0 || point.z >= -10.0 || point.z <= -30.0 {
            return None;
        }

        // Checkerboard pattern
        let checker = ((0.5 * point.x + 1000.0) as i32 + (0.5 * point.z) as i32) & 1;
        let color = if checker == 1 {
            Vec3::new(0.3, 0.3, 0.3)
        } else {
            Vec3::new(0.3, 0.2, 0.1)
        };

        Some((t, point, checkerboard_material(color)))
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
