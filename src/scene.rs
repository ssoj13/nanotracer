//! Scene definition and intersection logic

use crate::vec3::{Vector3, Vec3Ext};
use crate::geometry::{Sphere, ray_sphere_intersect};
use crate::material::Material;
use crate::environment::{EnvironmentMap, DEFAULT_SKY_COLOR};

/// Light source in the scene
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vector3,
}

/// Intersection result
#[derive(Debug, Clone, Copy)]
pub struct Intersection {
    pub hit: bool,
    pub point: Vector3,
    pub normal: Vector3,
    pub material: Material,
}

impl Intersection {
    pub fn new(hit: bool, point: Vector3, normal: Vector3, material: Material) -> Self {
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
            point: Vector3::ZERO,
            normal: Vector3::ZERO,
            material: Material {
                refractive_index: 1.0,
                albedo: [0.0; 4],
                diffuse_color: Vector3::ZERO,
                specular_exponent: 0.0,
            },
        }
    }
}

/// Scene containing all objects and lights
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub lights: Vec<Light>,
    pub environment: Option<EnvironmentMap>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            lights: Vec::new(),
            environment: None,
        }
    }

    pub fn set_environment(&mut self, environment: EnvironmentMap) {
        self.environment = Some(environment);
    }

    /// Sample the environment map or return default sky color
    pub fn sample_environment(&self, direction: Vector3) -> Vector3 {
        match &self.environment {
            Some(env) => env.sample(direction),
            None => DEFAULT_SKY_COLOR,
        }
    }
    
    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.spheres.push(sphere);
    }
    
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }
    
    /// Intersect a ray with the entire scene
    pub fn intersect(&self, orig: Vector3, dir: Vector3) -> Intersection {
        let mut point = Vector3::ZERO;
        let mut normal = Vector3::ZERO;
        let mut material = Material {
            refractive_index: 1.0,
            albedo: [0.0; 4],
            diffuse_color: Vector3::ZERO,
            specular_exponent: 0.0,
        };
        
        let mut nearest_dist = 1e10;
        
        // Intersect with checkerboard plane (y = -4)
        if dir.y.abs() > 0.001 {
            // Avoid division by zero
            let d = -(orig.y + 4.0) / dir.y;
            let p = orig + dir * d;
            
            if d > 0.001 && d < nearest_dist && p.x.abs() < 10.0 && p.z < -10.0 && p.z > -30.0 {
                nearest_dist = d;
                point = p;
                normal = Vector3::new(0.0, 1.0, 0.0);
                
                // Checkerboard pattern
                let color_factor = ((0.5 * p.x + 1000.0) as i32 + (0.5 * p.z) as i32) & 1;
                material.diffuse_color = if color_factor == 1 {
                    Vector3::new(0.3, 0.3, 0.3)
                } else {
                    Vector3::new(0.3, 0.2, 0.1)
                };
            }
        }
        
        // Intersect with all spheres
        for sphere in &self.spheres {
            let (intersection, d) = ray_sphere_intersect(orig, dir, sphere);
            if !intersection || d > nearest_dist {
                continue;
            }
            
            nearest_dist = d;
            point = orig + dir * nearest_dist;
            normal = (point - sphere.center).normalized();
            material = sphere.material;
        }
        
        Intersection::new(nearest_dist < 1000.0, point, normal, material)
    }
}