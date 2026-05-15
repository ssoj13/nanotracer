//! Scene container: objects, lights, environment.
//!
//! After the GPU refactor the scene is a pure builder — it does not perform
//! intersection or shading on the CPU side. The renderer/splat crates pull
//! the data out via `nano-gpu::gpu_scene::build_gpu_scene*`.

use glam::Vec3;

use crate::environment::EnvironmentMap;
use crate::geometry::Object;

/// Point light source.
///
/// All lights are point lights with implicit unit radiance; intensity is
/// implicit in the world units — same convention as the original
/// tinyraytracer demo.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vec3,
}

/// Scene built by the CLI and consumed by the GPU pipelines.
pub struct Scene {
    pub objects: Vec<Object>,
    pub lights: Vec<Light>,
    pub environment: Option<EnvironmentMap>,
    /// Add the procedural checkerboard plane at `y = -4`.
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

    pub fn add_object(&mut self, object: Object) {
        self.objects.push(object);
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
