//! Scene definition and intersection logic

use glam::Vec3;
use std::cmp::Ordering;

use crate::environment::{DEFAULT_SKY_COLOR, EnvironmentMap};
use crate::geometry::{Geometry, Hit, Object, Sphere};
use crate::material::{Material, checkerboard_material};
use rtbvh::{Aabb, Ray};

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
    /// Scene-level BVH for object culling
    scene_bvh: Option<SceneBvh>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
            environment: None,
            checkerboard_enabled: true,
            scene_bvh: None,
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

    /// Rebuild the scene-level BVH (call after modifying objects)
    pub fn rebuild_scene_bvh(&mut self) {
        if self.objects.is_empty() {
            self.scene_bvh = None;
            return;
        }

        let proxies = self
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| SceneObjectProxy::new(index, Self::object_aabb(object)))
            .collect::<Vec<_>>();

        self.scene_bvh = if proxies.is_empty() {
            None
        } else {
            Some(SceneBvh::new(proxies))
        };
    }

    /// Intersect a ray with the entire scene
    pub fn intersect(&self, orig: Vec3, dir: Vec3) -> Intersection {
        let mut best_t = f32::MAX;
        let mut result = Intersection::empty();
        let mut ray = Ray::from((orig, dir));

        if let Some(scene_bvh) = &self.scene_bvh {
            if !scene_bvh.nodes.is_empty() {
                let mut stack = vec![0usize];
                while let Some(node_idx) = stack.pop() {
                    let node = &scene_bvh.nodes[node_idx];
                    if node.aabb.intersect(&ray).is_none() {
                        continue;
                    }
                    if node.is_leaf() {
                        for proxy_idx in node.start..node.end {
                            let proxy = &scene_bvh.proxies[proxy_idx];
                            let object = &self.objects[proxy.index];
                            if let Some(hit) = object.geometry.intersect(orig, dir) {
                                if hit.t < best_t {
                                    best_t = hit.t;
                                    result = Intersection::from_hit(&hit, object.material);
                                    ray.t = best_t;
                                }
                            }
                        }
                    } else {
                        if node.right >= 0 {
                            stack.push(node.right as usize);
                        }
                        if node.left >= 0 {
                            stack.push(node.left as usize);
                        }
                    }
                }
            }
        } else {
            for object in &self.objects {
                if let Some(hit) = object.geometry.intersect(orig, dir) {
                    if hit.t < best_t {
                        best_t = hit.t;
                        result = Intersection::from_hit(&hit, object.material);
                    }
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
        // Fast early exit for rays parallel to plane
        let dir_y_abs = dir.y.abs();
        if dir_y_abs < 0.001 {
            return None;
        }

        // Calculate intersection with y = -4 plane
        let t = -(orig.y + 4.0) / dir.y;
        if t < 0.001 {
            return None;
        }

        let point = orig + dir * t;

        // Fast bounds check
        let px = point.x;
        let pz = point.z;
        if px.abs() >= 10.0 || pz >= -10.0 || pz <= -30.0 {
            return None;
        }

        // Fast checkerboard pattern calculation
        let checker = ((0.5 * px + 1000.0) as i32 + (0.5 * pz) as i32) & 1;
        // Use precomputed colors to avoid allocation
        let color = if checker != 0 {
            Vec3::new(0.3, 0.3, 0.3)  // white tile
        } else {
            Vec3::new(0.3, 0.2, 0.1)  // brown tile
        };

        Some((t, point, checkerboard_material(color)))
    }

    fn object_aabb(object: &Object) -> Aabb {
        match &object.geometry {
            Geometry::Sphere { center, radius } => {
                let center = *center;
                let offset = Vec3::splat(*radius);
                Aabb::from((center - offset, center + offset))
            }
            Geometry::Mesh(mesh) => mesh.bounding_box(),
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

struct SceneObjectProxy {
    index: usize,
    center: Vec3,
    aabb: Aabb,
}

impl SceneObjectProxy {
    fn new(index: usize, aabb: Aabb) -> Self {
        Self {
            index,
            center: aabb.center(),
            aabb,
        }
    }
}

struct SceneBvhNode {
    aabb: Aabb,
    start: usize,
    end: usize,
    left: i32,
    right: i32,
}

impl SceneBvhNode {
    fn is_leaf(&self) -> bool {
        self.left < 0 && self.right < 0
    }
}

struct SceneBvh {
    nodes: Vec<SceneBvhNode>,
    proxies: Vec<SceneObjectProxy>,
}

impl SceneBvh {
    fn new(mut proxies: Vec<SceneObjectProxy>) -> Self {
        let mut nodes = Vec::with_capacity(proxies.len().max(1) * 2);
        let len = proxies.len();
        if len > 0 {
            Self::build_node(&mut nodes, &mut proxies, 0, len);
        }
        Self { nodes, proxies }
    }

    fn build_node(
        nodes: &mut Vec<SceneBvhNode>,
        proxies: &mut [SceneObjectProxy],
        start: usize,
        end: usize,
    ) -> i32 {
        let count = end - start;
        let mut bounds = Aabb::empty();
        for proxy in &proxies[start..end] {
            bounds = bounds.union_of(&proxy.aabb);
        }

        let node_idx = nodes.len();
        nodes.push(SceneBvhNode {
            aabb: bounds,
            start,
            end,
            left: -1,
            right: -1,
        });

        if count <= 2 {
            return node_idx as i32;
        }

        let axis = bounds.longest_axis();
        proxies[start..end].sort_by(|a, b| {
            a.center[axis]
                .partial_cmp(&b.center[axis])
                .unwrap_or(Ordering::Equal)
        });

        let mid = start + count / 2;
        let left = Self::build_node(nodes, proxies, start, mid);
        let right = Self::build_node(nodes, proxies, mid, end);

        let node = &mut nodes[node_idx];
        node.left = left;
        node.right = right;
        node.start = start;
        node.end = end;

        node_idx as i32
    }
}
