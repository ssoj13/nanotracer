//! Scene container: objects, lights, environment.
//!
//! After the GPU refactor the scene is a pure builder — it does not perform
//! intersection or shading on the CPU side. The renderer/splat crates pull
//! the data out via `nano-gpu::gpu_scene::build_gpu_scene*`.

use glam::{Quat, Vec2, Vec3};

use crate::environment::EnvironmentMap;
use crate::geometry::Object;

/// Scene light source.
///
/// Variants cover the four analytic geometric primitives plus the
/// image-based-lighting "environment" entry. Sampling conventions:
///
/// * `Point` keeps the tinyraytracer-style "unit-radiance, no falloff"
///   model: the contribution is `radiance · max(N·L, 0)`. This is
///   *non-physical* (a true point source has 1/r² falloff) but matches
///   the pre-area-light demo scenes — so converting `Light::Point` with
///   `color=(1,1,1)` and `intensity=1.0` reproduces the legacy output.
/// * `Rect`, `Sphere`, `Box` use the standard area-form Monte-Carlo
///   estimator: contribution = `radiance · cos_x · cos_y / (r² · pdf_A)`,
///   where `pdf_A = 1 / area` for uniform surface sampling.
/// * `Env` is image-based: the diffuse contribution comes from the
///   precomputed Lambertian-convolved SH irradiance (already includes
///   the cosine integral). Specular IBL is a separate long-horizon item.
#[derive(Debug, Clone, Copy)]
pub enum Light {
    Point {
        position: Vec3,
        color: Vec3,
        intensity: f32,
    },
    Rect {
        center: Vec3,
        /// Half-extent along the local u axis (world-space vector).
        u: Vec3,
        /// Half-extent along the local v axis (world-space vector).
        /// Must be linearly independent of `u`; the rectangle's normal
        /// is `normalize(u × v)`.
        v: Vec3,
        color: Vec3,
        intensity: f32,
        /// If `true`, emits from both faces; otherwise only the face
        /// whose normal points along `u × v`.
        two_sided: bool,
    },
    Sphere {
        center: Vec3,
        radius: f32,
        color: Vec3,
        intensity: f32,
    },
    /// Oriented bounding box (OBB) emitter. Six faces emit outward.
    Box {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        color: Vec3,
        intensity: f32,
    },
    /// Image-based lighting — references the scene's [`EnvironmentMap`].
    Env { intensity: f32 },
}

impl Light {
    /// Legacy constructor: unit-radiance point light at `position`. Used
    /// by demo scenes that predate the per-light colour/intensity fields.
    pub fn point(position: Vec3) -> Self {
        Self::Point {
            position,
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }

    /// Geometric surface area of the emitter. Returns `1.0` for delta
    /// lights (`Point`, `Env`) — those use a degenerate PDF and the area
    /// is never read by the sampler.
    pub fn area(&self) -> f32 {
        match self {
            Light::Point { .. } | Light::Env { .. } => 1.0,
            Light::Rect { u, v, .. } => 4.0 * u.cross(*v).length(),
            Light::Sphere { radius, .. } => 4.0 * std::f32::consts::PI * radius * radius,
            Light::Box { half_extents, .. } => {
                let hx = half_extents.x;
                let hy = half_extents.y;
                let hz = half_extents.z;
                // OBB surface area: 2·(w·h + w·d + h·d) with full extents.
                8.0 * (hx * hy + hx * hz + hy * hz)
            }
        }
    }

    /// Pre-multiplied emitted radiance `color · intensity`. For `Env`
    /// this returns `vec3(intensity)` — the per-direction spectrum
    /// lives inside the environment map sampled in-shader.
    pub fn radiance(&self) -> Vec3 {
        match self {
            Light::Point { color, intensity, .. }
            | Light::Rect { color, intensity, .. }
            | Light::Sphere { color, intensity, .. }
            | Light::Box { color, intensity, .. } => *color * *intensity,
            Light::Env { intensity } => Vec3::splat(*intensity),
        }
    }

    /// Sample a point on the emitter's surface with a uniform-area PDF.
    /// Returns `(point, normal, pdf_area)`. For delta / non-surface
    /// variants the return is conventional:
    /// * `Point` → `(position, Vec3::ZERO, 1.0)` (no surface)
    /// * `Env`   → `(Vec3::ZERO, Vec3::ZERO, 1.0)` (no surface)
    ///
    /// `rand_uv` should be uniform in `[0, 1)²`.
    pub fn sample(&self, rand_uv: Vec2) -> (Vec3, Vec3, f32) {
        match self {
            Light::Point { position, .. } => (*position, Vec3::ZERO, 1.0),
            Light::Env { .. } => (Vec3::ZERO, Vec3::ZERO, 1.0),
            Light::Rect { center, u, v, .. } => {
                // rand_uv ∈ [0,1)²  →  local ∈ [-1, +1)² on the rectangle.
                let s = rand_uv.x * 2.0 - 1.0;
                let t = rand_uv.y * 2.0 - 1.0;
                let point = *center + *u * s + *v * t;
                let normal = u.cross(*v).normalize_or_zero();
                (point, normal, 1.0 / self.area().max(1e-8))
            }
            Light::Sphere { center, radius, .. } => {
                // Uniform-area sphere sampling (Marsaglia): map u₁ → z,
                // u₂ → φ. Yields a unit-direction; offset by `center`
                // and scaled by `radius` gives the surface point.
                let z = 1.0 - 2.0 * rand_uv.x;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = rand_uv.y * std::f32::consts::TAU;
                let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                let point = *center + dir * *radius;
                (point, dir, 1.0 / self.area().max(1e-8))
            }
            Light::Box {
                center,
                half_extents,
                rotation,
                ..
            } => {
                // Sample a face weighted by its area, then sample uniformly
                // on that face. rand_uv.x drives face selection AND the
                // first in-face coordinate via stratified extraction:
                // once the face's CDF bracket is known, the remainder of
                // rand_uv.x inside that bracket is itself uniform in
                // [0, 1) — no fract() trick, no resampling.
                let hx = half_extents.x;
                let hy = half_extents.y;
                let hz = half_extents.z;
                // Face pair areas: (+X,-X) share area 4·hy·hz, etc.
                let pair_areas = [4.0 * hy * hz, 4.0 * hx * hz, 4.0 * hx * hy];
                let total = 2.0 * (pair_areas[0] + pair_areas[1] + pair_areas[2]);
                // Build the 6-face CDF bracket boundaries (front then back
                // for each axis: +X, -X, +Y, -Y, +Z, -Z).
                let mut cdf = [0.0f32; 7];
                cdf[1] = pair_areas[0];
                cdf[2] = 2.0 * pair_areas[0];
                cdf[3] = cdf[2] + pair_areas[1];
                cdf[4] = cdf[2] + 2.0 * pair_areas[1];
                cdf[5] = cdf[4] + pair_areas[2];
                cdf[6] = total;
                let pick = rand_uv.x * total;
                let mut face_idx = 0usize;
                while face_idx < 5 && pick >= cdf[face_idx + 1] {
                    face_idx += 1;
                }
                let lo = cdf[face_idx];
                let hi = cdf[face_idx + 1];
                let u_local = ((pick - lo) / (hi - lo).max(1e-8)) * 2.0 - 1.0;
                let v_local = rand_uv.y * 2.0 - 1.0;
                let (axis, sign) = match face_idx {
                    0 => (0usize, 1.0f32),
                    1 => (0, -1.0),
                    2 => (1, 1.0),
                    3 => (1, -1.0),
                    4 => (2, 1.0),
                    _ => (2, -1.0),
                };
                let mut local_point = Vec3::ZERO;
                let mut local_normal = Vec3::ZERO;
                match axis {
                    0 => {
                        local_point.x = sign * hx;
                        local_point.y = u_local * hy;
                        local_point.z = v_local * hz;
                        local_normal.x = sign;
                    }
                    1 => {
                        local_point.y = sign * hy;
                        local_point.x = u_local * hx;
                        local_point.z = v_local * hz;
                        local_normal.y = sign;
                    }
                    _ => {
                        local_point.z = sign * hz;
                        local_point.x = u_local * hx;
                        local_point.y = v_local * hy;
                        local_normal.z = sign;
                    }
                }
                let point = *center + *rotation * local_point;
                let normal = (*rotation * local_normal).normalize_or_zero();
                (point, normal, 1.0 / total.max(1e-8))
            }
        }
    }
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

    /// `true` if the scene has at least one `Light::Env`. Used by the
    /// build pipeline to decide whether to auto-insert a default env
    /// light when an environment map is present but no explicit
    /// `Light::Env` was added.
    pub fn has_env_light(&self) -> bool {
        self.lights.iter().any(|l| matches!(l, Light::Env { .. }))
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn point_area_radiance() {
        let l = Light::Point {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::new(0.4, 0.5, 0.6),
            intensity: 2.0,
        };
        assert!((l.area() - 1.0).abs() < EPS);
        let r = l.radiance();
        assert!((r.x - 0.8).abs() < EPS);
        assert!((r.y - 1.0).abs() < EPS);
        assert!((r.z - 1.2).abs() < EPS);
    }

    #[test]
    fn rect_area_matches_cross_product() {
        // u along +X with half-len 2, v along +Y with half-len 3
        // → full extents 4×6 → area 24.
        let l = Light::Rect {
            center: Vec3::ZERO,
            u: Vec3::new(2.0, 0.0, 0.0),
            v: Vec3::new(0.0, 3.0, 0.0),
            color: Vec3::ONE,
            intensity: 1.0,
            two_sided: false,
        };
        assert!((l.area() - 24.0).abs() < EPS);
        let (p, n, pdf) = l.sample(Vec2::new(0.25, 0.75));
        // Point must lie inside the rectangle: |p·x̂| ≤ 2, |p·ŷ| ≤ 3.
        assert!(p.x.abs() <= 2.0 + EPS);
        assert!(p.y.abs() <= 3.0 + EPS);
        assert!(p.z.abs() < EPS);
        // Normal aligned with +Z (right-hand u × v).
        assert!((n.z - 1.0).abs() < EPS);
        assert!((pdf - 1.0 / 24.0).abs() < EPS);
    }

    #[test]
    fn sphere_area_is_four_pi_r_squared() {
        let l = Light::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
            color: Vec3::ONE,
            intensity: 1.0,
        };
        let expected = 4.0 * std::f32::consts::PI * 4.0;
        assert!((l.area() - expected).abs() < 1e-3);
        // Sampled point lies on the sphere; normal is the outward unit dir.
        let (p, n, _pdf) = l.sample(Vec2::new(0.3, 0.7));
        assert!((p.length() - 2.0).abs() < 1e-3);
        assert!((n.length() - 1.0).abs() < 1e-3);
        assert!((p.normalize().dot(n) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn box_area_obb_six_faces() {
        // Half-extents (1, 2, 3) → full extents 2×4×6.
        // Surface area = 2·(2·4 + 2·6 + 4·6) = 2·(8 + 12 + 24) = 88.
        let l = Light::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            color: Vec3::ONE,
            intensity: 1.0,
        };
        assert!((l.area() - 88.0).abs() < EPS);
        // Sampled point on one of the six faces: at least one coordinate
        // is at its half-extent (axis-aligned, identity rotation).
        let (p, n, _pdf) = l.sample(Vec2::new(0.4, 0.6));
        let on_face = (p.x.abs() - 1.0).abs() < EPS
            || (p.y.abs() - 2.0).abs() < EPS
            || (p.z.abs() - 3.0).abs() < EPS;
        assert!(on_face, "sampled point not on a face: {:?}", p);
        assert!((n.length() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn box_rotation_applied() {
        // Box rotated 90° about Y; +Z face normal must become +X.
        let l = Light::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::splat(1.0),
            rotation: Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
            color: Vec3::ONE,
            intensity: 1.0,
        };
        // Sample enough times to land on every face at least once and
        // verify each normal is one of ±X, ±Y, ±Z (rotated). After
        // rotation the legal normals are still axis-aligned (because
        // 90° about Y just permutes axes), so we check that property.
        for i in 0..32 {
            let u = (i as f32) / 32.0;
            let v = (i as f32 * 0.6180339).fract();
            let (_, n, _) = l.sample(Vec2::new(u, v));
            let max_abs = n.x.abs().max(n.y.abs()).max(n.z.abs());
            assert!((max_abs - 1.0).abs() < 1e-3, "non-axis normal: {:?}", n);
        }
    }

    #[test]
    fn env_radiance_is_intensity() {
        let l = Light::Env { intensity: 1.5 };
        assert_eq!(l.radiance(), Vec3::splat(1.5));
    }

    #[test]
    fn has_env_light_detects_variant() {
        let mut s = Scene::new();
        assert!(!s.has_env_light());
        s.add_light(Light::point(Vec3::ZERO));
        assert!(!s.has_env_light());
        s.add_light(Light::Env { intensity: 1.0 });
        assert!(s.has_env_light());
    }
}
