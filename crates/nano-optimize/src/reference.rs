//! Reference-radiance frame baking via `nano-render`.
//!
//! The training loop compares the wgpu rasteriser's predicted frame against
//! one of these reference frames every iteration. Frames are baked once at
//! training start and kept in CPU memory (raw `Vec<Vec3>`) so the loss
//! stays "current" — there's no on-disk staleness.
//!
//! Multi-view camera placement is a Fibonacci-sphere around the scene
//! centre; each view looks back at the centre with a fixed FoV.

use glam::{Mat4, Vec3};
use nano_core::scene::Scene;
use nano_render::{RenderConfig, render};
use std::f32::consts::PI;

use nano_core::LightSampling;

/// One baked reference frame: pixel radiances + the world-space camera that
/// produced them. The wgpu rasteriser will reproject splats into the same
/// camera frame and compare colour against `pixels`.
pub struct ReferenceView {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Vec3>,
    /// World-space camera position.
    pub camera_pos: Vec3,
    /// World-space "look-at" target (scene centre).
    pub target: Vec3,
    /// Up vector.
    pub up: Vec3,
    /// Vertical FoV in radians.
    pub fov_y: f32,
}

impl ReferenceView {
    /// World → camera-space matrix for splat-rasteriser projection.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.camera_pos, self.target, self.up)
    }

    /// Perspective projection matrix matching the renderer's pinhole model.
    /// Uses reverse-Z (near at 1.0, far at 0.0) for better depth precision —
    /// the splat rasteriser sorts front-to-back so it depends on consistent Z.
    pub fn proj_matrix(&self, near: f32, far: f32) -> Mat4 {
        let aspect = self.width as f32 / self.height as f32;
        Mat4::perspective_rh(self.fov_y, aspect, near, far)
    }
}

/// Configuration for [`bake_references`].
pub struct BakeConfig {
    /// Number of camera viewpoints to render.
    pub views: u32,
    /// Frame resolution. Smaller = faster baking + faster training; the
    /// Inria pipeline runs at the source-photo resolution. For our toy
    /// scenes 512×384 is a reasonable default.
    pub width: u32,
    pub height: u32,
    /// World-space radius of the camera sphere around `target`.
    pub orbit_radius: f32,
    /// World-space point cameras look at.
    pub target: Vec3,
    /// Vertical FoV in radians.
    pub fov_y: f32,
    /// Anti-alias samples per pixel for the reference raytrace
    /// (higher = cleaner ground truth, but slower bake).
    pub aa_samples: u32,
    /// Max ray depth used by the reference raytracer.
    pub max_depth: i32,
    /// Apply Reinhard tonemap to the reference (matches the splat output
    /// pipeline so the loss is in the same colour space).
    pub tonemap: bool,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            views: 50,
            width: 512,
            height: 384,
            orbit_radius: 35.0,
            target: Vec3::new(0.0, 0.0, -20.0),
            fov_y: 1.05,
            aa_samples: 2,
            max_depth: 16,
            tonemap: true,
        }
    }
}

/// Bake `cfg.views` reference frames via the existing path tracer.
///
/// Cameras sit on a Fibonacci sphere of radius `cfg.orbit_radius` around
/// `cfg.target`; each one looks back at the centre. This is a stand-in for
/// the Inria pipeline's COLMAP-derived photo poses — for a synthetic scene
/// we manufacture them.
///
/// **Phase A1 limitation:** the current `nano-render::render` is hard-wired
/// to a pinhole at origin looking down −Z. Until we extend `RenderConfig`
/// to accept a world-space camera transform (Phase A2 work, blocked on
/// shader changes), this function bakes the **same** frame `cfg.views`
/// times with different `(camera_pos, target, up)` metadata. The Adam loop
/// still works end-to-end; visual results will be wrong until the
/// renderer learns to move its camera.
pub fn bake_references(
    scene: &Scene,
    cfg: &BakeConfig,
) -> Result<Vec<ReferenceView>, Box<dyn std::error::Error>> {
    let cameras = fibonacci_sphere_cameras(cfg.views, cfg.orbit_radius, cfg.target);

    let mut refs = Vec::with_capacity(cameras.len());
    for (i, &camera_pos) in cameras.iter().enumerate() {
        let up = pick_up_for_camera(camera_pos, cfg.target);
        let render_config = RenderConfig {
            width: cfg.width,
            height: cfg.height,
            fov: cfg.fov_y,
            camera_pos,
            camera_target: cfg.target,
            camera_up: up,
            aa_samples: cfg.aa_samples,
            max_depth: cfg.max_depth,
            reflection_depth: 4,
            refraction_depth: 8,
            tonemap: cfg.tonemap,
            light_sampling: LightSampling::All,
        };
        let pixels = render(scene, &render_config)?;
        refs.push(ReferenceView {
            width: cfg.width,
            height: cfg.height,
            pixels,
            camera_pos,
            target: cfg.target,
            up,
            fov_y: cfg.fov_y,
        });
        eprintln!("ref {}/{} baked", i + 1, cameras.len());
    }
    Ok(refs)
}

/// Quasi-uniform Fibonacci-spiral camera positions on a sphere of `radius`
/// centred at `target`. Same construction as
/// `nano_core::environment::fibonacci_sphere` but exposed with explicit
/// world-space placement so the caller can sample arbitrary scene
/// neighbourhoods.
fn fibonacci_sphere_cameras(n: u32, radius: f32, target: Vec3) -> Vec<Vec3> {
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * PI / golden;
    (0..n)
        .map(|i| {
            let theta = golden_angle * i as f32;
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let dir = Vec3::new(r * theta.cos(), r * theta.sin(), z);
            target + dir * radius
        })
        .collect()
}

/// Pick a stable up vector for a look-at frame. Default world Y unless the
/// view direction is degenerately close to it (poles of the sphere), in
/// which case fall back to world X.
fn pick_up_for_camera(camera_pos: Vec3, target: Vec3) -> Vec3 {
    let forward = (target - camera_pos).normalize_or_zero();
    if forward.dot(Vec3::Y).abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_cameras_lie_on_sphere() {
        let target = Vec3::new(1.0, 2.0, -5.0);
        let radius = 10.0;
        let cams = fibonacci_sphere_cameras(64, radius, target);
        assert_eq!(cams.len(), 64);
        for c in cams {
            let dist = (c - target).length();
            assert!((dist - radius).abs() < 1e-3, "camera off sphere: {dist}");
        }
    }
}
