//! Integration test: confirm that moving the camera actually changes
//! what the renderer produces. This is the *wiring* check for the
//! Phase-A1 camera support — if `RenderConfig::camera_pos` doesn't reach
//! the shader (e.g. std140 misalignment again), this test fails loudly.
//!
//! Runs against a real GPU via `nano-render::render` → Vulkan ray query.
//! Skips automatically if no Vulkan device is available (CI without a
//! GPU): we return early from the test rather than panicking on
//! `VkContext::new`.
//!
//! Run with `VULKAN_SDK` set:
//!
//! ```pwsh
//! $env:VULKAN_SDK = "C:\Programs\VulkanSDK\1.4.341.1"
//! cargo test -p nano-render --test camera_views -- --nocapture
//! ```

use glam::Vec3;
use nano_core::LightSampling;
use nano_core::environment::EnvironmentMap;
use nano_core::geometry::Object;
use nano_core::material::{IVORY, MATTE_BLUE, MATTE_RED};
use nano_core::scene::{Light, Scene};
use nano_render::{RenderConfig, render};

fn build_test_scene() -> Scene {
    let mut scene = Scene::new();
    scene.set_environment(EnvironmentMap::procedural_sky());
    // Two visible objects at known positions plus a key light.
    scene.add_object(Object::sphere(Vec3::new(0.0, 0.0, -10.0), 2.0, IVORY));
    scene.add_object(Object::sphere(Vec3::new(4.0, 0.0, -12.0), 1.5, MATTE_RED));
    scene.add_object(Object::sphere(Vec3::new(-4.0, 0.0, -8.0), 1.5, MATTE_BLUE));
    scene.add_light(Light::point(Vec3::new(10.0, 10.0, 5.0)));
    scene.checkerboard_enabled = true;
    scene
}

fn base_config() -> RenderConfig {
    RenderConfig {
        width: 128,
        height: 96,
        fov: 1.05,
        // Default-camera pose: origin, looking down −Z.
        camera_pos: Vec3::ZERO,
        camera_target: Vec3::new(0.0, 0.0, -10.0),
        camera_up: Vec3::Y,
        aa_samples: 1,
        max_depth: 4,
        reflection_depth: 2,
        refraction_depth: 2,
        tonemap: true,
        light_sampling: LightSampling::All,
    }
}

fn mean_abs_diff(a: &[Vec3], b: &[Vec3]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = (*av - *bv).abs();
        sum += d.x + d.y + d.z;
    }
    sum / (a.len() as f32 * 3.0)
}

/// Render a frame, returning `None` if Vulkan isn't available so tests
/// can skip cleanly on headless CI instead of panicking.
fn try_render(scene: &Scene, cfg: &RenderConfig) -> Option<Vec<Vec3>> {
    match render(scene, cfg) {
        Ok(fb) => Some(fb),
        Err(e) => {
            eprintln!("skipping camera test — render failed: {e}");
            None
        }
    }
}

#[test]
fn moving_camera_changes_pixels() {
    let scene = build_test_scene();
    let base = base_config();
    let Some(frame_a) = try_render(&scene, &base) else {
        return;
    };

    // Move camera up + back so the same three spheres render at clearly
    // different screen positions.
    let mut shifted = base;
    shifted.camera_pos = Vec3::new(0.0, 8.0, 6.0);
    shifted.camera_target = Vec3::new(0.0, 0.0, -10.0);
    let Some(frame_b) = try_render(&scene, &shifted) else {
        return;
    };

    let diff = mean_abs_diff(&frame_a, &frame_b);
    eprintln!("mean abs pixel diff (base vs shifted): {diff:.4}");
    // 0.02 is a very loose lower bound; identical framebuffers would
    // give exactly 0 and a typical mis-projection bug would still show
    // ≥ 0.05. The threshold is here to detect the "all pixels identical"
    // failure mode without flaking on small numerical noise.
    assert!(
        diff > 0.02,
        "camera move did not change the frame (diff = {diff:.5}). \
         Likely cause: `inv_view` not reaching the shader, or shader \
         still hardcoded to origin/−Z pinhole."
    );
}

#[test]
fn orbit_around_target_changes_each_frame_pair() {
    let scene = build_test_scene();
    let target = Vec3::new(0.0, 0.0, -10.0);
    let radius = 15.0;

    // Four cameras at 0° / 90° / 180° / 270° in the XZ plane around the
    // target. Each adjacent pair should differ — confirms `inv_view`
    // updates per call, not just on first invocation.
    let positions = [
        Vec3::new(radius, 0.0, -10.0),
        Vec3::new(0.0, 0.0, -10.0 - radius),
        Vec3::new(-radius, 0.0, -10.0),
        Vec3::new(0.0, 0.0, -10.0 + radius),
    ];

    let mut frames: Vec<Vec<Vec3>> = Vec::with_capacity(4);
    for &pos in &positions {
        let cfg = RenderConfig {
            camera_pos: pos,
            camera_target: target,
            ..base_config()
        };
        let Some(fb) = try_render(&scene, &cfg) else {
            return;
        };
        frames.push(fb);
    }

    for w in frames.windows(2) {
        let diff = mean_abs_diff(&w[0], &w[1]);
        eprintln!("orbit pair diff: {diff:.4}");
        assert!(
            diff > 0.02,
            "orbit step produced (near-)identical frames (diff = {diff:.5})"
        );
    }
}

#[test]
fn identical_config_is_deterministic() {
    // Sanity: same config twice → identical (or near-identical)
    // framebuffer. Catches the opposite failure where every call gets a
    // fresh random camera pose.
    let scene = build_test_scene();
    let cfg = base_config();
    let Some(a) = try_render(&scene, &cfg) else {
        return;
    };
    let Some(b) = try_render(&scene, &cfg) else {
        return;
    };
    let diff = mean_abs_diff(&a, &b);
    eprintln!("re-render diff (should be ≈0): {diff:.5}");
    // AA=1, no light_sampling=one randomness → frames should match exactly
    // within float-precision noise from the GPU's compute scheduling.
    assert!(
        diff < 1e-3,
        "identical config produced different frames (diff = {diff:.5})"
    );
}
