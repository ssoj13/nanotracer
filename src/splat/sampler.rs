//! Geometry sampling for Gaussian splat generation
//!
//! Samples surfaces from ALL directions, not just camera view.
//! Each sample becomes a Gaussian splat with SH-encoded view-dependent color.

use glam::Vec3;
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::renderer::cast_ray_with_params;
use crate::scene::Scene;
use crate::vec3::Vec3Ext;

use super::sh::{fibonacci_hemisphere, fit_sh};
use super::{Gaussian, SplatConfig, SurfaceSample};

/// Sample all surfaces in the scene uniformly
pub fn sample_scene(scene: &Scene, config: &SplatConfig) -> Vec<SurfaceSample> {
    let mut samples = Vec::new();
    let mut rng = fastrand::Rng::new();

    // Sample spheres
    for sphere in &scene.spheres {
        let area = 4.0 * PI * sphere.radius * sphere.radius;
        let n_samples = (area * config.density).ceil() as usize;

        for _ in 0..n_samples {
            let dir = uniform_sphere(&mut rng);
            let pos = sphere.center + dir * sphere.radius;

            // Offset slightly outward to avoid self-intersection
            let pos = pos + dir * 0.001;

            samples.push(SurfaceSample {
                pos,
                normal: dir,
                material_color: sphere.material.diffuse_color,
            });
        }
    }

    // Sample checkerboard plane (y = -4, x in [-10, 10], z in [-30, -10])
    let plane_y = -4.0;
    let plane_x_range = (-10.0, 10.0);
    let plane_z_range = (-30.0, -10.0);

    let plane_width = plane_x_range.1 - plane_x_range.0;
    let plane_depth = plane_z_range.1 - plane_z_range.0;
    let plane_area = plane_width * plane_depth;
    let n_plane_samples = (plane_area * config.density).ceil() as usize;

    // Grid-based sampling with jitter for plane
    let samples_per_side = (n_plane_samples as f32).sqrt().ceil() as usize;
    let dx = plane_width / samples_per_side as f32;
    let dz = plane_depth / samples_per_side as f32;

    for i in 0..samples_per_side {
        for j in 0..samples_per_side {
            let jitter_x = rng.f32() - 0.5;
            let jitter_z = rng.f32() - 0.5;

            let x = plane_x_range.0 + (i as f32 + 0.5 + jitter_x * 0.8) * dx;
            let z = plane_z_range.0 + (j as f32 + 0.5 + jitter_z * 0.8) * dz;

            // Checkerboard color
            let checker = ((0.5 * x + 1000.0) as i32 + (0.5 * z) as i32) & 1;
            let color = if checker == 1 {
                Vec3::new(0.3, 0.3, 0.3)
            } else {
                Vec3::new(0.3, 0.2, 0.1)
            };

            samples.push(SurfaceSample {
                pos: Vec3::new(x, plane_y + 0.001, z),
                normal: Vec3::Y, // Plane normal is up
                material_color: color,
            });
        }
    }

    samples
}

/// Generate uniform random point on unit sphere
fn uniform_sphere(rng: &mut fastrand::Rng) -> Vec3 {
    // Use rejection sampling for simplicity
    loop {
        let x = rng.f32() * 2.0 - 1.0;
        let y = rng.f32() * 2.0 - 1.0;
        let z = rng.f32() * 2.0 - 1.0;

        let len_sq = x * x + y * y + z * z;
        if len_sq > 0.0001 && len_sq <= 1.0 {
            let len = len_sq.sqrt();
            return Vec3::new(x / len, y / len, z / len);
        }
    }
}

/// Build tangent frame from normal vector
fn tangent_frame(normal: Vec3) -> (Vec3, Vec3, Vec3) {
    let n = normal.normalize();

    // Choose tangent that's not parallel to normal
    let tangent = if n.y.abs() < 0.9 {
        n.cross(Vec3::Y).normalize()
    } else {
        n.cross(Vec3::X).normalize()
    };

    let bitangent = n.cross(tangent);

    (tangent, bitangent, n)
}

/// Transform local direction to world space using tangent frame
fn local_to_world(local: Vec3, tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Vec3 {
    tangent * local.x + bitangent * local.y + normal * local.z
}

/// Sample incoming radiance at a surface point for SH fitting
///
/// Traces rays in multiple directions on the hemisphere above the surface
/// and computes the radiance that would be seen from each direction.
fn sample_sh_radiance(
    scene: &Scene,
    sample: &SurfaceSample,
    material: &MaterialInfo,
    config: &SplatConfig,
) -> Vec<(Vec3, Vec3)> {
    // Get hemisphere directions in local space
    let local_dirs = fibonacci_hemisphere(config.sh_samples);

    // Build tangent frame
    let (tangent, bitangent, normal) = tangent_frame(sample.normal);

    let mut radiance_samples = Vec::with_capacity(config.sh_samples);

    for local_dir in local_dirs {
        // Transform to world space (direction pointing away from surface)
        let world_outgoing = local_to_world(local_dir, tangent, bitangent, normal);

        // bevy_gaussian_splatting evaluates SH with `ray_direction` pointing FROM camera TO splat.
        // Our `world_outgoing` points from point to camera, so flip it here.
        let view_dir = (-world_outgoing).normalized();

        // Trace ray to estimate radiance seen from that view direction
        let radiance = trace_incoming(scene, sample, material, world_outgoing, config);

        radiance_samples.push((view_dir, radiance));
    }

    radiance_samples
}

/// Trace incoming radiance at surface point from given direction
fn trace_incoming(
    scene: &Scene,
    sample: &SurfaceSample,
    material: &MaterialInfo,
    outgoing_dir: Vec3,
    config: &SplatConfig,
) -> Vec3 {
    // We want the radiance that arrives at this point from direction `outgoing_dir`
    // This is like asking: if a camera is at infinity in direction `outgoing_dir`,
    // what color does it see at this point?

    // For a simple approach: compute the shading at this point for a "virtual camera"
    // positioned along `outgoing_dir`

    // The incoming ray direction (from camera to point) is -outgoing_dir
    let incoming_dir = -outgoing_dir.normalize();

    // Compute lighting similar to cast_ray but at a known point
    let mut diffuse_intensity = 0.0;
    let mut specular_intensity = 0.0;

    for light in &scene.lights {
        let light_dir = (light.position - sample.pos).normalized();

        // Shadow test
        let shadow_hit = scene.intersect(sample.pos, light_dir);
        if shadow_hit.hit
            && (shadow_hit.point - sample.pos).norm() < (light.position - sample.pos).norm()
        {
            continue;
        }

        // Diffuse
        diffuse_intensity += light_dir.dot(sample.normal).max(0.0);

        // Specular (Phong)
        let reflect_dir = reflect(-light_dir, sample.normal);
        specular_intensity += (-reflect_dir.dot(incoming_dir))
            .max(0.0)
            .powf(material.specular_exponent);
    }

    // Combine components
    let diffuse = sample.material_color * diffuse_intensity * material.albedo[0];
    let specular = Vec3::ONE * specular_intensity * material.albedo[1];

    // For reflective/refractive materials, trace secondary rays
    let mut reflect_color = Vec3::ZERO;
    let mut refract_color = Vec3::ZERO;

    if material.albedo[2] > 0.0 {
        let reflect_dir = reflect(incoming_dir, sample.normal).normalized();
        reflect_color = cast_ray_with_params(
            scene,
            sample.pos,
            reflect_dir,
            0,
            0,
            0,
            config.max_depth,
            config.reflection_depth,
            config.refraction_depth,
        );
    }

    if material.albedo[3] > 0.0 {
        let refract_dir = refract(incoming_dir, sample.normal, material.refractive_index, 1.0);
        refract_color = cast_ray_with_params(
            scene,
            sample.pos,
            refract_dir.normalized(),
            0,
            0,
            0,
            config.max_depth,
            config.reflection_depth,
            config.refraction_depth,
        );
    }

    tonemap_srgb(
        diffuse
            + specular
            + reflect_color * material.albedo[2]
            + refract_color * material.albedo[3],
    )
}

fn tonemap_srgb(color_linear: Vec3) -> Vec3 {
    // bevy_gaussian_splatting's shader reconstructs color in sRGB and then converts to linear.
    // To match that pipeline, fit SH in sRGB space here.
    let mapped = tonemap_reinhard(color_linear);
    linear_to_srgb(mapped)
}

fn tonemap_reinhard(color: Vec3) -> Vec3 {
    let c = color.max(Vec3::ZERO);
    c / (Vec3::ONE + c)
}

fn linear_to_srgb(linear: Vec3) -> Vec3 {
    fn channel(v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    Vec3::new(channel(linear.x), channel(linear.y), channel(linear.z))
}

/// Reflect direction I about normal N
fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - n * 2.0 * i.dot(n)
}

/// Refract direction through surface
fn refract(i: Vec3, n: Vec3, eta_t: f32, eta_i: f32) -> Vec3 {
    let cosi = -i.dot(n).clamp(-1.0, 1.0);
    let (n, eta_i, eta_t, cosi) = if cosi < 0.0 {
        (-n, eta_t, eta_i, -cosi)
    } else {
        (n, eta_i, eta_t, cosi)
    };

    let eta = eta_i / eta_t;
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);

    if k < 0.0 {
        // Total internal reflection -> reflect
        reflect(i, n)
    } else {
        i * eta + n * (eta * cosi - k.sqrt())
    }
}

/// Material info helper struct
#[derive(Clone, Copy)]
struct MaterialInfo {
    albedo: [f32; 4],
    specular_exponent: f32,
    refractive_index: f32,
}

/// Find material properties at a sample point
fn find_material_at_point(scene: &Scene, sample: &SurfaceSample) -> MaterialInfo {
    // Check which object this sample belongs to (by proximity)
    for sphere in &scene.spheres {
        let dist = (sample.pos - sphere.center).length();
        if (dist - sphere.radius).abs() < 0.1 {
            return MaterialInfo {
                albedo: sphere.material.albedo,
                specular_exponent: sphere.material.specular_exponent,
                refractive_index: sphere.material.refractive_index,
            };
        }
    }

    // Default: checkerboard plane
    MaterialInfo {
        albedo: [0.9, 0.1, 0.0, 0.0],
        specular_exponent: 10.0,
        refractive_index: 1.0,
    }
}

fn opacity_for_material(material: &MaterialInfo) -> f32 {
    // Glass needs to be partially transparent to look like glass in splat renderers.
    // Opaque materials can stay close to 1.0.
    if material.albedo[3] > 0.0 { 0.15 } else { 0.98 }
}

/// Estimate splat scale from sampling density
fn estimate_scale(density: f32, overlap: f32) -> f32 {
    // Base radius from area per sample
    let area_per_sample = 1.0 / density;
    let base_radius = (area_per_sample / PI).sqrt();

    base_radius * overlap
}

/// Generate all Gaussian splats from scene (parallel processing)
pub fn generate_splats(scene: &Scene, config: &SplatConfig) -> Vec<Gaussian> {
    println!("Sampling scene surfaces...");
    let samples = sample_scene(scene, config);
    println!("Generated {} surface samples", samples.len());

    // Use override scale or auto-calculate with moderate overlap to avoid holes.
    let splat_scale = config
        .scale_override
        .unwrap_or_else(|| estimate_scale(config.density, 1.5));
    println!(
        "Splat scale: {:.4} (log: {:.4})",
        splat_scale,
        splat_scale.ln()
    );

    println!(
        "Fitting SH coefficients ({} samples per splat, degree {})...",
        config.sh_samples, config.sh_degree
    );

    // Process samples in parallel
    let gaussians: Vec<Gaussian> = samples
        .par_iter()
        .enumerate()
        .map(|(i, sample)| {
            if i % 1000 == 0 {
                // Progress indication (approximate due to parallel)
                eprint!("\rProcessing splat {}/{}...", i, samples.len());
            }

            let material = find_material_at_point(scene, sample);
            let opacity = opacity_for_material(&material);

            // Sample hemisphere and fit SH
            let radiance_samples = sample_sh_radiance(scene, sample, &material, config);
            let sh = fit_sh(&radiance_samples, config.sh_degree);

            Gaussian::from_sample(sample, &sh, splat_scale, opacity)
        })
        .collect();

    eprintln!("\rGenerated {} gaussians", gaussians.len());

    gaussians
}
