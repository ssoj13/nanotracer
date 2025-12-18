//! Geometry sampling for Gaussian splat generation

use glam::Vec3;
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::geometry::{Geometry, Object};
use crate::material::{checkerboard_material, Material};
use crate::mesh::Mesh;
use crate::renderer::{cast_ray_with_params, reflect, refract};
use crate::scene::Scene;

use super::sh::{fibonacci_hemisphere, fit_sh};
use super::{Gaussian, SplatConfig, SurfaceSample};

/// Fibonacci sphere direction sampling
fn fibonacci_sphere_dir(i: usize, n: usize, theta_offset: f32) -> Vec3 {
    debug_assert!(n > 0);
    let n_f = n as f32;
    let i_f = i as f32;

    let z = 1.0 - 2.0 * (i_f + 0.5) / n_f;
    let r = (1.0 - z * z).max(0.0).sqrt();

    let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
    let theta = theta_offset + golden_angle * i_f;

    Vec3::new(r * theta.cos(), r * theta.sin(), z)
}

/// Sample all surfaces in the scene uniformly
pub fn sample_scene(scene: &Scene, config: &SplatConfig) -> Vec<SurfaceSample> {
    let mut samples = Vec::new();
    let mut rng = fastrand::Rng::new();

    // Sample all objects
    for object in &scene.objects {
        sample_object(object, config.density, &mut rng, &mut samples);
    }

    // Sample checkerboard plane if enabled
    if scene.checkerboard_enabled {
        sample_checkerboard(config.density, &mut rng, &mut samples);
    }

    samples
}

/// Sample an object based on its geometry type
fn sample_object(
    object: &Object,
    density: f32,
    rng: &mut fastrand::Rng,
    samples: &mut Vec<SurfaceSample>,
) {
    match &object.geometry {
        Geometry::Sphere { center, radius } => {
            sample_sphere(*center, *radius, object.material, density, rng, samples);
        }
        Geometry::Mesh(mesh) => {
            sample_mesh(mesh, object.material, density, rng, samples);
        }
    }
}

/// Sample a sphere surface
fn sample_sphere(
    center: Vec3,
    radius: f32,
    material: Material,
    density: f32,
    rng: &mut fastrand::Rng,
    samples: &mut Vec<SurfaceSample>,
) {
    let area = 4.0 * PI * radius * radius;
    let n_samples = (area * density).ceil() as usize;
    let theta_offset = rng.f32() * 2.0 * PI;

    for i in 0..n_samples {
        let dir = fibonacci_sphere_dir(i, n_samples.max(1), theta_offset);
        let pos = center + dir * radius;
        let pos = pos + dir * 0.001; // offset to avoid self-intersection

        samples.push(SurfaceSample {
            pos,
            normal: dir,
            material,
        });
    }
}

/// Sample a mesh surface using area-weighted triangle selection
fn sample_mesh(
    mesh: &Mesh,
    material: Material,
    density: f32,
    rng: &mut fastrand::Rng,
    samples: &mut Vec<SurfaceSample>,
) {
    if mesh.indices.is_empty() {
        return;
    }

    // Compute triangle areas and build CDF
    let mut areas: Vec<f32> = Vec::with_capacity(mesh.indices.len());
    let mut total_area = 0.0;

    for tri in &mesh.indices {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let area = (v1 - v0).cross(v2 - v0).length() * 0.5;
        total_area += area;
        areas.push(total_area);
    }

    if total_area < 1e-8 {
        return;
    }

    let n_samples = (total_area * density).ceil() as usize;

    for _ in 0..n_samples {
        // Select triangle by area (CDF sampling)
        let r = rng.f32() * total_area;
        let tri_idx = areas.partition_point(|&a| a < r).min(areas.len() - 1);

        let tri = &mesh.indices[tri_idx];
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        // Uniform barycentric sampling
        let mut u = rng.f32();
        let mut v = rng.f32();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;

        let pos = v0 * w + v1 * u + v2 * v;

        // Interpolate normal if available
        let normal = if mesh.normals.len() == mesh.vertices.len() {
            let n0 = mesh.normals[tri[0] as usize];
            let n1 = mesh.normals[tri[1] as usize];
            let n2 = mesh.normals[tri[2] as usize];
            (n0 * w + n1 * u + n2 * v).normalize()
        } else {
            // Face normal
            (v1 - v0).cross(v2 - v0).normalize()
        };

        // Offset slightly along normal
        let pos = pos + normal * 0.001;

        samples.push(SurfaceSample {
            pos,
            normal,
            material,
        });
    }
}

/// Sample checkerboard plane (y = -4, x in [-10, 10], z in [-30, -10])
fn sample_checkerboard(density: f32, rng: &mut fastrand::Rng, samples: &mut Vec<SurfaceSample>) {
    let plane_y = -4.0;
    let plane_x_range = (-10.0, 10.0);
    let plane_z_range = (-30.0, -10.0);

    let plane_width = plane_x_range.1 - plane_x_range.0;
    let plane_depth = plane_z_range.1 - plane_z_range.0;
    let plane_area = plane_width * plane_depth;
    let n_plane_samples = (plane_area * density).ceil() as usize;

    let samples_per_side = (n_plane_samples as f32).sqrt().ceil() as usize;
    let dx = plane_width / samples_per_side as f32;
    let dz = plane_depth / samples_per_side as f32;

    for i in 0..samples_per_side {
        for j in 0..samples_per_side {
            let jitter_x = rng.f32() - 0.5;
            let jitter_z = rng.f32() - 0.5;

            let x = plane_x_range.0 + (i as f32 + 0.5 + jitter_x * 0.8) * dx;
            let z = plane_z_range.0 + (j as f32 + 0.5 + jitter_z * 0.8) * dz;

            let checker = ((0.5 * x + 1000.0) as i32 + (0.5 * z) as i32) & 1;
            let color = if checker == 1 {
                Vec3::new(0.3, 0.3, 0.3)
            } else {
                Vec3::new(0.3, 0.2, 0.1)
            };

            samples.push(SurfaceSample {
                pos: Vec3::new(x, plane_y + 0.001, z),
                normal: Vec3::Y,
                material: checkerboard_material(color),
            });
        }
    }
}

/// Build tangent frame from normal vector
fn tangent_frame(normal: Vec3) -> (Vec3, Vec3, Vec3) {
    let n = normal.normalize();

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
fn sample_sh_radiance(
    scene: &Scene,
    sample: &SurfaceSample,
    config: &SplatConfig,
) -> Vec<(Vec3, Vec3)> {
    let local_dirs = fibonacci_hemisphere(config.sh_samples);
    let (tangent, bitangent, normal) = tangent_frame(sample.normal);

    let mut radiance_samples = Vec::with_capacity(config.sh_samples);

    for local_dir in local_dirs {
        let world_outgoing = local_to_world(local_dir, tangent, bitangent, normal);
        let view_dir = -world_outgoing.normalize();

        let radiance = trace_incoming(scene, sample, world_outgoing, config);

        radiance_samples.push((view_dir, radiance));
    }

    radiance_samples
}

/// Trace incoming radiance at surface point from given direction
fn trace_incoming(
    scene: &Scene,
    sample: &SurfaceSample,
    outgoing_dir: Vec3,
    config: &SplatConfig,
) -> Vec3 {
    let incoming_dir = -outgoing_dir.normalize();
    let material = &sample.material;

    let mut diffuse_intensity = 0.0;
    let mut specular_intensity = 0.0;

    for light in &scene.lights {
        let light_dir = (light.position - sample.pos).normalize();

        // Shadow test - offset to avoid self-intersection
        let shadow_orig = if light_dir.dot(sample.normal) < 0.0 {
            sample.pos - sample.normal * 1e-3
        } else {
            sample.pos + sample.normal * 1e-3
        };
        let shadow_hit = scene.intersect(shadow_orig, light_dir);
        if shadow_hit.hit
            && (shadow_hit.point - sample.pos).length() < (light.position - sample.pos).length()
        {
            continue;
        }

        diffuse_intensity += light_dir.dot(sample.normal).max(0.0);

        let reflect_dir = reflect(-light_dir, sample.normal);
        specular_intensity += (-reflect_dir.dot(incoming_dir))
            .max(0.0)
            .powf(material.specular_exponent);
    }

    let diffuse = material.diffuse_color * diffuse_intensity * material.albedo[0];
    let specular = Vec3::ONE * specular_intensity * material.albedo[1];

    let mut reflect_color = Vec3::ZERO;
    let mut refract_color = Vec3::ZERO;

    if material.albedo[2] > 0.0 {
        let reflect_dir = reflect(incoming_dir, sample.normal).normalize();
        let reflect_orig = if reflect_dir.dot(sample.normal) < 0.0 {
            sample.pos - sample.normal * 1e-3
        } else {
            sample.pos + sample.normal * 1e-3
        };
        reflect_color = cast_ray_with_params(
            scene,
            reflect_orig,
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
        let refract_orig = if refract_dir.dot(sample.normal) < 0.0 {
            sample.pos - sample.normal * 1e-3
        } else {
            sample.pos + sample.normal * 1e-3
        };
        refract_color = cast_ray_with_params(
            scene,
            refract_orig,
            refract_dir.normalize(),
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

fn opacity_for_material(material: &Material) -> f32 {
    if material.albedo[3] > 0.0 {
        0.15
    } else {
        0.98
    }
}

/// Estimate splat scale from sampling density
fn estimate_scale(density: f32, overlap: f32) -> f32 {
    let area_per_sample = 1.0 / density;
    let base_radius = (area_per_sample / PI).sqrt();
    base_radius * overlap
}

/// Generate all Gaussian splats from scene (parallel processing)
pub fn generate_splats(scene: &Scene, config: &SplatConfig) -> Vec<Gaussian> {
    println!("Sampling scene surfaces...");
    let samples = sample_scene(scene, config);
    println!("Generated {} surface samples", samples.len());

    let splat_scale = config
        .scale_override
        .unwrap_or_else(|| estimate_scale(config.density, 2.0));
    println!(
        "Splat scale: {:.4} (log: {:.4})",
        splat_scale,
        splat_scale.ln()
    );

    println!(
        "Fitting SH coefficients ({} samples per splat, degree {})...",
        config.sh_samples, config.sh_degree
    );

    let gaussians: Vec<Gaussian> = samples
        .par_iter()
        .enumerate()
        .map(|(i, sample)| {
            if i % 1000 == 0 {
                eprint!("\rProcessing splat {}/{}...", i, samples.len());
            }

            let opacity = opacity_for_material(&sample.material);

            let radiance_samples = sample_sh_radiance(scene, sample, config);
            let sh = fit_sh(&radiance_samples, config.sh_degree);

            Gaussian::from_sample(sample, &sh, splat_scale, opacity)
        })
        .collect();

    eprintln!("\rGenerated {} gaussians", gaussians.len());

    gaussians
}
