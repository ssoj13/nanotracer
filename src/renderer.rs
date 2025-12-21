//! Ray casting and rendering logic

use fastrand;
use glam::Vec3;

use crate::material::Material;
use crate::scene::Scene;

/// Maximum recursion depth for ray tracing
pub const MAX_DEPTH: i32 = 16;
/// Maximum depth for reflection rays
pub const MAX_REFLECTION_DEPTH: i32 = 4;
/// Maximum depth for refraction rays
pub const MAX_REFRACTION_DEPTH: i32 = 10;

/// Ray tracing configuration (reduces argument count)
#[derive(Debug, Clone, Copy)]
pub struct RayConfig {
    pub max_depth: i32,
    pub max_reflection: i32,
    pub max_refraction: i32,
}

impl Default for RayConfig {
    fn default() -> Self {
        Self {
            max_depth: MAX_DEPTH,
            max_reflection: MAX_REFLECTION_DEPTH,
            max_refraction: MAX_REFRACTION_DEPTH,
        }
    }
}

impl RayConfig {
    pub fn new(max_depth: i32, max_reflection: i32, max_refraction: i32) -> Self {
        Self { max_depth, max_reflection, max_refraction }
    }
}

/// Offset ray origin to avoid self-intersection
#[inline]
pub fn offset_origin(point: Vec3, normal: Vec3, dir: Vec3) -> Vec3 {
    if dir.dot(normal) < 0.0 {
        point - normal * 1e-3
    } else {
        point + normal * 1e-3
    }
}

/// Reflect a ray direction I about normal N
pub fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - n * 2.0 * i.dot(n)
}

/// Refract a ray direction I through surface with normal N
pub fn refract(i: Vec3, n: Vec3, eta_t: f32, eta_i: f32) -> Vec3 {
    let cosi = -i.dot(n).clamp(-1.0, 1.0);
    if cosi < 0.0 {
        return refract(i, -n, eta_i, eta_t);
    }

    let eta = eta_i / eta_t;
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);

    if k < 0.0 {
        reflect(i, n) // Total internal reflection
    } else {
        (i * eta + n * (eta * cosi - k.sqrt())).normalize_or_zero()
    }
}

/// Cast a ray into the scene and return the computed color
pub fn cast_ray(scene: &Scene, orig: Vec3, dir: Vec3, depth: i32) -> Vec3 {
    cast_ray_cfg(scene, orig, dir, depth, 0, 0, &RayConfig::default())
}

/// Cast a ray with RayConfig (preferred API)
#[inline]
pub fn cast_ray_cfg(
    scene: &Scene,
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    refl_depth: i32,
    refr_depth: i32,
    cfg: &RayConfig,
) -> Vec3 {
    cast_ray_inner(scene, orig, dir, depth, refl_depth, refr_depth, cfg)
}

/// Cast a ray with configurable depth parameters (legacy API)
#[allow(clippy::too_many_arguments)]
pub fn cast_ray_with_params(
    scene: &Scene,
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32,
) -> Vec3 {
    let cfg = RayConfig::new(max_depth, max_reflection_depth, max_refraction_depth);
    cast_ray_inner(scene, orig, dir, depth, reflection_depth, refraction_depth, &cfg)
}

/// Core ray casting implementation
#[inline]
fn cast_ray_inner(
    scene: &Scene,
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    refl_depth: i32,
    refr_depth: i32,
    cfg: &RayConfig,
) -> Vec3 {
    let intersection = scene.intersect(orig, dir);

    if !intersection.hit {
        return scene.sample_environment(dir);
    }

    let material = intersection.material;
    let is_diffuse_only = material.albedo[1] <= 0.0 
        && material.albedo[2] <= 0.0 
        && material.albedo[3] <= 0.0;

    // Fast path for diffuse-only materials (skip on primary rays)
    if is_diffuse_only && depth > 0 {
        return cast_ray_diffuse(scene, orig, dir, depth, cfg);
    }

    if depth > cfg.max_depth {
        return scene.sample_environment(dir);
    }

    let point = intersection.point;
    let normal = intersection.normal;

    let rr_weight = match russian_roulette(depth, &material) {
        Some(w) => w,
        None => return Vec3::ZERO,
    };

    let mut reflect_color = Vec3::ZERO;
    let mut refract_color = Vec3::ZERO;

    if !is_diffuse_only {
        // Reflection
        if material.albedo[2] > 0.0 && refl_depth < cfg.max_reflection {
            let refl_dir = reflect(dir, normal);
            let refl_orig = offset_origin(point, normal, refl_dir);
            reflect_color = cast_ray_inner(
                scene, refl_orig, refl_dir,
                depth + 1, refl_depth + 1, refr_depth, cfg,
            );
        }

        // Refraction
        if material.albedo[3] > 0.0 && refr_depth < cfg.max_refraction {
            let refr_dir = refract(dir, normal, material.refractive_index, 1.0);
            let refr_orig = offset_origin(point, normal, refr_dir);
            refract_color = cast_ray_inner(
                scene, refr_orig, refr_dir,
                depth + 1, refl_depth, refr_depth + 1, cfg,
            );
        }
    }

    // Lighting
    let (diffuse_intensity, specular_intensity) = 
        compute_lighting(scene, point, normal, dir, &material, is_diffuse_only);

    // Final color composition
    let mut color = material.diffuse_color * diffuse_intensity * material.albedo[0];
    color += Vec3::ONE * specular_intensity * material.albedo[1];
    color += reflect_color * material.albedo[2];
    color += refract_color * material.albedo[3];

    color * rr_weight
}

/// Compute direct lighting at a point
#[inline]
fn compute_lighting(
    scene: &Scene,
    point: Vec3,
    normal: Vec3,
    view_dir: Vec3,
    material: &Material,
    diffuse_only: bool,
) -> (f32, f32) {
    let light_count = scene.lights.len();
    if light_count == 0 {
        return (0.0, 0.0);
    }

    let light_weight = light_count as f32;
    let light_idx = fastrand::usize(..light_count);
    let light = &scene.lights[light_idx];
    let light_dir = (light.position - point).normalize();

    // Shadow test
    let shadow_orig = offset_origin(point, normal, light_dir);
    let shadow_hit = scene.intersect(shadow_orig, light_dir);
    
    if shadow_hit.hit 
        && (shadow_hit.point - point).length() < (light.position - point).length() 
    {
        return (0.0, 0.0);
    }

    let diffuse = light_dir.dot(normal).max(0.0) * light_weight;
    
    let specular = if diffuse_only {
        0.0
    } else {
        (-reflect(-light_dir, normal))
            .dot(view_dir)
            .max(0.0)
            .powf(material.specular_exponent)
            * light_weight
    };

    (diffuse, specular)
}

fn russian_roulette(depth: i32, material: &Material) -> Option<f32> {
    if depth <= 4 {
        return Some(1.0);
    }

    // Fast computation of energy using cached values where possible
    let energy = material.albedo[0] + material.albedo[1] + material.albedo[2] + material.albedo[3] +
                 material.diffuse_color.max_element();
    let survival = energy.clamp(0.05, 1.0);

    if survival >= 1.0 {
        return Some(1.0);
    }

    // Use a faster random number generator
    if fastrand::f32() < survival {
        Some(1.0 / survival)
    } else {
        None
    }
}

/// Fast ray casting for diffuse-only materials (no reflection/refraction)
#[inline]
fn cast_ray_diffuse(
    scene: &Scene,
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    cfg: &RayConfig,
) -> Vec3 {
    if depth > cfg.max_depth {
        return scene.sample_environment(dir);
    }

    let intersection = scene.intersect(orig, dir);
    if !intersection.hit {
        return scene.sample_environment(dir);
    }

    let point = intersection.point;
    let normal = intersection.normal;
    let material = intersection.material;

    let rr_weight = match russian_roulette(depth, &material) {
        Some(w) => w,
        None => return Vec3::ZERO,
    };

    let (diffuse_intensity, _) = compute_lighting(scene, point, normal, dir, &material, true);
    material.diffuse_color * diffuse_intensity * material.albedo[0] * rr_weight
}
