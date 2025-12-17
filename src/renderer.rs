//! Ray casting and rendering logic

use crate::scene::Scene;
use crate::vec3::{Vec3Ext, Vector3};

/// Maximum recursion depth for ray tracing
pub const MAX_DEPTH: i32 = 16;
/// Maximum depth for reflection rays
pub const MAX_REFLECTION_DEPTH: i32 = 4;
/// Maximum depth for refraction rays
pub const MAX_REFRACTION_DEPTH: i32 = 10;

/// Reflect a ray direction I about normal N
pub fn reflect(i: Vector3, n: Vector3) -> Vector3 {
    i - n * 2.0 * i.dot(n)
}

/// Refract a ray direction I through surface with normal N
/// eta_t: refractive index of transmitted medium
/// eta_i: refractive index of incident medium (default: 1.0 for air)
pub fn refract(i: Vector3, n: Vector3, eta_t: f32, eta_i: f32) -> Vector3 {
    let cosi = -((-1.0_f32).max((1.0_f32).min(i.dot(n))));
    if cosi < 0.0 {
        // Ray comes from inside the object, swap media
        return refract(i, -n, eta_i, eta_t);
    }

    let eta = eta_i / eta_t;
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);

    if k < 0.0 {
        // Total internal reflection, return a special value
        return Vector3::new(1.0, 0.0, 0.0);
    } else {
        return i * eta + n * (eta * cosi - k.sqrt());
    }
}

/// Cast a ray into the scene and return the computed color
pub fn cast_ray(scene: &Scene, orig: Vector3, dir: Vector3, depth: i32) -> Vector3 {
    cast_ray_with_separate_depths(
        scene,
        orig,
        dir,
        depth,
        0,
        0,
        MAX_DEPTH,
        MAX_REFLECTION_DEPTH,
        MAX_REFRACTION_DEPTH,
    )
}

/// Cast a ray with configurable depth parameters
pub fn cast_ray_with_params(
    scene: &Scene,
    orig: Vector3,
    dir: Vector3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32,
) -> Vector3 {
    cast_ray_with_separate_depths(
        scene,
        orig,
        dir,
        depth,
        reflection_depth,
        refraction_depth,
        max_depth,
        max_reflection_depth,
        max_refraction_depth,
    )
}

/// Cast a ray with separate tracking for reflection and refraction depths
fn cast_ray_with_separate_depths(
    scene: &Scene,
    orig: Vector3,
    dir: Vector3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32,
) -> Vector3 {
    // Safety check to prevent infinite recursion
    if depth > max_depth {
        return scene.sample_environment(dir); // Sample environment or default sky
    }

    let intersection = scene.intersect(orig, dir);

    // If we missed, return environment color
    if !intersection.hit {
        return scene.sample_environment(dir); // Sample environment or default sky
    }

    let point = intersection.point;
    let normal = intersection.normal;
    let material = intersection.material;

    let reflect_dir = reflect(dir, normal).normalized();
    let reflect_color = if material.albedo[2] > 0.0 && reflection_depth < max_reflection_depth {
        let color = cast_ray_with_separate_depths(
            scene,
            point,
            reflect_dir,
            depth + 1,
            reflection_depth + 1,
            refraction_depth,
            max_depth,
            max_reflection_depth,
            max_refraction_depth,
        );
        color
    } else {
        Vector3::ZERO
    };

    let refract_dir = refract(dir, normal, material.refractive_index, 1.0).normalized();
    let refract_color = if material.albedo[3] > 0.0 && refraction_depth < max_refraction_depth {
        cast_ray_with_separate_depths(
            scene,
            point,
            refract_dir,
            depth + 1,
            reflection_depth,
            refraction_depth + 1,
            max_depth,
            max_reflection_depth,
            max_refraction_depth,
        )
    } else {
        Vector3::ZERO
    };

    let mut diffuse_light_intensity = 0.0;
    let mut specular_light_intensity = 0.0;

    // Calculate lighting from all lights
    for light in &scene.lights {
        let light_dir = (light.position - point).normalized();

        // Check if point is in shadow
        let shadow_intersection = scene.intersect(point, light_dir);
        if shadow_intersection.hit
            && (shadow_intersection.point - point).norm() < (light.position - point).norm()
        {
            continue;
        }

        diffuse_light_intensity += 0.0_f32.max(light_dir.dot(normal));
        specular_light_intensity += 0.0_f32
            .max((-reflect(-light_dir, normal)).dot(dir))
            .powf(material.specular_exponent);
    }

    material.diffuse_color * diffuse_light_intensity * material.albedo[0]
        + Vector3::ONE * specular_light_intensity * material.albedo[1]
        + reflect_color * material.albedo[2]
        + refract_color * material.albedo[3]
}
