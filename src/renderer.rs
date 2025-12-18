//! Ray casting and rendering logic

use glam::Vec3;

use crate::scene::Scene;

/// Maximum recursion depth for ray tracing
pub const MAX_DEPTH: i32 = 16;
/// Maximum depth for reflection rays
pub const MAX_REFLECTION_DEPTH: i32 = 4;
/// Maximum depth for refraction rays
pub const MAX_REFRACTION_DEPTH: i32 = 10;

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
        i * eta + n * (eta * cosi - k.sqrt())
    }
}

/// Cast a ray into the scene and return the computed color
pub fn cast_ray(scene: &Scene, orig: Vec3, dir: Vec3, depth: i32) -> Vec3 {
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
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32,
) -> Vec3 {
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
    orig: Vec3,
    dir: Vec3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32,
) -> Vec3 {
    if depth > max_depth {
        return scene.sample_environment(dir);
    }

    let intersection = scene.intersect(orig, dir);

    if !intersection.hit {
        return scene.sample_environment(dir);
    }

    let point = intersection.point;
    let normal = intersection.normal;
    let material = intersection.material;

    let reflect_dir = reflect(dir, normal).normalize();
    // Offset origin along normal to avoid self-intersection
    let reflect_orig = if reflect_dir.dot(normal) < 0.0 {
        point - normal * 1e-3
    } else {
        point + normal * 1e-3
    };
    let reflect_color = if material.albedo[2] > 0.0 && reflection_depth < max_reflection_depth {
        cast_ray_with_separate_depths(
            scene,
            reflect_orig,
            reflect_dir,
            depth + 1,
            reflection_depth + 1,
            refraction_depth,
            max_depth,
            max_reflection_depth,
            max_refraction_depth,
        )
    } else {
        Vec3::ZERO
    };

    let refract_dir = refract(dir, normal, material.refractive_index, 1.0).normalize();
    // Offset origin - inside surface for refraction
    let refract_orig = if refract_dir.dot(normal) < 0.0 {
        point - normal * 1e-3
    } else {
        point + normal * 1e-3
    };
    let refract_color = if material.albedo[3] > 0.0 && refraction_depth < max_refraction_depth {
        cast_ray_with_separate_depths(
            scene,
            refract_orig,
            refract_dir,
            depth + 1,
            reflection_depth,
            refraction_depth + 1,
            max_depth,
            max_reflection_depth,
            max_refraction_depth,
        )
    } else {
        Vec3::ZERO
    };

    let mut diffuse_light_intensity = 0.0;
    let mut specular_light_intensity = 0.0;

    for light in &scene.lights {
        let light_dir = (light.position - point).normalize();

        // Shadow check - offset origin to avoid self-shadowing
        let shadow_orig = if light_dir.dot(normal) < 0.0 {
            point - normal * 1e-3
        } else {
            point + normal * 1e-3
        };
        let shadow_intersection = scene.intersect(shadow_orig, light_dir);
        if shadow_intersection.hit
            && (shadow_intersection.point - point).length() < (light.position - point).length()
        {
            continue;
        }

        diffuse_light_intensity += light_dir.dot(normal).max(0.0);
        specular_light_intensity += (-reflect(-light_dir, normal))
            .dot(dir)
            .max(0.0)
            .powf(material.specular_exponent);
    }

    material.diffuse_color * diffuse_light_intensity * material.albedo[0]
        + Vec3::ONE * specular_light_intensity * material.albedo[1]
        + reflect_color * material.albedo[2]
        + refract_color * material.albedo[3]
}
