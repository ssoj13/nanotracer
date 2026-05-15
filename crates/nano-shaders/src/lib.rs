//! Shared GLSL string constants for the `nano-render` and `nano-splat`
//! compute shaders.
//!
//! Both shaders bind the same Vulkan resources (TLAS, vertex / normal /
//! triangle SSBOs, materials, lights, environment) and need the same
//! helpers (`reflect_dir`, `trace_ray`, `shadow_ray`, …). This crate
//! exposes those helpers as `&'static str` chunks that the per-shader
//! builders concatenate together before handing off to shaderc.
//!
//! The split is `PREAMBLE` (no global dependencies — sits at the top) +
//! `HELPERS` (needs `topLevelAS`, `env_map`, `params` declared first — sits
//! after the per-shader bindings).
//!
//! ```text
//!   <PREAMBLE>
//!   <shader-specific layout + bindings + Params block>
//!   <HELPERS>
//!   <shader-specific main()>
//! ```

/// GLSL fragment placed *before* per-shader bindings.
///
/// Contains: version + ray-query extension, the `Material` struct, shared
/// constants, and the pure helpers that do not touch any binding.
pub const PREAMBLE: &str = r#"#version 460
#extension GL_EXT_ray_query : require

struct Material {
    vec3 diffuse;
    float specular_exponent;
    vec4 albedo;
    float refractive_index;
    uint flags;
    uint _pad0;
    uint _pad1;
};

const uint FLAG_CHECKER = 1u;
const float EPS = 2e-3;
const int MAX_STACK = 16;
const float PI = 3.14159265;

uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float rand01(uint seed) {
    return float(wang_hash(seed)) / 4294967296.0;
}

float max_component(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

vec3 reflect_dir(vec3 i, vec3 n) {
    return i - 2.0 * dot(i, n) * n;
}

vec3 refract_dir(vec3 i, vec3 n, float eta_t, float eta_i) {
    float cosi = clamp(-dot(i, n), -1.0, 1.0);
    float eta_i_local = eta_i;
    float eta_t_local = eta_t;
    vec3 n_local = n;
    if (cosi < 0.0) {
        cosi = -cosi;
        n_local = -n_local;
        float tmp = eta_i_local;
        eta_i_local = eta_t_local;
        eta_t_local = tmp;
    }
    float eta = eta_i_local / eta_t_local;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (k < 0.0) {
        return reflect_dir(i, n_local);
    }
    return normalize(i * eta + n_local * (eta * cosi - sqrt(k)));
}

vec3 offset_origin(vec3 point, vec3 normal, vec3 dir) {
    if (dot(dir, normal) < 0.0) {
        return point - normal * EPS;
    }
    return point + normal * EPS;
}

vec3 checker_color(vec3 pos) {
    int checker = (int(0.5 * pos.x + 1000.0) + int(0.5 * pos.z)) & 1;
    if (checker != 0) {
        return vec3(0.3, 0.3, 0.3);
    }
    return vec3(0.3, 0.2, 0.1);
}

vec3 tonemap_reinhard(vec3 c) {
    vec3 v = max(c, vec3(0.0));
    return v / (vec3(1.0) + v);
}

vec3 linear_to_srgb(vec3 linear) {
    vec3 v = clamp(linear, vec3(0.0), vec3(1.0));
    vec3 low = v * 12.92;
    vec3 high = 1.055 * pow(v, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 cutoff = vec3(0.0031308);
    return mix(high, low, lessThanEqual(v, cutoff));
}
"#;

/// GLSL fragment placed *after* per-shader bindings.
///
/// Functions here reference globals that must already be declared in the
/// per-shader text: the acceleration structure `topLevelAS`, the
/// `params` uniform block (must expose `use_sky`, `exposure`), and the
/// `env_map` combined image sampler.
pub const HELPERS: &str = r#"
vec3 sample_environment(vec3 dir) {
    if (params.use_sky != 0u) {
        vec3 sky_blue = vec3(0.5, 0.7, 1.0);
        vec3 horizon  = vec3(1.0, 0.9, 0.7);
        float t = (normalize(dir).y + 1.0) * 0.5;
        return sky_blue * t + horizon * (1.0 - t);
    }

    vec3 n = normalize(dir);
    float phi   = atan(n.z, n.x);
    float theta = acos(clamp(-n.y, -1.0, 1.0));
    float u = fract(phi / (2.0 * PI) + 0.5);
    float v = clamp(theta / PI, 0.0, 1.0);
    return texture(env_map, vec2(u, v)).rgb * params.exposure;
}

bool trace_ray(vec3 origin, vec3 dir, float t_max,
               out uint prim_id, out vec2 bary, out float t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return false;
    }
    prim_id = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    bary    = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    t       = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

float shadow_ray(vec3 origin, vec3 dir, float dist) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001, dir, dist);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return 1.0;
    }
    return 0.0;
}
"#;

/// Convenience: concatenate `PREAMBLE`, the per-shader middle, and `HELPERS`
/// + the per-shader main into a single owned `String` ready for shaderc.
pub fn assemble(bindings: &str, body: &str) -> String {
    let mut out = String::with_capacity(PREAMBLE.len() + bindings.len() + HELPERS.len() + body.len());
    out.push_str(PREAMBLE);
    out.push_str(bindings);
    out.push_str(HELPERS);
    out.push_str(body);
    out
}
