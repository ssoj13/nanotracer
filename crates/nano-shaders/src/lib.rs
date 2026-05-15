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

// Real SH basis constants — Condon–Shortley phase, matching the
// graphdeco-inria 3DGS convention. Keep in sync with `nano_core::sh` if
// any value changes; existing 3DGS viewers depend on this exact set.
const float SH_C0 = 0.2820948;
const float SH_C1 = 0.488602;
const float SH_C2[5] = float[5](
     1.0925485, -1.0925485, 0.31539157, -1.0925485, 0.54627424
);
const float SH_C3[7] = float[7](
    -0.5900436, 2.8906114, -0.4570458, 0.37317634,
    -0.4570458, 1.4453057, -0.5900436
);

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

// Convert legacy Phong exponent n to GGX roughness α (Walter 2007).
// n = 10 → α ≈ 0.41 (rough),  n = 1500 → α ≈ 0.04 (near-mirror).
float phong_to_alpha(float n) {
    return sqrt(2.0 / max(n + 2.0, 2.0));
}

// GGX/Trowbridge–Reitz specular BRDF × cos(θ_L) — the per-light radiance
// contribution that adds directly to the outgoing radiance once you
// multiply by L_in. `f0` is the per-material Schlick base reflectance
// (= mat.albedo.y in nano-core::material's energy-conserving layout).
float ggx_specular(vec3 n, vec3 v, vec3 l, float alpha, float f0) {
    float NdotL = max(dot(n, l), 0.0);
    if (NdotL <= 0.0) return 0.0;
    vec3 h = normalize(v + l);
    float NdotV = max(dot(n, v), 1e-3);
    float NdotH = max(dot(n, h), 0.0);
    float VdotH = max(dot(v, h), 0.0);
    float a2 = alpha * alpha;
    float d_denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * d_denom * d_denom);
    float k = alpha * 0.5;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    float G = gv * gl;
    float F = f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
    return (D * F * G) / (4.0 * NdotV);
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

// Evaluate the Lambertian-convolved env irradiance at surface normal `n`.
// The 9 SH coefficients in `params.irradiance_sh` were pre-convolved on the
// CPU (see nano_core::environment::irradiance_sh — Ramamoorthi–Hanrahan).
// Returns radiance in the same units as direct-light contributions, ready
// to multiply by `kd * diffuse_color` and add to the lit term.
vec3 eval_env_irradiance(vec3 n) {
    float x = n.x;
    float y = n.y;
    float z = n.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    vec3 c =
          params.irradiance_sh[0].rgb *  SH_C0
        + params.irradiance_sh[1].rgb * (-SH_C1 * y)
        + params.irradiance_sh[2].rgb * ( SH_C1 * z)
        + params.irradiance_sh[3].rgb * (-SH_C1 * x)
        + params.irradiance_sh[4].rgb * (SH_C2[0] * x * y)
        + params.irradiance_sh[5].rgb * (SH_C2[1] * y * z)
        + params.irradiance_sh[6].rgb * (SH_C2[2] * (2.0 * zz - xx - yy))
        + params.irradiance_sh[7].rgb * (SH_C2[3] * x * z)
        + params.irradiance_sh[8].rgb * (SH_C2[4] * (xx - yy));
    return max(c, vec3(0.0));
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
