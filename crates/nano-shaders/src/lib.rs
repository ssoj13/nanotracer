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

// 64-byte std430 light record — keep in sync with `nano_gpu::gpu_scene::GpuLight`.
// Field semantics vary by `kind`; see the Rust-side doc comment.
struct GpuLight {
    uint kind;
    uint two_sided;
    uint _lpad0;
    uint _lpad1;
    vec4 center;   // .xyz = center / position; .w = radius for Sphere
    vec4 axis_u;   // Rect: u.xyz half-extent; Box: rotation quaternion (x,y,z,w)
    vec4 axis_v;   // Rect: v.xyz half-extent; Box: half_extents.xyz
};

// `LightSample` is the unified output of `sample_light`. `radiance` is the
// *effective* radiance — already premultiplied by the geometric factor and
// PDF appropriate for the light's sampling strategy. The caller applies
// only the surface cosine and BRDF:
//
//     diffuse_contrib  = ls.radiance * max(N·L, 0)
//     specular_contrib = ls.radiance * ggx_specular(N, V, L, α, F₀)
//
// Exception: `kind == LIGHT_ENV` carries the cosine-convolved SH irradiance
// directly; callers add `ls.radiance` straight into `diffuse_radiance` and
// skip the shadow ray.
struct LightSample {
    uint  kind;
    vec3  dir;       // hit_pos → sampled point on the emitter surface, unit
    float dist;      // length(point − hit_pos); used by shadow_ray (Env: 0)
    vec3  radiance;  // effective radiance (see above)
    vec3  light_n;   // outward normal at the sampled point (Env/Point: 0)
};

const uint LIGHT_POINT  = 0u;
const uint LIGHT_RECT   = 1u;
const uint LIGHT_SPHERE = 2u;
const uint LIGHT_BOX    = 3u;
const uint LIGHT_ENV    = 4u;

const uint FLAG_CHECKER = 1u;
const float EPS = 2e-3;
const int MAX_STACK = 16;
const float PI = 3.14159265;

// Rotate a vector by a quaternion (x, y, z, w). Matches glam's storage order.
vec3 quat_rotate(vec4 q, vec3 v) {
    vec3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

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

// Multi-scattering compensation for GGX (Turquin 2018, fit after Heitz/Hill).
// Single-scatter GGX leaks energy at high roughness — light that ought to
// have bounced between microfacets simply vanishes. The boost factor is
//   1 + F̄ · (1 − Ē) / (1 − F̄ · (1 − Ē))
// where Ē is the GGX hemispherical directional albedo (fit below) and F̄ is
// the Schlick average ((20·F₀ + 1)/21). At α → 0 the boost is 1 (no missing
// energy); at α → 1 it climbs noticeably.
float ggx_msc_boost(float alpha, float f0) {
    float e_avg = clamp(1.0 - 0.5 * alpha * (0.93 + 0.21 * alpha), 0.05, 1.0);
    float f_avg = (20.0 * f0 + 1.0) / 21.0;
    float ms = f_avg * (1.0 - e_avg) / max(1.0 - f_avg * (1.0 - e_avg), 1e-4);
    return 1.0 + ms;
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
// CPU (see nano_core::environment::irradiance_sh — Ramamoorthi–Hanrahan)
// and carry the physical cosine integrals (π, 2π/3, π/4).
//
// Our direct lights are evaluated in a non-physical "unit-radiance" convention
// (`kd · color · (L·N)` — no 1/π Lambertian factor), so to add IBL on top
// without overwhelming the direct contribution we divide by π once here.
// Effectively this returns the average env colour weighted by the directional
// SH, in the same magnitude as a single per-direction unit-radiance light.
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
    return max(c, vec3(0.0)) * (1.0 / PI);
}

// Sample a scene light. Caller pre-binds `lights[]` (GpuLight SSBO) and
// `light_radiance[]` (vec4 SSBO) — both indexed by `idx`. `rand_uv` is a
// uniform sample in [0,1)² used by the area variants; ignored by Point /
// Env. The returned `LightSample.radiance` is *effective* — already
// premultiplied by geometric attenuation and PDF — so the caller does
// only `radiance * cos_x * BRDF`. See struct doc above.
//
// Conventions:
//  • Point  — non-physical "unit-radiance, no falloff" model (matches
//             tinyraytracer demo); contribution = L_e · cos_x.
//  • Rect / Box — uniform-area MC: contribution = L_e · cos_x · cos_y · A / r².
//  • Sphere — solid-angle MC (PBRT-style, falls back to area when the
//             receiver is inside the light).
//  • Env    — pre-convolved Lambertian SH; caller adds `radiance` directly
//             into diffuse and skips the shadow ray (kind check).
LightSample sample_light(uint idx, vec3 hit_pos, vec3 hit_normal, vec2 rand_uv) {
    GpuLight L = lights[idx];
    vec3 radiance_in = light_radiance[idx].rgb;
    LightSample s;
    s.kind = L.kind;
    s.dir = vec3(0.0);
    s.dist = 0.0;
    s.radiance = vec3(0.0);
    s.light_n = vec3(0.0);

    if (L.kind == LIGHT_POINT) {
        vec3 to_light = L.center.xyz - hit_pos;
        float d = length(to_light);
        s.dir = to_light / max(d, 1e-6);
        s.dist = d;
        s.light_n = -s.dir;
        // No falloff (legacy demo convention).
        s.radiance = radiance_in;
        return s;
    }

    if (L.kind == LIGHT_RECT) {
        // Sample uniformly across the rectangle: local coords in [-1, +1].
        float a = rand_uv.x * 2.0 - 1.0;
        float b = rand_uv.y * 2.0 - 1.0;
        vec3 point = L.center.xyz + a * L.axis_u.xyz + b * L.axis_v.xyz;
        vec3 normal = normalize(cross(L.axis_u.xyz, L.axis_v.xyz));
        vec3 to_light = point - hit_pos;
        float d = length(to_light);
        s.dir = to_light / max(d, 1e-6);
        s.dist = d;
        float cos_y = dot(-s.dir, normal);
        if (L.two_sided != 0u) {
            cos_y = abs(cos_y);
            normal = (cos_y > 0.0 && dot(-s.dir, normal) < 0.0) ? -normal : normal;
        } else if (cos_y <= 0.0) {
            // Back-face hit: contributes nothing.
            return s;
        }
        s.light_n = normal;
        float area = 4.0 * length(cross(L.axis_u.xyz, L.axis_v.xyz));
        s.radiance = radiance_in * (cos_y * area / max(d * d, 1e-6));
        return s;
    }

    if (L.kind == LIGHT_SPHERE) {
        vec3 to_center = L.center.xyz - hit_pos;
        float d_c = length(to_center);
        float radius = L.center.w;
        if (d_c <= radius) {
            // Receiver inside the light: degenerate solid-angle case;
            // fall back to uniform-area sampling.
            float z = 1.0 - 2.0 * rand_uv.x;
            float r = sqrt(max(0.0, 1.0 - z * z));
            float phi = 2.0 * PI * rand_uv.y;
            vec3 dir_local = vec3(r * cos(phi), r * sin(phi), z);
            vec3 point = L.center.xyz + dir_local * radius;
            vec3 normal = dir_local;
            vec3 to_light = point - hit_pos;
            float d = length(to_light);
            s.dir = to_light / max(d, 1e-6);
            s.dist = d;
            s.light_n = normal;
            float cos_y = max(dot(-s.dir, normal), 0.0);
            float area = 4.0 * PI * radius * radius;
            s.radiance = radiance_in * (cos_y * area / max(d * d, 1e-6));
            return s;
        }
        // Solid-angle sampling around the receiver: sample direction inside
        // the cone subtending the sphere, then ray-intersect to find the
        // exact surface point (PBRT §14.2.2).
        float sin2_max = (radius * radius) / (d_c * d_c);
        float cos_max = sqrt(max(0.0, 1.0 - sin2_max));
        float cos_t = (1.0 - rand_uv.x) + rand_uv.x * cos_max;
        float sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
        float phi = 2.0 * PI * rand_uv.y;
        // Build a local frame whose +Z points from hit_pos to the light center.
        vec3 wz = to_center / max(d_c, 1e-6);
        vec3 ax = abs(wz.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 wx = normalize(cross(ax, wz));
        vec3 wy = cross(wz, wx);
        vec3 dir = sin_t * cos(phi) * wx + sin_t * sin(phi) * wy + cos_t * wz;
        // Intersect ray (hit_pos, dir) with the sphere; pick the near hit.
        vec3 oc = hit_pos - L.center.xyz;
        float b = dot(oc, dir);
        float c = dot(oc, oc) - radius * radius;
        float disc = max(b * b - c, 0.0);
        float t_hit = -b - sqrt(disc);
        if (t_hit <= 0.0) {
            // Numerical fallback: the cone math guarantees a hit, but
            // float roundoff can land us on the far side.
            t_hit = max(-b + sqrt(disc), 0.0);
        }
        vec3 point = hit_pos + dir * t_hit;
        vec3 normal = normalize(point - L.center.xyz);
        s.dir = dir;
        s.dist = t_hit;
        s.light_n = normal;
        // pdf_solid = 1 / (2π · (1 − cos_max))   →  effective = L_e / pdf_solid.
        float pdf_solid = 1.0 / max(2.0 * PI * (1.0 - cos_max), 1e-8);
        s.radiance = radiance_in / pdf_solid;
        return s;
    }

    if (L.kind == LIGHT_BOX) {
        // Same area-form MC as Rect, but six faces weighted by their areas.
        vec4 q = L.axis_u;
        vec3 he = L.axis_v.xyz;
        vec3 pair_a = vec3(he.y * he.z, he.x * he.z, he.x * he.y) * 4.0;
        float total = 2.0 * (pair_a.x + pair_a.y + pair_a.z);
        float pick = rand_uv.x * total;
        float cdf1 = pair_a.x;
        float cdf2 = 2.0 * pair_a.x;
        float cdf3 = cdf2 + pair_a.y;
        float cdf4 = cdf2 + 2.0 * pair_a.y;
        float cdf5 = cdf4 + pair_a.z;
        // 6-face stratified extraction: derive an in-face u from the same
        // pick (the residual within the chosen bracket is itself uniform).
        float lo;
        float hi;
        int axis;
        float sign_;
        if (pick < cdf1) { lo = 0.0; hi = cdf1; axis = 0; sign_ =  1.0; }
        else if (pick < cdf2) { lo = cdf1; hi = cdf2; axis = 0; sign_ = -1.0; }
        else if (pick < cdf3) { lo = cdf2; hi = cdf3; axis = 1; sign_ =  1.0; }
        else if (pick < cdf4) { lo = cdf3; hi = cdf4; axis = 1; sign_ = -1.0; }
        else if (pick < cdf5) { lo = cdf4; hi = cdf5; axis = 2; sign_ =  1.0; }
        else                  { lo = cdf5; hi = total; axis = 2; sign_ = -1.0; }
        float u_l = ((pick - lo) / max(hi - lo, 1e-8)) * 2.0 - 1.0;
        float v_l = rand_uv.y * 2.0 - 1.0;
        vec3 local_point = vec3(0.0);
        vec3 local_normal = vec3(0.0);
        if (axis == 0) {
            local_point = vec3(sign_ * he.x, u_l * he.y, v_l * he.z);
            local_normal = vec3(sign_, 0.0, 0.0);
        } else if (axis == 1) {
            local_point = vec3(u_l * he.x, sign_ * he.y, v_l * he.z);
            local_normal = vec3(0.0, sign_, 0.0);
        } else {
            local_point = vec3(u_l * he.x, v_l * he.y, sign_ * he.z);
            local_normal = vec3(0.0, 0.0, sign_);
        }
        vec3 point = L.center.xyz + quat_rotate(q, local_point);
        vec3 normal = normalize(quat_rotate(q, local_normal));
        vec3 to_light = point - hit_pos;
        float d = length(to_light);
        s.dir = to_light / max(d, 1e-6);
        s.dist = d;
        s.light_n = normal;
        float cos_y = dot(-s.dir, normal);
        if (cos_y <= 0.0) { return s; }
        s.radiance = radiance_in * (cos_y * total / max(d * d, 1e-6));
        return s;
    }

    // LIGHT_ENV: cosine-convolved SH irradiance. No shadow ray; caller
    // detects this via `kind == LIGHT_ENV` and adds to diffuse directly.
    s.dir = hit_normal;
    s.dist = 1e30;
    s.light_n = -hit_normal;
    s.radiance = eval_env_irradiance(hit_normal) * radiance_in.r;
    return s;
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
