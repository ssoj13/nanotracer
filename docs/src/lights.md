# Lights & IBL

`Light` is an enum in `nano-core::scene` with five variants:

```rust
pub enum Light {
    Point  { position, color, intensity },
    Rect   { center, u, v, color, intensity, two_sided },
    Sphere { center, radius, color, intensity },
    Box    { center, half_extents, rotation, color, intensity },
    Env    { intensity },
}
```

Each non-env variant carries its own emissive radiance (`color *
intensity`, premul). `Env` is image-based — the per-direction spectrum
comes from the scene's `EnvironmentMap`; `intensity` scales the
pre-convolved Lambertian SH.

## GPU layout

CPU `Light` → 64-byte `GpuLight` + parallel `light_radiance` SSBO:

```text
GpuLight (16-byte aligned, 64 B total)
├─ kind (u32)         // 0=Point 1=Rect 2=Sphere 3=Box 4=Env
├─ two_sided (u32)
├─ _pad[2]
├─ center (vec4)      // .xyz center; .w = radius (Sphere) or unused
├─ axis_u (vec4)      // Rect: u.xyz half-extent; Box: rotation quat (x,y,z,w)
└─ axis_v (vec4)      // Rect: v.xyz half-extent; Box: half_extents.xyz

light_radiance[i] (vec4)   // color · intensity, w=padding
```

Two parallel SSBOs instead of bit-packing keeps the schema readable
and future-extensible (per-light IES profiles, per-light SH probes,
etc. would be additional parallel buffers).

## Sampling

The shared GLSL helper `sample_light(idx, hit_pos, hit_normal, rand_uv)`
in `nano-shaders::HELPERS` dispatches by kind and returns a unified
`LightSample` whose `radiance` is **already pre-multiplied by the
geometric attenuation + PDF** appropriate for the variant:

```glsl
struct LightSample {
    uint  kind;
    vec3  dir;      // hit_pos → sampled point on emitter, unit
    float dist;     // for shadow_ray; ∞ for Env
    vec3  radiance; // effective — see below
    vec3  light_n;  // outward normal at sampled point
};
```

Caller does only the cosine + BRDF:

```glsl
LightSample ls = sample_light(li, hit_pos, normal, rand_uv);
if (ls.kind == LIGHT_ENV) {
    diffuse_radiance += ls.radiance;   // SH already cos-weighted
    continue;
}
visibility = shadow_ray(hit_pos, ls.dir, ls.dist - EPS);
if (visibility > 0.0) {
    cos_x = max(dot(ls.dir, normal), 0.0);
    diffuse_radiance  += ls.radiance * cos_x;
    specular_radiance += ls.radiance * ggx_specular(...);
}
```

### Per-variant conventions

| Variant | Effective radiance returned | Notes |
|---------|-----------------------------|-------|
| Point   | `L_e` (no falloff)          | Legacy tinyraytracer "unit-radiance" convention — preserves backward-compatible visuals from before the refactor. A true inverse-square point light is a follow-up if needed. |
| Rect    | `L_e · cos_y · A / r²`      | Uniform-area MC. `cos_y = (light_n · −dir)`. `two_sided` takes `abs(cos_y)`. |
| Box     | `L_e · cos_y · A / r²`      | Same as Rect but six faces weighted by area; stratified extraction from one uniform sample picks face + in-face position without re-sampling. |
| Sphere  | `L_e / pdf_solid`           | PBRT-style solid-angle MC: sample inside the cone subtending the sphere, ray-intersect to find the exact surface point. Falls back to uniform-area sampling when the receiver is *inside* the light. |
| Env     | `eval_env_irradiance(N) · intensity` | Caller adds directly to diffuse (already includes the cosine integral); no shadow ray. |

## IBL flow

A single `Light::Env { intensity }` plays the role of image-based
lighting:

- `EnvironmentMap::irradiance_sh` precomputes a degree-2 SH of the
  environment using Ramamoorthi–Hanrahan band factors `A_0 = π`,
  `A_1 = 2π/3`, `A_2 = π/4`.
- The 9 vec4 SH coefficients live in the `params.irradiance_sh` UBO
  on the GPU.
- `eval_env_irradiance(N)` evaluates the SH at the surface normal,
  divides by π once (so the result matches our non-physical
  unit-radiance direct-light convention).
- `sample_light` for `LIGHT_ENV` returns `eval_env_irradiance(N) ·
  intensity` as effective radiance.

When the scene has an environment map but no explicit `Light::Env`,
`nano-gpu::gpu_scene::build_gpu_scene*` auto-inserts a unit-intensity
one. The CPU `Scene` itself is not mutated — this is a build-time
policy that lets library callers (tests, training infrastructure)
get IBL without having to remember the explicit add.

## Why area lights instead of analytic sampling

The forward-fit splat shader and the raytracer both already do
shadow rays. Adding area-form Monte-Carlo is essentially free: one
extra random pair per sample, one extra dot product for `cos_y`.
The payoff is correct soft shadows + perceptually expected light
falloff with size, both of which the splat fitter and the trainer's
reference frames benefit from when SSIM-style perceptual losses
eventually land (see [Roadmap](./roadmap.md)).
