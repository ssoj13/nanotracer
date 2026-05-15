# AGENTS.md — nanotracer-rs dataflow & codepath map

Auto-maintained map of the codebase that a future agent can read instead
of re-scanning every file. ASCII diagrams here; mermaid versions live in
`DIAGRAMS.md`.

Workspace layout is the 7-crate structure introduced 2026-05 (see
`plan1.md`, `README.md` "Workspace layout"). The historical single-crate
`src/`-shaped path map at the bottom of this file is preserved only for
agents triaging older commits.

---

## 1. Top-level dataflow (live path only)

```
                    ┌────────────────────────────────────────────┐
                    │ src/main.rs                                │
                    │  - clap CLI (Args)                         │
                    │  - gpu-mem startup line                    │
                    │  - builds Scene { objects, lights, env }   │
                    │  - branches on --splats flag               │
                    └────────────────┬───────────────────────────┘
                                     │
                          ┌──────────┴───────────┐
                          │                      │
                ┌─────────▼────────┐    ┌────────▼─────────────┐
                │ nano-render      │    │ nano-splat           │
                │  ::render(scene, │    │  ::generate_splats_  │
                │   cfg) -> Vec    │    │   gpu(scene, cfg)    │
                │   <Vec3>         │    │   -> Vec<Gaussian>   │
                └─────────┬────────┘    └────────┬─────────────┘
                          │                      │
                 ┌────────▼────────┐    ┌────────▼─────────────┐
                 │ nano-io::utils  │    │ nano-splat::ply::    │
                 │  ::save_image   │    │  write_ply           │
                 │  (PNG)          │    │  (binary 3DGS PLY)   │
                 └─────────────────┘    └──────────────────────┘
```

Both render paths use `nano-gpu::gpu_scene::build_gpu_scene*` for the
Scene → GPU buffer marshalling and `nano-gpu::vk_runtime::VkContext` for
Vulkan setup. Both shaders share `nano-shaders::{PREAMBLE, HELPERS}`
chunks concatenated through `assemble(BINDINGS, BODY)`.

---

## 2. Scene → GPU buffer marshalling

```
nano-core::scene::Scene
        │
        │ for each Object:
        │   ┌─ Geometry::Sphere ─► sphere_mesh (UV sphere)
        │   └─ Geometry::Mesh   ─► append_mesh
        │
        │ checkerboard_enabled ─► checkerboard_plane_mesh
        ▼
nano-gpu::gpu_scene::build_gpu_scene_with_detail_boost
        │
        ├─► vertices [vec4], normals [vec4]
        ├─► triangles [uvec4]
        ├─► tri_materials [u32]
        ├─► tri_cdf [f32]   (area×detail-boost weight, normalised)
        ├─► tri_areas [f32]
        ├─► materials [GpuMaterial]
        └─► lights [vec4]
              │
              ▼
        nano-gpu::vk_runtime  (SSBO/UBO/AS uploads)
```

`detail_boost` (`gpu_scene.rs`) re-weights triangles by
`1 - |face_normal · vertex_normal|`, so curvy regions of a smoothed
mesh receive extra splat density. Sampling-only heuristic — no effect
on the image renderer.

---

## 3. nano-render image renderer

```
                ┌──────────────┐
                │ Vulkan TLAS  │ ◄── vk_runtime::build_acceleration_structures
                └─────┬────────┘
                      │
                shader  = nano_shaders::assemble(
                            RENDERER_BINDINGS,
                            RENDERER_BODY)
                      │
                      ▼
                compute shader (8×8 threads):
                  per pixel:
                    for s in 0..aa²:
                      halton-jittered ray from pinhole
                      stack-based path trace (MAX_STACK = 16):
                        trace_ray (in HELPERS)
                        shade with all/one lights
                        push reflection / refraction with weight
                        Russian roulette after depth > 3
                    average → outImage[gid] (rgba32f, linear)
                      │
                      ▼
                copy_image_to_buffer
                      │
                      ▼
        Vec<Vec3>  ─►  nano-io::utils::save_image
                       (tonemap + sRGB + clamp on CPU)
```

---

## 4. nano-splat splat generator

```
   Scene + SplatConfigGpu
            │
            ▼
   build_gpu_scene_with_detail_boost
            │
            ▼
   sample_count = ceil(total_area × density)
            │
            ▼
   shader = nano_shaders::assemble(SPLAT_BINDINGS, SPLAT_BODY)
            │
            ▼
   compute shader (64-thread workgroup):
   id < sample_count:
     1. pick triangle by binary search on tri_cdf  (find_triangle)
     2. uniform barycentric sample (sqrt r1 / r2)
     3. interpolate pos, normal, fetch material
     4. build tangent frame around normal
     5. for s in 0..(sh_samples + glossy_extra):
          local_dir from hemisphere sampler
          view_dir  = tangent·x + bitangent·y + normal·z
          radiance  = shade_surface(pos, normal, view_dir, mat)
            shade_surface:
              ∑ all lights (Phong: diffuse + spec via reflect(-L,N)·V)
              + trace_path(reflect(d,N))*albedo.z
              + trace_path(refract(d,N))*albedo.w
          radiance → luma clamp → tonemap? → linear→sRGB → (− 0.5)
          accumulate into ATA / ATB for 16 SH basis funcs Y_lm, l=0..3
     6. solve_linear three times (R/G/B) with band-aware Tikhonov
     7. sh_dc = coeffs[0]                                              (F1)
     8. if albedo.z>0 OR albedo.w>0: zero coeffs[1..15]                (F4)
     9. pack pos / normal / sh_dc / sh_rest planar /
            opacity (logit) / scale (log: tangent, tangent, normal·0.5)/ (F5)
            rotation = quat aligning local Z to surface normal
    10. write GaussianOut to SSBO at index id
            │
            ▼
   Vec<Gaussian>  ─►  nano-splat::ply::write_ply
                      (binary little-endian, Inria 3DGS layout)
```

Tags `(F1..F5)` index the splat-fit bias notes in `README.md` "Splat-fit
notes" and `plan1.md` §1.

---

## 5. Crate dependency map (live edges only)

```
                          ┌──── src/main.rs ────┐
                          │                     │
   ┌──────────────────────┴────┬────┬────┬──────┴────────┐
   ▼                           ▼    ▼    ▼               ▼
 nano-core               nano-io  nano-render  nano-splat  gpu-mem
   │ ▲                    │ │       │   ▲       │   ▲
   │ │                    │ │       │   │       │   │
   │ └──────────┬─────────┘ │       │   │       │   │
   │            │           │       │   │       │   │
   │            ▼           ▼       ▼   │       ▼   │
   │     (deps on nano-core for types)  │   (deps on│
   │                                    │   nano-core, nano-gpu, nano-shaders)
   ▼
 (glam, bytemuck, exr)                  │
                                        │
                          ┌─────────────┴─┐
                          ▼               ▼
                      nano-gpu       nano-shaders
                      (ash + shaderc) (str chunks)
                          │
                          ▼
                      (ash, shaderc, bytemuck, glam, nano-core)
```

Concrete edges from `Cargo.toml`:

```
nanotracer-rs (bin) → nano-core, nano-io, nano-render, nano-splat, gpu-mem
nano-render        → nano-core, nano-gpu, nano-shaders
nano-splat         → nano-core, nano-gpu, nano-shaders
nano-gpu           → nano-core
nano-io            → nano-core
nano-shaders       → (none — pure string constants)
nano-core          → (none — pure CPU data + glam)
gpu-mem            → (none — std only)
```

`splat-ref/` lives under `crates/` but is intentionally excluded from
the workspace (its imports reference the pre-refactor `crate::renderer`
namespace that no longer exists). It is a browsable source tree, not a
crate.

---

## 6. Dead-code map (post-cleanup)

The pre-2026-05 dead CPU pipeline (Intersection, Scene::intersect,
SceneBvh*, Mesh::intersect / build_bvh / normal_at / bounding_box /
surface_area, Sphere / Sphere::to_object, Hit, intersect_sphere,
DEFAULT_SKY_COLOR, and the `rtbvh` dependency) was removed in the
workspace-reorg wave. Only the splat-ref directory retains the historical
CPU helper sources, and only as a documentation artefact.

`MEMORY.md`-style debt as of 2026-05-15:

| Area | Note |
|---|---|
| `splat-ref/sampler.rs` | references `crate::renderer::*` that does not exist; intentionally excluded from workspace |
| `nano-splat::SplatConfigGpu::tonemap` | live and consumed |
| `nano-render::RenderConfig::light_sampling` | live; splat path always ignores |

---

## 7. Shared GLSL split (nano-shaders)

```
                        PREAMBLE (no globals)        HELPERS (needs bindings)
                        ───────────────────          ──────────────────────
  Material struct                  ✓                          —
  EPS / MAX_STACK / PI             ✓                          —
  wang_hash / rand01               ✓                          —
  max_component                    ✓                          —
  reflect_dir / refract_dir        ✓                          —
  offset_origin / checker_color    ✓                          —
  tonemap_reinhard / linear_to_srgb ✓                          —
  sample_environment               —                          ✓  (uses params, env_map)
  trace_ray / shadow_ray           —                          ✓  (uses topLevelAS)
```

Per-shader assembly:

```
GLSL source = PREAMBLE
              || <per-shader BINDINGS: local_size + bindings + Params + env_map>
              || HELPERS
              || <per-shader BODY: shader-specific helpers + main()>
```

The Rust helper `nano_shaders::assemble(bindings, body)` performs this
concatenation and the renderer / splat crates pass the result straight
to shaderc.

---

## 8. Decision log

- 2026-05-15 — Splat regression triaged: 5 shader fixes (F1–F5) landed
  in `nano-splat::generator`. Splat-ref CPU reference kept as docs.
- 2026-05-15 — Workspace reorganised into 7 crates. `rtbvh` dropped.
  Materials made energy-conserving. GLSL deduplicated via `nano-shaders`.
  Single `LightSampling` enum lifted to `nano-core`. `gpu-mem` vendored
  for VRAM/RAM probing. SH unit tests ported into `nano-core::sh`.
  wgpu evaluated (see `WGPU_RESEARCH.md`) — stay on `ash`.
- 2026-05-15 — `cargo clippy --workspace --all-targets`: 0 warnings,
  0 errors. `cargo test --workspace --lib`: 5 + 6 = 11 tests passing.
