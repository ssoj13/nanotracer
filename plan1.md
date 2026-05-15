# plan1.md — Bug Hunt Report

> **HISTORICAL DOCUMENT.** Written before the workspace reorg in the same
> 2026-05 wave. File paths below refer to the pre-reorg `src/`-shaped
> layout. The path translation is:
>
> | Old | New |
> |---|---|
> | `src/scene.rs`        | `crates/nano-core/src/scene.rs`     |
> | `src/geometry.rs`     | `crates/nano-core/src/geometry.rs`  |
> | `src/mesh.rs`         | `crates/nano-core/src/mesh.rs`      |
> | `src/material.rs`     | `crates/nano-core/src/material.rs`  |
> | `src/color.rs`        | `crates/nano-core/src/color.rs`     |
> | `src/environment.rs`  | `crates/nano-core/src/environment.rs` |
> | `src/gltf_loader.rs`  | `crates/nano-io/src/gltf_loader.rs` |
> | `src/utils.rs`        | `crates/nano-io/src/utils.rs`       |
> | `src/vk_runtime.rs`   | `crates/nano-gpu/src/vk_runtime.rs` |
> | `src/gpu_scene.rs`    | `crates/nano-gpu/src/gpu_scene.rs`  |
> | `src/rt_renderer.rs`  | `crates/nano-render/src/renderer.rs`|
> | `src/rt_splats.rs`    | `crates/nano-splat/src/generator.rs`|
> | `src/splat_gpu.rs`    | `crates/nano-splat/src/ply.rs`      |
> | `src/splat/**`        | `crates/splat-ref/src/**` (excluded from workspace) |
>
> Status of each section as of 2026-05-15 close-out:
>
> - §1 (F1–F5 shader fixes) — landed in `crates/nano-splat/src/generator.rs`.
> - §2.1 (dead CPU code) — removed, except for `splat-ref/` which is
>   intentionally retained as a browsable reference and excluded from the
>   workspace.
> - §2.2 (GLSL duplication) — landed via `crates/nano-shaders` and the
>   `assemble(BINDINGS, BODY)` helper.
> - §2.3 (duplicated Rust types) — `LightSampling` lifted to `nano-core`;
>   `Gaussian` / `write_ply` / `quat_from_normal` / `estimate_scale`
>   duplicates resolved when the orphan `splat/` moved to `splat-ref/`.
> - §2.4 (material energy non-conservation) — fixed in `nano-core::material`.
> - §3.1 / §3.2 / §3.3 — done (see above).
> - §3.4 (Phong normalisation, `--sh-keep-glossy`, ported tests) —
>   tests ported to `nano-core::sh`; Phong normalisation and CLI flag
>   parked in `todo.md`.
> - §5 (open questions) — resolved in `todo.md` "Parked questions".

---

**Scope:** `nanotracer-rs` codebase audit triggered by the Gaussian-splat regression.
**Date:** 2026-05-15.
**Status of the immediate bug:** three fixes already landed in `src/rt_splats.rs` for the visible "colored noise" (see §1). This document catalogues everything **still** worth doing, prioritised, with file:line references.

> Convention used throughout: `path/file.rs:LINE` for clickable refs in IDE / GitHub.

---

## 0. Architectural picture (as discovered)

The project is mid-way through a **CPU → GPU refactor that was never finished**.
Two parallel realities coexist:

| Concern | Active path (GPU) | Orphan path (dead CPU code) |
|---|---|---|
| Image rendering | `rt_renderer.rs` (Vulkan ray-query compute) | none (CPU renderer was removed) |
| Splat generation | `rt_splats.rs` + `splat_gpu.rs` (PLY writer) | `splat/{mod,sh,sampler,ply}.rs` (CPU-side reference) |
| Ray intersection | Vulkan AS / ray query | `Mesh::intersect`, `Geometry::intersect`, `Scene::intersect`, `SceneBvh*` |
| Scene → triangles | `gpu_scene::build_gpu_scene*` | (none) |
| Light/env sampling | shader-side | `Scene::sample_environment`, `DEFAULT_SKY_COLOR` |

The orphan path is **not even reachable** — `src/splat/` is not declared in `src/lib.rs:1-13`. Worse, `src/splat/sampler.rs:11` imports `crate::renderer::{RayConfig, cast_ray_cfg, offset_origin, reflect, refract}` — a module that **no longer exists** (renamed to `rt_renderer` and stripped to GPU-only). So even if you added `pub mod splat;` to `lib.rs`, it would not compile. This is the smoking gun for the "lost pieces" the user suspected.

---

## 1. What was already fixed today (`src/rt_splats.rs`)

Documented here so future-me does not undo them.

| # | Location | Before | After | Reason |
|---|---|---|---|---|
| F1 | `rt_splats.rs:~1346` (shader `sh_dc =`) | `coeffs[0] / SH_C0` | `coeffs[0]` | LSQ already produces 3DGS-convention DC (viewer multiplies by SH_C0). Extra division gave ~3.5× boost → black/white blowout. |
| F2 | `rt_splats.rs:shade_surface` (~`1018-1055`) | branched on `light_sampling==1u` (random one light) | always loops all lights | One-light Monte-Carlo gives high variance per direction; LSQ over hemisphere captures that variance as multicolour SH ringing. CPU reference in `splat/sampler.rs:267-287` also sums all lights. |
| F3 | `rt_splats.rs:~1294-1304` (Tikhonov diagonal) | scalar `+= 1e-4` | band-aware: 1e-3 (l=0) … 2e-2 (l=3) | Hemisphere-only LSQ is rank-deficient for high-frequency SH modes; flat 1e-4 was not enough to damp ringing on otherwise constant surfaces. |
| F4 | `rt_splats.rs:~1340-1348` (post-fit) | (no clamping of bands) | for `albedo.z>0 \|\| albedo.w>0` zero `coeffs[1..15]` | Sharp specular / refraction cannot be represented by order-3 SH on a hemisphere → "rainbow speckle" on mirror & glass. Falls back to DC-only (matches CPU `from_sample_constant`). |
| F5 | `rt_splats.rs:~1377-1383` (`normal_sigma`) | `splat_scale * 0.3` | `splat_scale * 0.5` | At `test_lo.cmd` settings (`--splat-density 512 --splat-scale 0.02`) the discs were 0.006 thin — visible grazing-angle gaps showed as speckle. |

These changes were verified to compile (shaderc) and produce a PLY with DC values in a sane range and the expected number of zero-rest entries for reflective/refractive splats.

---

## 2. Findings — ordered by impact

### 2.1. HIGH — Dead code from abandoned refactor (deletable today)

Removing this is the single biggest readability win and what the user explicitly asked about ("lost pieces from a big refactoring").

**`src/splat/` (entire directory) — never compiled in:**
- `src/splat/mod.rs:6-8` declares `ply`, `sampler`, `sh` but **`splat` itself is not declared in `src/lib.rs`**.
- `src/splat/sampler.rs:11` imports a non-existent `crate::renderer::…` — bit-rotted.
- Duplicate types vs the live path:
  - `splat/mod.rs:71` `pub struct Gaussian` ↔ `splat_gpu.rs:7` `pub struct Gaussian` (identical layout).
  - `splat/ply.rs:25` `write_ply` ↔ `splat_gpu.rs:17` `write_ply` (line-for-line identical).
  - `splat/mod.rs:157` `quat_from_normal` (Rust) ↔ `rt_splats.rs:~1141` (GLSL) — same math, two languages.
  - `splat/sampler.rs:321` `estimate_scale` ↔ `rt_splats.rs:440` `estimate_scale` (identical).
- CPU-only reference fitter: `splat/sh.rs` `sh_basis`, `sh_basis_all`, `sh_dc_from_srgb`, `fit_sh`, `fibonacci_hemisphere`, `eval_sh`, `solve_linear_system`. Useful as a docstring reference but currently disconnected.

**`src/scene.rs` — large CPU-pipeline residue:**
- `Intersection` struct + `empty/from_hit/new` — only used by `scene.rs` itself and the orphan `splat/sampler.rs`.
- `Scene::intersect` (`scene.rs:134`), `Scene::intersect_checkerboard` (`scene.rs:191`), `Scene::rebuild_scene_bvh` (`scene.rs:113`), `Scene::sample_environment` (`scene.rs:90`), `Scene::object_aabb` (`scene.rs:225`) — none called from `main.rs`, `rt_renderer`, `rt_splats`, `gpu_scene`.
- `SceneBvh`, `SceneBvhNode`, `SceneObjectProxy` (`scene.rs:243-332`) — entire CPU BVH for object-level culling, never invoked.
- `Scene::add_sphere(Sphere)` (`scene.rs:98`), `Sphere::new` (`geometry.rs:119`), `Sphere::to_object` (`geometry.rs:128`) — legacy adapter; `main.rs` only uses `Object::sphere` + `Scene::add_object`.
- `DEFAULT_SKY_COLOR` (`environment.rs:184`) — only referenced by the dead `Scene::sample_environment`.

**`src/geometry.rs` & `src/mesh.rs` — CPU intersection helpers:**
- `Geometry::intersect` (`geometry.rs:56`), `Geometry::surface_area` (`geometry.rs:72`), `intersect_sphere` (`geometry.rs:81`).
- `Mesh::intersect` (`mesh.rs:139`), `Mesh::intersect_triangle` (`mesh.rs:163`), `Mesh::normal_at` (`mesh.rs:205`), `Mesh::bounding_box` (`mesh.rs:216`), `Mesh::surface_area` (`mesh.rs:228`).
- `TriangleHit`, `Hit` structs.
- **Perf consequence:** `Mesh::new` and `Mesh::with_normals` (`mesh.rs:65,79`) call `build_bvh()` for every mesh ever constructed (cube/pyramid/torus/sphere shells in `gpu_scene::sphere_mesh`, glTF loads, randomised demo objects in `main.rs:391`). The BVH is then never read. This is wasted setup time on every run.

### 2.2. HIGH — Duplicated GLSL shader source between `rt_renderer.rs` and `rt_splats.rs`

Both files embed a >700-line `const X: &str = r#"…"#` shader. They share **identical** copies of:

- `wang_hash`, `rand01` (renderer `:754-765`, splats `:774-785`)
- `reflect_dir`, `refract_dir`, `offset_origin` (renderer `:771-800`, splats `:803-832`)
- `checker_color`, `sample_environment` (renderer `:802-825`, splats `:834-857`)
- `trace_ray`, `shadow_ray` (renderer `:827-848`, splats `:872-893`)
- `Material` struct + bindings 0…7 of the descriptor set
- (Specific to splats: `tonemap_reinhard`, `linear_to_srgb`, SH machinery, `quat_from_normal`, `find_triangle`, hemisphere sampler.)

Recommendation: a single `pub const GLSL_COMMON: &str = …` (e.g. in a new `src/shaders.rs` or as a sibling const in `vk_runtime`) prepended to each shader at compile-time. The shaderc invocation already passes a single combined string, so concatenation is trivial. **Pure deduplication, no logic change.**

### 2.3. MEDIUM — Duplicated Rust types & constants

- `pub enum LightSampling { All, One }` declared twice: `rt_renderer.rs:28` and `rt_splats.rs:17`. Same semantics for the Vulkan params marshalling, different conceptual meaning at the splat call-site (after F2 the splat fitter no longer uses the "One" branch). Cleanup: lift to a shared module, OR keep the duplication explicit and rename the splat one to `SplatLightSampling` to signal "rendering only".
  - Note: as of this audit `main.rs:11-12` already aliases them as `RenderLightSampling` / `SplatLightSampling`, so the renaming work is half-done.
- `Gaussian` (twice — covered in §2.1).
- `write_ply` (twice — covered in §2.1).
- `estimate_scale` (twice — covered in §2.1).
- `quat_from_normal` (CPU `splat/mod.rs:157` + GLSL `rt_splats.rs:~1141`). After deleting `splat/`, only the GLSL copy remains.

### 2.4. MEDIUM — Energy non-conservation in materials

Inherited verbatim from tinyraytracer; documented here so it is not "rediscovered" as a bug later.

- `material.rs:38` `RED_RUBBER.albedo = [1.4, 0.3, 0.0, 0.0]` — diffuse multiplier 1.4.
- `material.rs:45` `MIRROR.albedo = [0.0, 16.0, 0.8, 0.0]` — specular intensity 16.0.
- `material.rs:24` `IVORY.albedo = [0.9, 0.5, 0.1, 0.0]` — total 1.5.

Consequence in the splat path: F4 (drop SH for `albedo.z>0`) catches `IVORY` (z=0.1). That is fine for splat output but means IVORY visually behaves identically to a slightly-reflective matte. If a stricter threshold is wanted, change the check at the F4 site to e.g. `albedo.z > 0.3 || albedo.w > 0.3` so IVORY keeps its view-dependent SH.

### 2.5. LOW — Misnamed function

- `splat/mod.rs:51` `pub fn rest_interleaved(&self) -> Vec<f32>` returns data in **planar** order (R[1..15] then G[1..15] then B[1..15]), as confirmed by its own body and matched by `rt_splats.rs:~1341-1343`. Either rename to `rest_planar`, or just delete with the rest of `splat/`.

### 2.6. LOW — `Scene::checkerboard_enabled` & the floor

The checkerboard plane is currently materialised twice:
- Inside `gpu_scene::checkerboard_plane_mesh` (`gpu_scene.rs:253`) — fed to the GPU AS; this is the live one.
- Inside `Scene::intersect_checkerboard` (`scene.rs:191`) — CPU-side, never called.

Both hard-code the same constants (`y=-4`, `x∈[-10,10]`, `z∈[-30,-10]`). If we keep the GPU mesh, the CPU copy goes away with §2.1. If at any point you want a CLI flag to move/resize the plane, extract those constants to one place first.

### 2.7. LOW — Dispatch and AS observations (no bug, sanity checked)

Verified during the audit and intentionally **not** changing:

- `gpu_scene.rs:140-148` builds `tri_cdf` normalised to [0, 1] with a guard against zero total weight. ✓
- `find_triangle` in `rt_splats.rs:~1174` is a correct lower-bound binary search on that CDF. ✓
- Barycentric sampling in `rt_splats.rs:~1208-1215` follows the canonical `sqrt(r1)` formula (weights labelled u/v/w in unusual order but sum to 1). ✓
- `Material` `std430` layout (Rust `gpu_scene.rs:11-21` with `_pad0`/`_pad1`) matches the GLSL struct (`rt_renderer.rs:699-707` / `rt_splats.rs:716-724`). ✓
- Compute dispatch `(sample_count + 63) / 64` matches local size 64 with a per-thread early-out. ✓
- PLY field order (`splat_gpu.rs:30-65`) matches Inria 3DGS convention: pos, normal, f_dc[3], f_rest[45 planar], opacity (logit), scale[3] (log), rot[4] (w,x,y,z). ✓

---

## 3. Recommended action plan

Each item is independently mergeable.

### 3.1. Delete dead code — single dedicated PR

Touch list (deletions only — verify no `grep` survivors after each step, then `cargo build`):

```
src/splat/                                  (entire directory)
src/scene.rs:                               Intersection (whole impl),
                                            Scene::intersect, intersect_checkerboard,
                                            rebuild_scene_bvh, sample_environment,
                                            object_aabb, add_sphere,
                                            SceneObjectProxy, SceneBvhNode, SceneBvh
src/geometry.rs:                            Hit, intersect_sphere,
                                            Geometry::{intersect, surface_area},
                                            Sphere, Sphere::{new, to_object}
src/mesh.rs:                                TriangleHit, TrianglePrimitive,
                                            Mesh::{build_bvh, intersect, intersect_triangle,
                                                   normal_at, bounding_box, surface_area},
                                            Mesh.bvh & Mesh.primitives fields,
                                            rtbvh dep usage in this module
                                            (compute_smooth_normals stays — still useful)
src/environment.rs:                         DEFAULT_SKY_COLOR
src/scene.rs:                               import of DEFAULT_SKY_COLOR
```

`Cargo.toml`: `rtbvh = "0.6.2"` is only needed for the CPU BVH we're deleting. If nothing else uses it after the prune, remove from dependencies.

Imports in `scene.rs:5-9` shrink to `environment::EnvironmentMap`, `geometry::Object`, `material::Material` (and `Material` will be unused once `Intersection` goes — re-check).

After the prune the `pub struct Mesh { vertices, indices, normals }` is a plain POD that `gpu_scene` reads; `Mesh::new` / `Mesh::with_normals` become trivial constructors.

### 3.2. Deduplicate GLSL — small standalone PR

1. Introduce `src/shaders/common.rs` (or a `const GLSL_COMMON` in `vk_runtime.rs`) containing:
   - The `Material` struct text.
   - `wang_hash`, `rand01`, `max_component`.
   - `reflect_dir`, `refract_dir`, `offset_origin`.
   - `checker_color`, `sample_environment`.
   - `trace_ray`, `shadow_ray`.
2. In both `rt_renderer.rs` and `rt_splats.rs`, build the shader string with `format!("{}{}\n{}", VERSION_HEADER, GLSL_COMMON, SPECIFIC_PART)`.
3. Verify both still compile through shaderc and produce identical SPIR-V semantics.

This is the King-and-Queen rule of the task brief: one source of truth.

### 3.3. Lift `LightSampling` — trivial PR

Move into a new `src/render_common.rs` (or just `scene.rs`) and `pub use` from both renderers. Drop the alias dance in `main.rs:11-12`.

### 3.4. Optional polish

- **F4 threshold knob**: expose `--sh-keep-glossy` (default off). When on, do not zero `coeffs[1..15]` for `albedo.z>0` materials. Useful if someone really wants the ringing in exchange for shiny streaks.
- **Material consts**: leave non-physical values as-is for now (tinyraytracer fidelity), but add a one-line doc-comment on `material.rs:43-48` explaining MIRROR's albedo[1]=16.0 is intentional tinyraytracer style.
- **Tests**: `splat/sh.rs:269-345` has good unit tests for SH fitting and Fibonacci hemisphere — if we kill the orphan module, **port those tests** to a unit test crate driving the GLSL via a tiny CPU mirror (or just keep `sh.rs` alive *as a tested CPU reference* by adding `pub mod splat;` to `lib.rs` after stripping `sampler.rs` and `mod.rs::Gaussian/SplatConfig` duplications).

---

## 4. Quick verification recipe

After any deletion in 3.1:

```
$env:VULKAN_SDK = "C:\Programs\VulkanSDK\1.4.341.1"
cargo build --release
.\target\release\nanotracer-rs.exe -n 12 --seed 123 -S smoke.ply `
    --splat-density 64 --sh-samples 32 --light-sampling all `
    --splat-scale 0.05 -m 6 -r 3 -f 4
```

Expected: ~3-4 s total, "splat generation complete", `smoke.ply` opens in SuperSplat/Luma without rainbow speckle on mirror/glass.

---

## 5. Open questions to confirm before deletion PR

1. **Keep `splat/sh.rs` as a tested CPU reference?** It has working unit tests and validates the math used in the GLSL `sh_basis`. If kept, prune `sampler.rs` + `mod.rs` only and re-attach to `lib.rs` after dedup. If dropped, delete `splat/` whole.
2. **Tighten F4 threshold for IVORY?** Decision parameter is a single number; cheap either way.
3. **`rtbvh` dependency**: if dropped per §3.1, remove from `Cargo.toml`; if any future feature needs scene-level BVH, leave the dep but still delete the dead code.

Awaiting approval before executing 3.1 / 3.2 / 3.3.
