# Origin & evolution

The codebase started as a Rust port of Dmitry V. Sokolov's
[tinyraytracer](https://github.com/ssloy/tinyraytracer) and has been
restructured several times. Each restructuring removed a layer rather
than adding one — at every stage the live path got narrower and more
specialised.

## tinyraytracer (CPU, single file)

The original was a C++ raytracer in one source file: pinhole camera,
sphere intersection, Phong shading, recursive reflection/refraction,
checker plane. The Rust port replicated that shape file-for-file
(`renderer.rs`, `scene.rs`, `material.rs`, `mesh.rs`) and added a
threaded tile scheduler.

## CPU pipeline (multi-file Rust)

Mesh support, glTF import, BVH (via `rtbvh`), HDR environment maps,
Gaussian-splat **fitting** on the CPU. The fitter sampled hemispheres
around surface points and solved a least-squares system for order-3
spherical-harmonic coefficients per splat. Output was 3DGS-compatible
PLY (Inria binary schema), readable by SuperSplat / Luma /
bevy_gaussian_splatting.

This phase produced visually correct splat clouds but slowly — a few
seconds per thousand splats. The CPU BVH was the throughput cliff.

## GPU refactor (Vulkan ray queries)

The CPU intersection layer (`Intersection`, `SceneBvh`, `rtbvh` etc.)
was deleted outright. Vulkan compute shaders with `VK_KHR_ray_query`
took over both image rendering and splat fitting; the host became a
buffer marshaller. Both shaders shared a single helper preamble
(`reflect_dir`, `trace_ray`, `shadow_ray`, etc.) via the
`nano-shaders` crate.

A bug-hunt wave (`plan1.md` at the time) fixed five splat-fit issues
that had been masked by CPU noise: DC double-scaling, hemisphere LSQ
ringing, glossy-material rainbow speckle, light-sampling speckle,
disc-normal gaps. Documentation pinned these as `F1`–`F5` references.

## Workspace reorganisation

The monolithic `src/` was split into seven crates (`gpu-mem`,
`nano-core`, `nano-io`, `nano-shaders`, `nano-gpu`, `nano-render`,
`nano-splat`) with explicit boundaries:

- `nano-core` — CPU data: `Scene`, `Object`, `Mesh`, `Material`,
  `Light`, `EnvironmentMap`, SH reference. No GPU types.
- `nano-gpu` — `ash` runtime + scene → GPU buffer marshalling. No
  shader strings.
- `nano-shaders` — `&'static str` GLSL chunks (`PREAMBLE`, `HELPERS`,
  `assemble()`). Pure constants, zero deps.
- `nano-render` / `nano-splat` — pipelines that compose the above.

GGX/Trowbridge–Reitz replaced Phong; multi-scattering compensation
(Turquin 2018) was added. Materials were rebalanced to be
energy-conserving (`Σ kd + ks + kr + kt ≤ 1`).

## Area lights + IBL as `Light`

`Light` was lifted from a position-only struct into an enum
`{Point, Rect, Sphere, Box, Env}`:

- Each non-env variant carries per-light color + intensity.
- GPU side gets a 64-byte `GpuLight` paired with a parallel
  `light_radiance` SSBO. No bit-packing tricks.
- A single `sample_light(idx, hit_pos, hit_n, rand_uv)` GLSL helper
  dispatches by kind: point (unit-radiance, no falloff — legacy
  convention), rect / box (uniform-area MC), sphere (solid-angle MC),
  env (cosine-convolved SH).
- The old `--ibl-strength` knob was retired; IBL intensity flows
  through `Light::Env { intensity }`. The previous double-knob
  collapsed into one dimension.

## Plan A — 3DGS training pipeline (new crate)

`nano-optimize` introduced gradient-based splat optimisation on
`wgpu`. The raytrace path stayed on `ash` because the reference
baking needs stable hardware ray queries; `wgpu`'s `RAY_QUERY` was
still experimental at the time (see `WGPU_RESEARCH.md`).

Six sub-phases shipped over a single development cycle:

| Sub-phase | What |
|-----------|------|
| A2.0 | `WgpuCtx`, `GpuSplatBuffer` (6 SSBOs, vec4-padded). |
| A2.1 | `project_gaussians.wgsl` + CPU oracle parity. |
| A2.2 | GPU exclusive prefix-scan (three-level Hillis-Steele). |
| A2.3 | Stable GPU radix sort (1-bit-at-a-time, 32 passes). |
| A2.4 | Tile-binning pipeline (count → scan → emit → sort → ranges). |
| A2.5 | `rasterize.wgsl` (per-tile α-blend with workgroup-shared loads). |
| A2.6 | End-to-end forward in `train()` with MSE log + PNG dump. |
| A3.1 | `GradSplatBuffers` + `ProjectedGrad` (atomic-add-f32 via CAS). |
| A3.2 | `rasterize_backward.wgsl` + numerical-stability T-clamp. |
| A3.3 | `project_backward.wgsl` (full chain rule). |
| A3.4 | Finite-difference verification (≤ 5–10% rel on all params). |
| A3.5 | Wire into `train()` + gradient-norm logging. |
| A4.1 | `GpuSplatBuffer::sync_from` for in-place updates. |
| A4.2 | Adam updates per attribute + physical constraints. |
| A5.1 | Prune low-opacity splats every 100 iters. |
| A5.2 | Densify (clone + split) every 200 iters + Adam-state resize. |

The pivotal A3.2 bug — backward T-reconstruction blowing up past
the forward early-out boundary — produced NaN gradients on real
scenes despite the CPU finite-difference test passing on a synthetic
one. The fix was a one-line `if T > 1.0001 { done }` guard, but
diagnosing it took half the A3.5 commit.

## Plan B — Interactive viewer

`nano-view` lands the standalone viewer in four sub-phases:

| Sub-phase | What |
|-----------|------|
| B1   | winit + `wgpu::Surface`, CPU-readback presentation. |
| Bx.1 | Full `egui_dock` UI + GPU-only `tonemap.wgsl`. |
| Bx.2 | PLY loader (`--view-ply scene.ply` fast path). |
| Bx.3 | RMB pan + WASD/QE fly mode. |
| Bx.4 | Live training preview — worker thread + `egui_plot` loss curve. |

The GPU-only path (Bx.1) eliminated the per-frame CPU bounce. Bx.4's
worker shuttles per-iteration `SplatBuffer` snapshots via
`Arc<RwLock<Option<TrainSnapshot>>>`; the viewer compares monotonic
versions, re-uploads on change (full realloc when densify shrinks /
grows the count, `sync_from` otherwise).

## Removed along the way

- `rtbvh`, the CPU BVH crate (post-Vulkan).
- `Intersection`, `SceneBvh`, `Mesh::intersect`, `Hit`, the entire CPU
  raytrace path (post-Vulkan).
- The `--ibl-strength` CLI flag and `RenderConfig::ibl_strength` /
  `SplatConfigGpu::ibl_strength` (replaced by `Light::Env`).
- `plan1.md` (the bug-hunt history was archived after the workspace
  reorg).
- Phong specular (replaced by GGX).
- The four-arg byte-radix sort attempt (atomicAdd scatter is
  non-stable; switched to 1-bit-at-a-time).
- The CPU readback blit in B1 (replaced by `tonemap.wgsl` + egui
  texture register in Bx.1).

What remained is the live path: scene → GPU → splats → train → view.
