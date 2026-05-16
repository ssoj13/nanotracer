# Roadmap

Active and parked work. Item priority: 🟥 ship-blocking, 🟧 visible
improvement, 🟨 nice-to-have, 🟩 long-horizon. Tick items as they land,
prune as they age.

## Done in 2026-05 wave

- [x] Splat regression: 5 shader fixes (`SH_C0` DC, light sampling in
      SH fitter, band-aware Tikhonov, DC-only for glossy/refractive,
      thicker disc normal). See `plan1.md` §1.
- [x] Workspace reorg into 7 crates (`gpu-mem`, `nano-{core,io,shaders,
      gpu,render,splat}`) with `splat-ref` retained as docs.
- [x] Physical materials (energy-conserving, `[kd, ks, kr, kt]`,
      `Σ ≤ 1`).
- [x] Dead CPU intersection code removed (Intersection / SceneBvh /
      Mesh::intersect / etc.), `rtbvh` dropped.
- [x] GLSL deduplicated — `nano-shaders::{PREAMBLE, HELPERS, assemble}`.
- [x] Single `LightSampling` in `nano-core`.
- [x] `gpu-mem` vendored + startup VRAM print.
- [x] CPU SH reference (`nano-core::sh`) + 4 unit tests ported from
      `splat-ref/sh.rs`.
- [x] `cargo clippy --workspace`: 0 warnings, 0 errors.
- [x] README / AGENTS / DIAGRAMS synced.

## Done in 2026-05 wave (cont.)

- [x] Normalised Phong specular `(n+2)/(2π)` (subsequently superseded
      by GGX — see below). Materials rebalanced so `ks ≈ F₀ ≈ 0.04`.
- [x] `--sh-keep-glossy` CLI flag wired through `SplatConfigGpu` and the
      splat shader uniform block.
- [x] Schlick Fresnel reflection/refraction split for dielectric
      materials (`kr > 0 && kt > 0`), e.g. `GLASS`.
- [x] Truncate diffuse-only splats to degree-1 SH (higher bands are
      noise for Lambertian; cleaner fit).
- [x] Curvature-aware detail boost using pairwise vertex-normal
      disagreement (`max(1 - n_i·n_j)`) — catches creases the old
      face/vertex delta missed.
- [x] Progress bar split: 8 phases — `upload buffers` / `build accel` /
      `upload env` / `compile shader` / `create pipeline` /
      `write descriptors` / `dispatch` / `readback`.
- [x] GGX / Trowbridge–Reitz microfacet specular replaces Phong in both
      shaders (`phong_to_alpha(n)` mapping for back-compat). Smith
      geometry + Schlick Fresnel absorbed into the BRDF.
- [x] IBL diffuse via degree-2 SH env-irradiance pre-convolved on CPU
      (Ramamoorthi–Hanrahan band factors). Uploaded as `vec4[9]` in the
      Params UBO and evaluated at the surface normal in both shaders.

## Active

**Plan A — full 3DGS gradient-based optimisation.** Wired in `nano-optimize`
crate (wgpu 29). Phase A1 scaffolding landed; rasteriser + backward pass +
loss + densify still ahead.

- [x] **A1 Scaffolding** — `nano-optimize` crate, `SplatBuffer`,
      `AdamState`, multi-view Fibonacci-sphere reference baking via
      `nano-render`, training-loop skeleton, `--train` CLI flag + tuning
      flags (`--train-iters`, `--train-views`, `--train-width`,
      `--train-height`, `--train-max-splats`).
- [x] **Area lights + IBL as `Light::Env`.** `Light` is now an enum
      `{Point, Rect, Sphere, Box, Env}` in `nano-core` with `area()`,
      `radiance()`, `sample()`. 64-byte `GpuLight` in `nano-gpu` paired
      with a `light_radiance` SSBO. Unified `sample_light` GLSL helper
      in `nano-shaders` dispatches by kind (point: unit-radiance no
      falloff; rect/box: uniform-area MC; sphere: solid-angle MC;
      env: cosine-convolved SH). CLI: `--point-light`, `--rect-light`,
      `--sphere-light`, `--box-light`, `--env-light` (replaces the old
      `--ibl-strength` knob — IBL intensity now flows through the
      explicit env-light).
- [x] **A2 Differentiable forward rasteriser** — tile-based α-blending
      front-to-back in WGSL compute, view/proj projection of splats,
      forward pass producing predicted frames. Six sub-phases shipped:
      - A2.0 — wgpu context + `GpuSplatBuffer` upload/readback roundtrip.
      - A2.1 — `project_gaussians.wgsl`: 3D Σ → 2D conic, depth, radius,
        full SH eval + CPU oracle for parity testing.
      - A2.2 — GPU exclusive prefix-scan (three-level, up to 16M elts).
      - A2.3 — Stable GPU radix sort (1-bit-at-a-time, 32 passes).
      - A2.4 — Tile binning: count → scan → emit → sort → per-tile ranges.
      - A2.5 — `rasterize.wgsl`: per-tile workgroup α-blends sorted
        splats front-to-back into `vec4` framebuffer (xyz + coverage).
      - A2.6 — End-to-end integration in `train()`: forward rasterise
        per iteration, MSE vs reference, PNG dump on iter 0.
- [x] **A3 Backward pass** — gradients through α-blend back to per-splat
      (pos, rot, scale, opacity, sh). Validated with finite differences.
      - A3.1 — `GradSplatBuffers` + `ProjectedGrad` (48-byte per-splat
        2D-state grad slot).
      - A3.2 — `rasterize_backward.wgsl`: per-tile reverse α-blend with
        CAS-based atomic-add-f32; T-clamp at >1 to handle forward
        early-out reconstruction (numerical stability).
      - A3.3 — `project_backward.wgsl`: chain through sigmoid, SH basis,
        conic inverse, Σ-sandwich, quaternion derivatives, log-σ.
      - A3.4 — Finite-difference verification (ε=1e-3, ≤ 5–10% rel):
        sh_dc, opacity, sh_rest, pos, log-scale all parity. Catches
        sign / scale errors in any chain rule step.
      - A3.5 — Integrated into `train()`: forward → MSE → backward →
        gradient-norm log. Adam updates land in A4.
- [ ] **A4 Loss** — L1 + SSIM photometric loss vs reference views, grad
      backprop into Adam state.
- [ ] **A5 Densify-and-prune** — split high-gradient splats, prune
      low-opacity / low-importance; cap at `--train-max-splats`.

**Plan B — Splat viewer.** Built on top of Plan A's WGSL rasteriser.
"Сразу нормальный" — no half-jobs, proper navigation, depth-sort, env
background, gamma-correct output. Blocked on A2 (shared kernel).

- [ ] **B1 Standalone PLY viewer.** New crate `nano-view`. winit +
      wgpu surface, loads a `.ply`, presents in a window. Camera:
      orbit (LMB rotate, RMB pan, scroll zoom) + WASD fly mode + FoV
      slider. Per-frame radix depth-sort by camera direction. HUD:
      FPS, frame-time, splat count, current camera pose. Env-map
      background reuses `nano-core::environment`. CLI: `--view scene.ply`.
      Reuses A2's `rasterize.wgsl` verbatim — no duplicate kernels.
- [ ] **B2 Training-time live preview.** Reuse B1's window + pipeline.
      Flag `--view-training`: during `train()` the window shows the
      current predicted frame from a fixed reference camera, refreshed
      every N iterations. Surface present runs on a separate command
      pool from training compute so they don't serialise. Side panel:
      live loss curve (L1 + SSIM), iteration counter, splat-count
      trend. Useful for spotting divergence early.
- [x] **A1 extension — multi-view camera in renderer.** `RenderConfig`
      now carries `camera_pos / camera_target / camera_up`; the shader
      consumes `inv_view`; `bake_references` produces distinct frames.

Multi-scattering compensation (Heitz/Hill) landed earlier as
`nano_shaders::ggx_msc_boost`.

## 🟧 Visible improvements

_(empty — Plan A subsumes the per-roughness mip-chain idea via
differentiable rasterisation; if a non-PBR demo needs IBL specular fast
it can revisit.)_

## 🟨 Polish

_(empty — all landed)_

## 🟩 Long horizon

- [ ] Per-roughness env-map mip chain for IBL specular (split-sum
      à la Real Shading in UE4). Only matters if Plan A doesn't ship —
      direct optimisation captures the same effect physically.
- [ ] Full `wgpu` consolidation: migrate `nano-render` / `nano-gpu` /
      `nano-splat` off ash once `EXPERIMENTAL_RAY_QUERY` stabilises.
      See `WGPU_RESEARCH.md`.
- [ ] `no-std` Rust subset for ARM SBC targets. Path: lift `nano-core`
      to `#![no_std]`, keep `nano-gpu`/`nano-shaders` host-only.

## Parked questions

- Are the test scripts (`test_hi.cmd`, `test_lo.cmd`) the right
  default-quality presets after the physical-materials change? Re-tune
  if visual feedback says yes.
- Should `splat-ref/` ever be wired back into the workspace? Decision
  so far: no — keep as reference. Re-evaluate when porting tests for
  any other CPU helper.
