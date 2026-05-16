# Changelog

Notable changes to `nanotracer-rs`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses calendar-style milestone notes rather than strict SemVer until
Plan A stabilises.

## 2026-05-15 — Plan A complete + interactive viewer

End-to-end 3DGS training pipeline (forward rasteriser → backward pass →
Adam updates → densify-and-prune) now ships and converges. Standalone
interactive viewer lands alongside.

### Added — Area lights + IBL as a unified `Light`

- `nano-core::scene::Light` is now an enum with five variants:
  `Point`, `Rect`, `Sphere`, `Box`, `Env`. Each non-env variant carries
  per-light `color: Vec3` + `intensity: f32`; methods `area()`,
  `radiance()`, `sample(rand_uv)` expose the surface integral and
  uniform-area sampling. `Env` is image-based lighting promoted to a
  first-class light.
- `nano-gpu::gpu_scene::GpuLight` — 64-byte `std430` record, paired
  with a parallel `light_radiance` SSBO (no W-channel bit-packing).
- Unified `sample_light(idx, hit_pos, hit_n, rand_uv)` helper in
  `nano-shaders::HELPERS` dispatches by `kind`:
  - **Point** — legacy unit-radiance, no falloff.
  - **Rect / Box** — uniform-area Monte-Carlo, two-sided gate, area-form
    geometric attenuation `cos_y · A / r²`.
  - **Sphere** — solid-angle Monte-Carlo (PBRT §14.2.2), area fallback
    when the receiver is inside the light.
  - **Env** — pre-convolved Lambertian SH irradiance
    (Ramamoorthi–Hanrahan).
- New CLI flags: `--point-light`, `--rect-light`, `--sphere-light`,
  `--box-light`, `--env-light`. Robust `parse_floats` helper with
  loud panics on malformed input.

### Removed

- `--ibl-strength` CLI knob and `RenderConfig::ibl_strength` /
  `SplatConfigGpu::ibl_strength`. IBL intensity now flows through
  `Light::Env`, eliminating the duplicate dial. `nano-gpu::gpu_scene`
  auto-inserts an implicit `Light::Env { intensity: 1.0 }` when a
  scene has an environment map but no explicit env-light (back-compat
  for tests and library callers).

### Added — Phase A2: Differentiable forward rasteriser

Six sub-phases, all on `wgpu` (the raytracer crates stay on `ash`).

- **A2.0** wgpu context + `GpuSplatBuffer` (6 SSBOs, `vec3` attrs
  padded to `vec4` for `std430`; `sh_rest` padded 45 → 48 floats per
  splat). CPU↔GPU roundtrip test verifies bit-equality.
- **A2.1** `shaders/project_gaussians.wgsl` — per-splat
  world→screen projection. `quat→mat3`, `Σ_3D = R·diag(σ²)·R^T`,
  Jacobian sandwich `J·W·Σ·W^T·J^T`, Inria 3-pixel low-pass dilation,
  conic = `inv(Σ_2D)`, 3σ pixel-space radius, screen-frustum cull,
  full SH evaluation (DC + bands 1–3, 16 coefficients per channel),
  opacity sigmoid + colour non-negative clamp. CPU oracle in
  `cpu_oracle` mirrors the WGSL exactly; GPU↔CPU parity ≤ 1e-3.
- **A2.2** GPU exclusive prefix-scan (Hillis-Steele). Three-level
  iterative scan supports up to `256³ = 16,777,216` elements.
  Standalone tests: all-ones → indices, random 70k vs CPU reference.
- **A2.3** Stable GPU radix sort. 32 passes of 1-bit stable split
  (predicate → exclusive scan → total-zeros counter → scatter).
  Each pass is stable by construction — atomicAdd-based byte-radix
  was tried first and discarded because it breaks the radix-sort
  invariant for equal-keyed entries.
- **A2.4** `shaders/tile_{count,emit,ranges}.wgsl` + Rust
  `TileBinner`. Pipeline: per-splat tile-bbox count + atomic total →
  exact buffer alloc → `PrefixScan` → emit `(tile_id << 16 |
  depth_u16, splat_idx)` keys → `RadixSort` → per-tile `[begin, end)`
  ranges. Tests with 4 splats in 4 tiles and 2 splats sharing a tile
  pin tile-ids, payload mapping, and depth ordering.
- **A2.5** `shaders/rasterize.wgsl` — per-tile front-to-back α-blend.
  One workgroup per 16×16 tile (256 threads = 256 pixels), splats
  loaded into workgroup-shared memory in chunks of 256 (amortising
  global storage reads over 256 pixels), standard "over" operator
  with `T < 1e-4` early-out. Single-splat test verifies analytic
  Gaussian falloff + monotonic α along the diagonal.
- **A2.6** End-to-end integration into `train()`. Per iteration:
  project → bin → composite into a `vec4` framebuffer, readback to
  CPU, compute MSE vs the baked reference view, periodic log. Iter-0
  dumps `train_predicted.png` / `train_reference.png` for visual
  smoke before sitting through a long run.

### Added — Phase A3: Backward pass

- **A3.1** `GradSplatBuffers` (6 `f32` SSBOs matching forward layout)
  + `ProjectedGrad` Pod (48-byte per-splat 2D-state gradient:
  `dmean.xy + dopacity`, `dconic.xyz`, `dcolor.xyz`, all padded to
  3·vec4). Zero helpers for re-use across iterations.
- **A3.2** `shaders/rasterize_backward.wgsl` — per-tile reverse
  α-blend. Walks the same sorted slice as forward but back-to-front,
  maintaining `T = T_{i+1}/(1−α_i)` and the accumulator
  `S = Σ_{j>i} T_j α_j c_j`. Per-pixel contributions accumulate into
  `projected_grad[splat]` via a CAS-based `atomic_add_f32` (WGSL has
  no native `f32` atomics). Numerical-stability fix:
  `if T > 1.0001 { done }` — once the divide-reconstruction overshoots
  the legitimate transmittance range, the pixel has crossed back into
  the forward early-out's "invisible prefix"; processing further is
  both wasteful and numerically harmful.
- **A3.3** `shaders/project_backward.wgsl` — per-splat backward.
  Chain rule through: sigmoid(opacity_logit), SH bands 0..3
  (basis-function gradients per coefficient), `conic = inv(Σ_2D)`
  (sandwich `dL/dΣ = −conic · dL/dconic · conic`),
  `Σ_2D = J·W·Σ_3D·W^T·J^T` (Jacobian sandwich both sides),
  `Σ_3D = R·diag(σ²)·R^T` (`dL/dR = 2·dL/dΣ_3D·R·S²`,
  `dL/d(σ_i²) = R_i^T · dL/dΣ_3D · R_i`), `σ_i = exp(log_σ_i)`,
  `R = quat_to_mat3(q)` with closed-form per-component derivatives,
  pinhole projection back through the view matrix.
- **A3.4** Finite-difference verification (gold-standard test).
  Single splat with known params, loss = Σ rgb pixels (uniform
  `dL/dC`), 5 parameters checked at `ε = 1e-3` central-difference
  within 5–10 % relative tolerance: `sh_dc.r`, `opacity_logit`,
  `sh_rest[0]`, `pos.x`, `log_scale.x`. Catches sign errors in any
  chain rule step.
- **A3.5** Integration into `train()`. Per iter after forward:
  compute `dL/dC = 2·(predicted − target)/(3·W·H)` on CPU, upload,
  zero `projected_grad` + `GradSplatBuffers`, run `composite_backward`
  then `project_backward`. Gradient-norm logging every 100 iters
  (`‖pos‖`, `‖opacity‖`, `‖sh_dc‖`, `‖scale‖`).

### Added — Phase A4: Adam updates (training learns)

- **A4.1** `GpuSplatBuffer::sync_from(ctx, &SplatBuffer)` — re-uploads
  an updated CPU `SplatBuffer` into the existing GPU buffers via
  `queue.write_buffer`. Avoids 3 MiB × 30k-iter realloc churn.
- **A4.2** Adam updates wired into `train()`. Per iter after backward:
  readback gradients, flatten to scalar slabs (strip `vec4` padding,
  `sh_rest` 48 → 45), step the existing `AdamState` per attribute,
  write updated slabs back into `SplatBuffer` with physical
  constraints (quaternion re-normalised to unit length, log-σ
  clamped to `[-10, 5]`, opacity-logit to `[-10, 10]`), then
  `gpu_splats.sync_from(...)`. Verified: 100-iter run on a 4860-splat
  scene moves MSE 0.6178 → 0.5870 with monotonic gradient-norm decay.

`L1+SSIM` photometric loss is parked under "Visible improvements" —
MSE works for first-cut convergence; SSIM needs a 5-pass Gaussian-blur
forward + analytic backward.

### Added — Phase A5: Densify-and-prune

- **A5.1** `prune_low_opacity` — every 100 iters past warmup, sweep
  splats with `opacity_logit < −5.3` (σ ≈ 0.005). `swap_remove`
  applied in lockstep to `SplatBuffer`, all 6 `AdamState` slabs, the
  `grad_acc` accumulator, and `gpu_splats.n`. Per-iter accumulates
  `|d_position| L1` into `grad_acc` for the densify gate.
- **A5.2** `densify` (clone + split) — every 200 iters past warmup.
  Candidates with `avg(|d_position|) > 2e-4`:
  - "small" splat (`max(log σ) < −2.3`) → CLONE: append duplicate
    with halved opacity, halve parent's opacity (preserves net
    emission while letting Adam separate them).
  - "large" splat (`max(log σ) ≥ −2.3`) → SPLIT: append 2 children
    sampled from `N(0, parent.Σ)` (Box-Muller + LCG seeded by
    `iter` → reproducible), scales reduced by `ln(1.6)`, remove
    parent.
  - `max_splats` cap enforced by skipping over-budget candidates.
- When the splat count changes, full GPU buffer realloc
  (`gpu_splats`, `projected`, `projected_grad`, `GradSplatBuffers`);
  otherwise `sync_from` is enough.
- Unit tests on both `prune_low_opacity` and `densify` verify all
  AdamState slabs end up the right length and the correct survivors
  remain.

### Added — Phase B1: Standalone interactive viewer

- New crate `crates/nano-view`. winit 0.30 `ApplicationHandler` app
  loop, `wgpu::Surface` configured against an sRGB swapchain format
  when available. Reuses `Rasterizer::project / composite` and
  `TileBinner::bin` verbatim — no duplicate kernels.
- Orbit camera: scene-centroid target, scene-bbox-derived default
  distance, LMB drag rotates azimuth / elevation, scroll-wheel zooms.
  ESC quits.
- Surface presentation: composite → readback `vec4<f32>` → CPU
  Reinhard tonemap + sRGB-encode (when the surface format isn't
  already an `*_SRGB`) → `queue.write_texture` to the swapchain →
  present. Simple, no new shaders; the natural follow-up is a
  GPU-only `tonemap.wgsl` + blit fragment pair to eliminate the
  per-frame CPU bounce.
- CLI: `nanotracer-rs --splats out.ply --view` (works with `--train`
  too) opens the viewer after generating / training the splats.

### Added — Bx.1 / Bx.2 / Bx.3 / Bx.4 — egui_dock viewer extensions

- **Bx.1 — egui_dock + GPU-only tonemap.** Viewer presentation
  rewritten on top of the `egui 0.34 / egui-wgpu 0.34 / egui-winit
  0.34 / egui_dock 0.19 / egui_plot 0.35` stack. New `tonemap.wgsl`
  compute kernel does Reinhard + sRGB encode straight into an
  `rgba8unorm` storage texture; egui samples it as a native texture
  in the `Viewport` dock tab. CPU readback path retired — the splat
  image stays on the GPU all the way to the swapchain. Inspector
  dock tab shows FPS, splat count, camera pose, FoV slider.
  Viewport tab rounds its requested size to multiples of the 16-pixel
  tile and the host reallocates render targets to match.
- **Bx.2 — PLY loader.** `nano_splat::read_ply(path)` parses the
  Inria binary-little-endian schema with a tolerant property-name
  mapping (handles minor reordering, missing optional `nx/ny/nz`,
  errors loudly on missing required fields). New
  `nanotracer-rs --view-ply scene.ply` fast path bypasses scene
  generation entirely. Round-trip test pins write_ply → read_ply
  bit-equal.
- **Bx.3 — Full camera controls.** Beyond LMB-orbit + scroll-zoom:
  RMB-drag pans the target in the camera's screen plane (magnitude
  scales with distance), WASD/QE fly mode moves the target along
  the camera frame (forward/right/up). Pointer-capture aware — egui
  consumes input first so dragging widgets doesn't move the camera.
- **Bx.4 — Live training preview.** `nano_view::run_with_training(
  scene, cfg)` spawns a worker thread that runs the full `train()`
  loop while the viewer event loop runs on the main thread; an
  `Arc<RwLock<Option<TrainSnapshot>>>` shuttles per-iteration
  splats + stats from worker to viewer. Per frame the viewer compares
  the snapshot's monotonic version, copies the new `SplatBuffer`
  into its GPU buffer (full realloc on densify count changes,
  `sync_from` otherwise), and pushes the new MSE into a history
  vec. New `Training` dock tab shows iter / MSE / splat-count plus
  an `egui_plot::Line` of the loss curve.

  `train()` signature gained an `on_iter: FnMut(u32, &SplatBuffer, f32)`
  callback parameter — headless callers pass a no-op closure. CLI
  flag `--view-training` opens the viewer with live preview; ESC
  closes the window and detaches the worker (which finishes the
  remaining iterations on its own).

### Tracked / Parked

- **A4-ext** — `0.2·L1 + 0.8·DSSIM` perceptual loss (5-pass Gaussian
  blur kernel + analytic backward). Quality knob, not a correctness
  gate; MSE alone converges.

### Infrastructure

- Default wgpu device limits raised to `adapter.limits()` in
  `nano-optimize::WgpuCtx` — `project_backward` binds 13 storage
  buffers per stage, exceeding the conservative default of 8.
- Workspace now has 9 crates: `gpu-mem`, `nano-{core, io, shaders,
  gpu, render, splat, optimize, view}` (`splat-ref` retained as
  source-only documentation).
- 45 workspace unit tests + 1 finite-difference integration test pass;
  `cargo clippy --workspace --all-targets` clean.
