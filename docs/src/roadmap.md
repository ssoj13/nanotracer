# Roadmap & TODO

Everything not yet built, organised by priority.

## 🟧 Visible improvements

Things a user would notice if implemented; clear scope.

### A4-ext — `0.2·L1 + 0.8·DSSIM` loss

Inria's reference recipe. Five-pass Gaussian blur over predicted +
target produces `μ_p, μ_t, σ_p², σ_pt, σ_t²`; per-pixel SSIM
combines them; loss = `1 - mean(SSIM)`. Analytic backward chains
back to per-pixel `dL/dC`.

Effort: ~200 lines WGSL (blur + SSIM combine + backward) + ~50
lines Rust glue. Tracked because perceptual fidelity > MSE for the
final 20% of training quality, but training already converges on
MSE — this is a quality knob, not a gating issue.

### Pure-GPU Adam step

CPU Adam costs ~3–5 ms of readback / upload per iteration. A GPU
kernel reading `(params, grads, m, v)` and applying Adam in-place
would eliminate the bounce.

Sketch:

```wgsl
@compute @workgroup_size(64)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    let g = grads[gid.x];
    var m = moments_m[gid.x];
    var v = moments_v[gid.x];
    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g * g;
    let m_hat = m / (1.0 - pow(beta1, f32(t)));
    let v_hat = v / (1.0 - pow(beta2, f32(t)));
    params_buf[gid.x] -= lr * m_hat / (sqrt(v_hat) + eps);
    moments_m[gid.x] = m;
    moments_v[gid.x] = v;
}
```

Six independent dispatches (one per attribute) or one combined
kernel with branchy logic. The quaternion re-norm and log-σ /
opacity-logit clamps would also need to live in WGSL — that's the
non-trivial part.

### GPU densify

Move `prune` and `densify` from CPU to GPU. Per-attribute
`swap_remove` becomes a stream compaction (prefix-scan over
keep-mask, scatter). Clone / split can run as a single dispatch
per pass. Same `Adam` slabs need parallel resize.

Big win for very dense scenes (10M+ splats) where the per-iteration
CPU readback dominates.

### Bookkeeping HUD widgets

Current Inspector tab is minimal. Useful additions:

- VRAM usage bar (we already have `gpu-mem`).
- Per-iteration timing breakdown (project / bin / composite / etc.).
- Densify event log (when did things split / clone, parent stats).
- Save-button to dump the current `SplatBuffer` to PLY.

## 🟨 Polish

Smaller refinements; not user-facing on their own.

- **Tab close / float / spawn buttons** in `egui_dock`. Currently
  the layout is fixed at startup; dragging tabs around works but
  closing the Inspector loses it forever.
- **Camera presets** (front / top / 3/4 / save current). Right-
  click → "save view" → bind to numeric key.
- **Persisted layout** between launches. egui already serialises
  `DockState` — write to `~/.config/nano-view/layout.ron`.
- **`pump_app_events` mode for B2** as an opt-in alternative to the
  worker thread. Useful when GPU concurrency between trainer and
  viewer is more contention than threading is worth.
- **Render command-queue timings** via `wgpu::QuerySet` (timestamp
  queries) — display per-pass GPU time in the HUD.
- **Configurable training hyperparameters in the Training tab.**
  Adam learning rates, prune / densify intervals — adjust live.

## 🟩 Long horizon

Aspirational; only matter if specific use cases come up.

### `wgpu` consolidation (one runtime end-to-end)

Re-evaluate when:

1. `wgpu` ships `RAY_QUERY` outside an `EXPERIMENTAL_*` flag.
2. WGSL ray-query syntax is part of an accepted WebGPU spec PR.
3. DX12 backend reaches feature parity with Vulkan for AS build.

When all three hold, `nano-render` / `nano-gpu` / `nano-splat` can
migrate off `ash` + `shaderc`. The win is a single GPU runtime,
loss of the `VULKAN_SDK` build dependency on Windows, and a path
to WebGPU.

### Web target

Run the viewer in a browser via WebGPU. Mostly mechanical once
`wgpu` consolidation lands:

- `nano-optimize` and `nano-view` already build on `wasm32` with
  the right `wgpu` features.
- `winit` has a wasm backend (canvas).
- `egui` has `egui-wgpu` for the web.
- PLY loading would need a `FileReader` async path. Training would
  need a workgroup-shared-memory check (some WebGPU implementations
  have stricter limits).

### `no-std` core

Lift `nano-core` to `#![no_std]` for ARM SBC targets. Path:

- Replace `Box<dyn Error>` returns with `&'static str`.
- Replace `std::Vec` use in `Mesh` etc. with `alloc::Vec` behind
  a feature flag.
- Keep `nano-gpu` / `nano-shaders` / `nano-render` host-only.

### Higher-quality splat fitter

The forward fit produces 1–5k splats from procedural scenes. Real
3DGS pipelines train on photographs and produce hundreds of
thousands. A future direction: take real images / video as input
(SfM via something like `colmap-rs` would be nice but is its own
project) and use them as reference frames for training, replacing
the procedural-scene seed step.

### Per-splat material parameters in the training loop

The forward fit captures `(pos, rotation, scale, SH, opacity)`.
Reconstructing materials directly (`kd, ks, kr, kt` per splat) and
training a deferred BRDF rather than purely view-dependent
radiance would let the trained scene relight under a different
environment map. Significant scope; would integrate naturally with
the existing material system.

## Parked questions

- **Are `test_hi.cmd` / `test_lo.cmd` still the right quality
  presets after the physical-materials change?** Re-tune if
  visual feedback says yes.
- **Should `splat-ref/` ever be wired back into the workspace?**
  Currently kept as browsable documentation. Re-evaluate when
  porting any other CPU helper.
- **Stable random-seed mode for densify?** LCG seeded by `iter`
  is reproducible but doesn't decouple from iteration count. A
  separate seed flag would let A/B tests vary one variable at a
  time.
- **Should the worker thread in B2 share the viewer's wgpu device?**
  Would halve VRAM use for buffers and skip one `Instance::new`.
  wgpu's `Device` and `Queue` are `Send + Sync` so it's possible.
  Currently separate for clearer lifetime ownership.

## Tasks completed (kept for context)

See `CHANGELOG.md` for the full chronological record. Highlights:

- ✅ Plan A: A1 scaffolding → A2 forward rasteriser → A3 backward +
  FD verify → A4 Adam updates → A5 densify-and-prune. Training
  works end-to-end with MSE.
- ✅ Plan B: B1 standalone viewer → Bx.1 egui_dock + GPU-only
  tonemap → Bx.2 PLY loader → Bx.3 RMB-pan + WASD/QE fly → Bx.4
  live training preview with `egui_plot` loss curve.
- ✅ Area lights + IBL as `Light::Env` (replaces `--ibl-strength`).
- ✅ GGX shading + multi-scattering compensation (replaces Phong).
- ✅ Workspace reorg into 9 explicit crates with enforced
  dependency edges.
