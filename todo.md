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

- [x] Normalised Phong specular `(n+2)/(2π)` in both shaders. Materials
      rebalanced so `ks ≈ F₀ ≈ 0.04` (energy-as-stored).
- [x] `--sh-keep-glossy` CLI flag wired through `SplatConfigGpu` and the
      splat shader uniform block.
- [x] Schlick Fresnel reflection/refraction split for dielectric
      materials (`kr > 0 && kt > 0`), e.g. `GLASS`.

## Active

_nothing in flight — pick from the parked list below_

## 🟧 Visible improvements

## 🟨 Polish

- [ ] Spawn a coarser SH (degree 1) variant for surfaces flagged
      "mostly diffuse" — saves storage and DC quality stays the same.
- [ ] Detail-boost heuristic in `gpu_scene::append_mesh` is currently
      a one-shot variance gauge — try a curvature-aware variant once
      we have measured timings.
- [ ] Progress bar fidelity: separate buffer-upload vs AS-build vs
      pipeline-build steps when timings are wildly different.

## 🟩 Long horizon

- [ ] Full PBR (GGX / Cook–Torrance + multi-scattering compensation).
      Big shader rewrite — pencilled for when materials need texturing
      support too.
- [ ] Real environment SH convolution per splat instead of "DC only"
      for reflective/refractive — captures average env reflection
      without per-direction ringing.
- [ ] `wgpu` migration. Blocked on RT extension stabilising; see
      `WGPU_RESEARCH.md` for the unblock criteria.
- [ ] `no-std` Rust subset for ARM SBC targets — moved here from the
      previous freeform note. Path: lift `nano-core` to `#![no_std]`,
      keep `nano-gpu`/`nano-shaders` host-only.

## Parked questions

- Are the test scripts (`test_hi.cmd`, `test_lo.cmd`) the right
  default-quality presets after the physical-materials change? Re-tune
  if visual feedback says yes.
- Should `splat-ref/` ever be wired back into the workspace? Decision
  so far: no — keep as reference. Re-evaluate when porting tests for
  any other CPU helper.
