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

_nothing in flight — pick from the parked list below_

## 🟧 Visible improvements

_(empty — all landed)_

## 🟨 Polish

_(empty — all landed)_

## 🟩 Long horizon

- [ ] Multi-scattering compensation for GGX (Heitz / Hill) — current
      single-scatter BRDF is a touch dim at high roughness.
- [ ] Per-roughness env-map mip chain for IBL specular (split-sum
      approximation à la Real Shading in UE4). Today only IBL **diffuse**
      is wired; specular reflections still come from `trace_path`.
- [ ] `wgpu` migration. Blocked on RT extension stabilising; see
      `WGPU_RESEARCH.md` for the unblock criteria.
- [ ] `no-std` Rust subset for ARM SBC targets. Path: lift `nano-core`
      to `#![no_std]`, keep `nano-gpu`/`nano-shaders` host-only.

## Parked questions

- Are the test scripts (`test_hi.cmd`, `test_lo.cmd`) the right
  default-quality presets after the physical-materials change? Re-tune
  if visual feedback says yes.
- Should `splat-ref/` ever be wired back into the workspace? Decision
  so far: no — keep as reference. Re-evaluate when porting tests for
  any other CPU helper.
