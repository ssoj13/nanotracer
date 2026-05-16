# Overview

`nanotracer-rs` is an end-to-end 3D-Gaussian-splat (3DGS) training and
viewer system in Rust. The codebase ships three independently useful
modes:

1. **Raytrace renderer** — GPU path tracer with Vulkan ray queries,
   GGX microfacets, EXR + procedural environments, area lights.
2. **Forward-fit + train** — generate Gaussian splats from a scene's
   geometry, then run the differentiable rasteriser + Adam loop until
   the splat field matches multi-view reference frames.
3. **Interactive viewer** — open any trained PLY, or attach to a live
   training run and watch splats converge with a side-panel loss curve.

This book is a technical reference, not a tutorial. Chapters cover the
math of each pass, the structural decisions, and the open follow-ups.

## Reading order

- New to the codebase: [Origin & evolution](./history.md) → [Workspace
  layout](./architecture.md) → [Plan A overview](./training-overview.md).
- Modifying the trainer: [Forward](./training-forward.md) →
  [Backward](./training-backward.md) → [Adam + densify](./training-adam.md).
- Modifying the viewer: [Viewer architecture](./viewer.md).
- Picking up where the project stops: [Roadmap & TODO](./roadmap.md).

## Conventions

- WGSL kernels live in `crates/nano-optimize/src/shaders/` and
  `crates/nano-view/src/shaders/` (when added); GLSL chunks live in
  `crates/nano-shaders/src/lib.rs`.
- Matrices follow `glam`'s column-major / right-handed view convention
  (`Mat4::look_at_rh`).
- "Camera-space depth" means **positive into the scene** (we flip the
  view-matrix Z inside projection so Inria-derived formulas drop in).
- Splats store rotation as `(w, x, y, z)`; `glam::Quat` stores
  `(x, y, z, w)`. Conversion is explicit at every boundary — see
  [Shader catalog](./shaders.md).

## Building & running

See [`README.md`](../../README.md) at the repo root. The book itself
builds with:

```bash
cargo install mdbook
mdbook serve docs --open   # live-reload on http://localhost:3000
```
