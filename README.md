# nanotracer-rs

GPU Gaussian-splat training + viewer in Rust. End-to-end pipeline from
scene description (or HDRI + meshes) to a trained 3DGS PLY, with an
interactive `egui_dock` viewer that doubles as a live training preview.

- **Forward path:** Vulkan ray queries (hardware BVH) → procedural
  scene → forward-fit Gaussian splats with order-3 SH.
- **Training:** `wgpu` differentiable rasteriser (project + tile-bin +
  α-blend), Adam updates, densify + prune. Forward-only takes ~1 s for
  ~5k splats; training scales linearly per iteration.
- **Viewer:** winit + `wgpu` surface + `egui_dock`. Dockable Viewport
  + Inspector + Training tabs; orbit / RMB-pan / WASD fly; live MSE
  plot via `egui_plot` during training.

Originally a Rust port of
[tinyraytracer](https://github.com/ssloy/tinyraytracer); the CPU
pipeline has long since been replaced by the GPU stack described here.

![Screenshot](data/splat.jpg)

## Quick start

```bash
# Headless render
cargo run --release

# Forward-fit a procedural scene → PLY
cargo run --release -- -S out.ply --splat-density 200

# Open the viewer after the fit
cargo run --release -- -S out.ply --view

# Open a pre-trained PLY directly (Inria binary schema)
cargo run --release -- --view-ply scene.ply

# Train a scene from scratch
cargo run --release -- -S out.ply --train --train-iters 30000

# Live training preview — viewer + worker thread + MSE plot
cargo run --release -- -S out.ply --view-training --train-iters 30000
```

ESC closes the viewer. LMB drag orbits, RMB drag pans, WASD/QE fly,
scroll-wheel zooms. The Inspector tab carries an FoV slider, the
Training tab shows live iter / MSE / splat count + a loss curve.

## CLI summary

The full flag list is in `--help`. Highlights:

| Flag | Effect |
|------|--------|
| `-S/--splats <PATH>` | Export splats to PLY (3DGS-compatible). |
| `--view`            | Open the viewer after generation / training. |
| `--view-ply <PATH>` | Open the viewer on an existing PLY (no scene). |
| `--view-training`   | Live preview during training. |
| `--train`           | Run the Adam loop after the forward fit. |
| `--train-iters N`   | Total training iterations. |
| `--train-max-splats N` | Hard cap on splat count after densify. |
| `--mesh TYPE`       | `cube` / `pyramid` / `torus` / `all` / `none`. |
| `--glb <PATH>`      | Load a glTF / GLB mesh into the scene. |
| `-e/--env <PATH>`   | EXR HDR environment map. |
| `--env-light F`     | IBL intensity (replaces the old `--ibl-strength`). |
| `--rect-light cx,cy,cz,ux,uy,uz,vx,vy,vz,r,g,b,i[,two]` | Add a rectangle area light. |
| `--sphere-light cx,cy,cz,radius,r,g,b,i` | Add a sphere area light. |
| `--box-light cx,cy,cz,hx,hy,hz,qx,qy,qz,qw,r,g,b,i` | Add an oriented-box light. |
| `--point-light x,y,z,r,g,b,i` | Add a point light. |

The training loop dumps `train_predicted.png` and `train_reference.png`
at iter 0 so you can sanity-check the rasteriser before a long run.

## Workspace layout

```
nanotracer-rs/
├─ Cargo.toml          (workspace + bin)
├─ src/main.rs         (CLI dispatch)
├─ crates/
│  ├─ gpu-mem/         (cross-platform VRAM / RAM probe — std-only)
│  ├─ nano-core/       (Scene, Light enum, Material, EnvironmentMap, SH)
│  ├─ nano-io/         (glTF loader, PNG writer)
│  ├─ nano-shaders/    (PREAMBLE + HELPERS GLSL — shared by raytrace + splat shaders)
│  ├─ nano-gpu/        (ash + shaderc, GpuLight 64-byte std430, scene → GPU buffers)
│  ├─ nano-render/     (raytrace renderer — ash compute shader, ray-query)
│  ├─ nano-splat/      (forward-fit splat generator, binary 3DGS PLY r/w)
│  ├─ nano-optimize/   (wgpu differentiable rasteriser, Adam, densify, training loop)
│  └─ nano-view/       (egui_dock viewer + live training preview)
```

`nano-render` + `nano-splat` + `nano-gpu` run on `ash` because they
need stable hardware ray queries (`VK_KHR_ray_query`). `nano-optimize`
+ `nano-view` run on `wgpu` — the rasteriser is portable, and the
viewer can target DX12 / Metal / Vulkan / WebGPU. See `WGPU_RESEARCH.md`
for the migration calculus.

## Build prerequisites

- Rust **edition 2024** toolchain.
- Vulkan SDK for `shaderc-sys`. Point `VULKAN_SDK` at it, e.g.

  ```pwsh
  $env:VULKAN_SDK = "C:\Programs\VulkanSDK\1.4.341.1"
  ```

- A GPU with `VK_KHR_acceleration_structure` + `VK_KHR_ray_query`
  (any RTX 20-series / RDNA2+ / Arc / Apple GPU via MoltenVK).
- For the viewer: any wgpu-supported surface (Vulkan, DX12, Metal,
  GL — wgpu picks).

## Documentation

- **`CHANGELOG.md`** — milestone-grouped change log; the
  end-to-end Plan A (training) + Plan B (viewer) story.
- **`AGENTS.md`** — codepath / dataflow map for agent triage.
- **`DIAGRAMS.md`** — mermaid versions of the AGENTS diagrams.
- **`todo.md`** — active roadmap + parked questions.
- **`WGPU_RESEARCH.md`** — ash-vs-wgpu evaluation, decision log.
- **`docs/` (mdbook)** — technical reference: math derivations,
  shader-by-shader pipeline, training loop internals.

  ```bash
  cargo install mdbook
  mdbook serve docs --open
  ```

## Shading model (raytrace + splat-fit paths)

Both the image renderer and the forward-fit splat shader run the same
shading code via `nano-shaders::{PREAMBLE, HELPERS}`:

- **Diffuse:** Lambertian, `k_d · diffuse_color · (N·L)` per light.
- **Specular:** GGX / Trowbridge–Reitz with Smith geometry + Schlick
  Fresnel (F₀ = ks). Legacy `specular_exponent` maps to GGX α via
  `α = √(2/(n+2))`. Multi-scattering compensation
  (`ggx_msc_boost`, Turquin 2018) keeps high-roughness energy.
- **Reflection / refraction:** weighted by `kr` / `kt`. For dielectrics
  with both > 0 (e.g. `GLASS`) a per-hit Schlick term rebalances; at
  grazing the surface becomes mirror-like.
- **IBL:** `Light::Env { intensity }` carries a pre-convolved degree-2
  SH (Ramamoorthi–Hanrahan) sampled per surface normal. Area lights
  (rect / sphere / box) are full Monte-Carlo with proper PDF + visibility.

## Training pipeline (Plan A — `nano-optimize`)

End-to-end on `wgpu`, mirroring the Inria 3DGS recipe:

1. Bake N reference views via the raytrace renderer (`bake_references`).
2. Forward-fit seed splats via `generate_splats_gpu`.
3. Per iteration:
   - `project_gaussians.wgsl`: 3D Σ → 2D conic + depth + radius + SH eval.
   - `tile_count.wgsl` → `PrefixScan` → `tile_emit.wgsl` → `RadixSort`
     (1-bit-at-a-time stable, 32 passes) → `tile_ranges.wgsl`.
   - `rasterize.wgsl`: per-tile 16×16 workgroup α-blends sorted splats.
   - Loss `dL/dC = 2·(predicted − target)/(3·W·H)`.
   - `rasterize_backward.wgsl` + `project_backward.wgsl`: gradients
     through α-blend, sigmoid, SH basis, conic-inverse sandwich,
     Σ_3D ↔ Σ_2D Jacobian sandwich, quaternion → rotation, log-σ.
   - Adam step on CPU per attribute; sync back to GPU.
   - Every 100 iters: prune low-opacity splats.
   - Every 200 iters: densify (clone small / split large).

Finite-difference verification (`tests/backward_finite_diff.rs`) pins
every chain-rule step to ≤ 5–10 % relative error at ε = 1e-3.

## Viewer pipeline (Plan B — `nano-view`)

- winit 0.30 + `wgpu::Surface` + `egui_dock`.
- GPU-only presentation: composite `vec4<f32>` buffer →
  `tonemap.wgsl` → `rgba8unorm` storage texture → egui's
  `Image` widget → swapchain.
- Dockable tabs: Viewport (splat image, resizes the render target
  to match), Inspector (FPS / camera / FoV slider), Training (live
  MSE plot via `egui_plot`).
- Training preview spawns the training loop on a worker thread; an
  `Arc<RwLock<Option<TrainSnapshot>>>` shuttles per-iteration
  splats + stats between threads. Viewer re-uploads to its
  `GpuSplatBuffer` every frame (full realloc on densify count
  changes, `sync_from` otherwise).

## Materials

Energy-conserving `[kd, ks, kr, kt]` albedo (`Σ ≤ 1`). `ks` is GGX F₀
(≈ 0.04 for typical dielectrics).

| Material | kd | ks (F₀) | kr | kt | n → α | Look |
|---|---|---|---|---|---|---|
| `IVORY` | 0.85 | 0.04 | 0.10 | 0.00 | 50 → 0.20 | Warm off-white, soft highlight |
| `GLASS` | 0.00 | 0.04 | 0.10 | 0.85 | 300 → 0.082 | Fresnel-blended dielectric |
| `RED_RUBBER` | 0.90 | 0.04 | 0.00 | 0.00 | 10 → 0.41 | Saturated matte red |
| `MIRROR` | 0.00 | 0.04 | 0.96 | 0.00 | 1500 → 0.037 | Near-perfect metallic mirror |
| `MATTE_*` | 0.95 | 0.04 | 0.00 | 0.00 | 20 → 0.30 | Pure matte family |

## Dependencies

`ash`, `shaderc`, `wgpu`, `egui` (+ `egui-wgpu`, `egui-winit`,
`egui_dock`, `egui_plot`), `winit`, `glam`, `bytemuck`, `pollster`,
`exr`, `gltf`, `image`, `indicatif`, `clap`, `fastrand`.

## License

MIT.
