# AGENTS.md — codepath & dataflow map

Compact dataflow map for agent triage. Full explanations live in the
mdbook at `docs/src/` — start at `architecture.md`, `training-overview.md`,
`viewer.md`. Mermaid diagrams in `DIAGRAMS.md`.

---

## Workspace at a glance (current)

```
nanotracer-rs/
├─ src/main.rs       CLI dispatch (clap)
└─ crates/
   ├─ gpu-mem/       std-only VRAM / RAM probe
   ├─ nano-core/     Scene + Light enum + Material + Environment + SH
   ├─ nano-io/       glTF loader, PNG writer
   ├─ nano-shaders/  GLSL chunks (PREAMBLE + HELPERS + sample_light)
   ├─ nano-gpu/      ash runtime, GpuLight 64B, scene → SSBO
   ├─ nano-render/   raytrace renderer compute pipeline (ash)
   ├─ nano-splat/    forward-fit splat generator + PLY r/w (ash)
   ├─ nano-optimize/ wgpu differentiable rasteriser + Adam + densify
   └─ nano-view/     winit + wgpu surface + egui_dock viewer
```

Concrete edges:

```
bin                 → nano-{core,io,render,splat,optimize,view}, gpu-mem
nano-render         → nano-{core,gpu,shaders}
nano-splat          → nano-{core,gpu,shaders}
nano-optimize       → nano-{core,io,render,splat}
nano-view           → nano-{core,optimize,splat}
nano-gpu, nano-io   → nano-core
nano-shaders, gpu-mem → (none)
```

`splat-ref/` lives under `crates/` but is intentionally excluded from
the workspace (pre-refactor namespace, browsable docs only).

---

## Top-level CLI dispatch

```
main.rs::main
  ├── if --view-ply <PATH>:
  │     nano_splat::read_ply → SplatBuffer → nano_view::run
  │
  ├── if --view-training:
  │     scene + train_cfg → nano_view::run_with_training
  │                          (spawns worker → nano_optimize::train)
  │
  ├── if --splats <PATH> && --train:
  │     nano_optimize::train(scene, cfg, noop callback)
  │       └→ Gaussians → write_ply
  │       └→ if --view: nano_view::run
  │
  ├── if --splats <PATH>:
  │     nano_splat::generate_splats_gpu
  │       └→ write_ply
  │       └→ if --view: nano_view::run
  │
  └── else:
        nano_render::render → save_image (PNG)
```

---

## Per-iteration training loop (nano-optimize::train)

```
ReferenceView (baked once) ──┐
SplatBuffer + 6 AdamState  ──┤
                              │
   per iter ┌─────────────────▼────────────────────────────┐
            │ project_gaussians.wgsl                       │
            │   ↓                                          │
            │ tile_count → PrefixScan → tile_emit →        │
            │ RadixSort (32 × 1-bit-stable) → tile_ranges  │
            │   ↓                                          │
            │ rasterize.wgsl (per-tile α-blend)            │
            │   ↓                                          │
            │ readback predicted → CPU MSE + dL/dC         │
            │   ↓                                          │
            │ rasterize_backward.wgsl (atomic_add_f32)     │
            │   ↓                                          │
            │ project_backward.wgsl                        │
            │   ↓                                          │
            │ readback all 6 grad buffers                  │
            │   ↓                                          │
            │ CPU AdamState::step × 6                      │
            │   ↓                                          │
            │ apply constraints (quat-norm, log-σ + opa    │
            │   logit clamps)                              │
            │   ↓                                          │
            │ accumulate |d_position| L1 into grad_acc     │
            │   ↓                                          │
            │ if iter % 100: prune_low_opacity             │
            │ if iter % 200: densify (clone + split)       │
            │   ↓                                          │
            │ gpu_splats.sync_from (or full realloc on n   │
            │ change)                                      │
            │   ↓                                          │
            │ on_iter(iter, &splats, mse)  ← live viewer   │
            └──────────────────────────────────────────────┘
```

Full math: see `docs/src/training-{overview,forward,backward,adam}.md`.

---

## Viewer per-frame loop (nano-view::frame)

```
winit event ──┬──► egui_winit::on_window_event (UI input)
              ├──► orbit camera state (LMB / RMB / WASD)
              │
              └── RedrawRequested ──►
                    │ Optional: poll training snapshot RwLock,
                    │ on version change → State::update_splats
                    │   (sync_from or realloc if n changed)
                    ▼
                  Rasterizer::project
                    ▼
                  TileBinner::bin
                    ▼
                  Rasterizer::composite       (vec4f buffer)
                    ▼
                  Rasterizer::tonemap         (rgba8unorm texture)
                    ▼
                  egui_ctx::run (DockArea → Viewport / Inspector / Training)
                    ▼
                  egui_renderer::render to surface
                    ▼
                  surface.present
```

Full architecture: see `docs/src/viewer.md`.

---

## Scene → GPU marshalling (nano-gpu::gpu_scene)

```
nano_core::Scene { objects, lights, environment }
   │
   │ for each Object: Mesh → append_mesh
   │ for the procedural checkerboard (if enabled)
   ▼
build_gpu_scene_with_detail_boost
   │
   ├─► vertices [vec4], normals [vec4], triangles [uvec4]
   ├─► tri_materials [u32], tri_cdf [f32], tri_areas [f32]
   ├─► materials [GpuMaterial]
   ├─► lights [GpuLight 64B], light_radiance [vec4]   ← Light enum dispatch
   └─► light_count u32 (logical, before zero-pad fallback)
        │
        ▼
   nano_gpu::vk_runtime  ── SSBO/UBO/AS uploads
```

Implicit `Light::Env { intensity: 1.0 }` added when `scene.environment.is_some()
&& !scene.has_env_light()`.

---

## Shader inventory

GLSL chunks live in `nano-shaders::{PREAMBLE, HELPERS}`. Per-shader
bindings + bodies are inlined in `nano-render` and `nano-splat`. WGSL
files live in `crates/nano-optimize/src/shaders/`.

Full catalog: see `docs/src/shaders.md`.

---

## Recent decision points (one-liners)

| Where to look | Decision |
|---------------|----------|
| `docs/src/decisions.md` | Why ash vs wgpu split, why 1-bit stable radix, why CAS atomic add f32, why `Light::Env` instead of `--ibl-strength`. |
| `docs/src/training-backward.md` | Why `T > 1.0001` early-bail (numerical-stability fix). |
| `docs/src/lights.md` | Per-variant sampling math. |
| `docs/src/training-adam.md` | Adam stride per attribute + densify policy. |
| `CHANGELOG.md` | Chronological narrative of every milestone. |
