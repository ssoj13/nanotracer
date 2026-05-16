# DIAGRAMS.md — mermaid flow diagrams

Companion to `AGENTS.md`. Mermaid versions of every diagram for
GitHub / IDE rendering. Full prose in the mdbook (`docs/src/`).

---

## 1. CLI dispatch

```mermaid
flowchart TD
    A[src/main.rs] --> S[build Scene]
    S --> SPL{flags?}

    SPL -->|--view-ply| LD[nano-splat::read_ply]
    LD --> VW1[nano-view::run]

    SPL -->|--view-training| RT[nano-view::run_with_training]
    RT --> WT[worker thread]
    WT --> TR[nano-optimize::train + callback]

    SPL -->|--splats + --train| TRN[nano-optimize::train]
    TRN --> WPLY[write_ply]
    WPLY -->|--view| VW2[nano-view::run]

    SPL -->|--splats| FF[nano-splat::generate_splats_gpu]
    FF --> WPLY

    SPL -->|else| RR[nano-render::render]
    RR --> PNG[save_image PNG]
```

## 2. Crate dependency graph

```mermaid
flowchart TD
    bin[bin nanotracer-rs] --> nc[nano-core]
    bin --> nio[nano-io]
    bin --> nr[nano-render]
    bin --> ns[nano-splat]
    bin --> no[nano-optimize]
    bin --> nv[nano-view]
    bin --> gm[gpu-mem]

    nr  --> nc & ng & nshd
    ns  --> nc & ng & nshd
    no  --> nc & nio & nr & ns
    nv  --> nc & no & ns
    ng  --> nc
    nio --> nc

    ng[nano-gpu]
    nshd[nano-shaders]

    classDef ash fill:#3a2a4a,color:#fff,stroke:#88f
    classDef wgpu fill:#1f4a4a,color:#fff,stroke:#4fc
    classDef pure fill:#1f4a1f,color:#fff,stroke:#0a0
    class ng,nr,ns ash
    class no,nv wgpu
    class nc,nio,nshd,gm pure
```

Pure (green) = no GPU runtime. ash (purple) = needs Vulkan + ray
queries. wgpu (teal) = portable compute / surface.

## 3. Scene → GPU buffers

```mermaid
flowchart LR
    SC[nano-core::Scene] --> BG[nano-gpu::gpu_scene::build]
    BG --> V[vertices vec4]
    BG --> N[normals vec4]
    BG --> T[triangles uvec4]
    BG --> TM[tri_materials u32]
    BG --> CDF[tri_cdf f32 normalised]
    BG --> M[materials GpuMaterial]
    BG --> L[lights GpuLight 64B]
    BG --> LR[light_radiance vec4]
    V & N & T & TM & CDF & M & L & LR --> VK[Vulkan SSBO/UBO buffers]
    V & T --> AS[BLAS + TLAS via vk_runtime]
```

## 4. Training pipeline (one iteration)

```mermaid
flowchart TD
    splats[SplatBuffer + AdamStates] --> upload[GpuSplatBuffer::sync_from]
    upload --> proj[project_gaussians.wgsl]
    proj --> tcnt[tile_count.wgsl + atomic total]
    tcnt --> rb[CPU readback total]
    rb --> scan[PrefixScan three-level]
    scan --> emit[tile_emit.wgsl]
    emit --> radix[RadixSort 32×1-bit stable]
    radix --> ranges[tile_ranges.wgsl]
    ranges --> rast[rasterize.wgsl per-tile α-blend]
    rast --> readback[readback predicted vec4]
    readback --> loss[CPU MSE + dL/dC]
    loss --> upload2[write dL/dC to GPU]
    upload2 --> bwd1[rasterize_backward.wgsl atomic_add_f32]
    bwd1 --> bwd2[project_backward.wgsl]
    bwd2 --> readback2[readback 6 grad buffers]
    readback2 --> adam[CPU AdamState::step × 6]
    adam --> constr[apply constraints quat-norm log-σ clamp]
    constr --> grad_acc[accumulate per-splat grad_acc]
    grad_acc --> maint{iter % 100 / 200?}
    maint -->|every 100| prune[prune_low_opacity]
    maint -->|every 200| densify[densify clone + split]
    maint -->|neither| skip
    prune --> sync
    densify --> sync
    skip --> sync[sync_from or realloc]
    sync --> cb[on_iter callback]
    cb --> splats
```

## 5. Viewer per-frame loop

```mermaid
flowchart LR
    winit[winit events] --> egui_in[egui_winit::on_window_event]
    winit --> cam[orbit / WASD / pan]
    cam --> camu[CameraUniform]
    train[Optional: TrainSnapshot poll] -->|version > last_seen| upd[State::update_splats]
    upd --> camu
    camu --> proj[Rasterizer::project]
    proj --> bin[TileBinner::bin]
    bin --> comp[Rasterizer::composite vec4f]
    comp --> tone[Rasterizer::tonemap rgba8unorm]
    tone --> egui_ui[egui_ctx::run DockArea]
    egui_in --> egui_ui
    egui_ui --> egui_render[egui_renderer::render]
    egui_render --> present[surface.present]
```

## 6. Light enum dispatch

```mermaid
flowchart TD
    Light --> kind{kind}
    kind -->|Point|  P[L_e &middot; cos_x]
    kind -->|Rect|   R[L_e &middot; cos_x &middot; cos_y &middot; A / r²]
    kind -->|Box|    B[same as Rect over 6 faces]
    kind -->|Sphere| S[solid-angle MC, area fallback inside]
    kind -->|Env|    E[eval_env_irradiance N &middot; intensity]
    P & R & B & S --> shadow[shadow_ray]
    E -.skip shadow.-> diffuse
    shadow --> diffuse[diffuse_radiance + specular_radiance]
```

## 7. Backward T-reconstruction (with stability clamp)

```mermaid
flowchart LR
    forward["forward early-out at T &lt; 1e-4"] --> stored[1 - T_final stored in output.w]
    stored --> bwd[backward read T_final = 1 - coverage]
    bwd --> walk[walk splats back-to-front]
    walk --> divide[T = T / 1-α]
    divide --> check{T &gt; 1.0001?}
    check -->|yes| done[done, gradient zero from here back]
    check -->|no| contrib[accumulate gradients]
    contrib --> walk
```

## 8. Quick reference — which crate owns what

| Concern | Crate | Key types / fns |
|---|---|---|
| Scene + geometry | `nano-core` | `Scene`, `Object`, `Geometry`, `Mesh`, `Light`, `EnvironmentMap` |
| Materials | `nano-core::material` | `Material`, `IVORY`, `GLASS`, `MIRROR`, `MATTE_*` |
| Colour pipeline | `nano-core::color` | `tonemap_reinhard`, `linear_to_srgb` |
| CPU SH reference | `nano-core::sh` | `sh_basis`, `fit_sh`, `eval_sh` |
| Light enum | `nano-core::scene::Light` | `Point`/`Rect`/`Sphere`/`Box`/`Env` |
| glTF / PNG | `nano-io` | `load_glb_mesh`, `save_image` |
| GLSL chunks | `nano-shaders` | `PREAMBLE`, `HELPERS`, `sample_light`, `assemble` |
| Vulkan runtime | `nano-gpu::vk_runtime` | `VkContext` |
| Scene → GPU | `nano-gpu::gpu_scene` | `build_gpu_scene*`, `GpuLight` 64B |
| Image renderer | `nano-render` | `render`, `RenderConfig` |
| Forward-fit splat | `nano-splat::generator` | `generate_splats_gpu`, `SplatConfigGpu` |
| Splat PLY r/w | `nano-splat::ply` | `read_ply`, `write_ply`, `Gaussian` |
| wgpu context | `nano-optimize::gpu` | `WgpuCtx` |
| GPU splat buffer | `nano-optimize::splat_gpu` | `GpuSplatBuffer`, `GradSplatBuffers` |
| Rasteriser passes | `nano-optimize::raster` | `Rasterizer` (project/composite/backward/tonemap), `ProjectedSplat`, `ProjectedGrad` |
| Tile binning | `nano-optimize::tile_binner` | `TileBinner`, `TilingParams` |
| Scan / sort | `nano-optimize::{prefix_scan, radix_sort}` | `PrefixScan`, `RadixSort` |
| Adam optimiser | `nano-optimize::adam` | `AdamState`, `AdamConfig` |
| Training loop | `nano-optimize::train` | `train(scene, cfg, on_iter)`, `TrainConfig` |
| Reference baking | `nano-optimize::reference` | `bake_references`, `ReferenceView`, `BakeConfig` |
| Viewer | `nano-view` | `run`, `run_with_training`, `TrainStats` |
| VRAM probe | `gpu-mem` | `query`, `sys_mem` |

---

Full prose: see `docs/src/` (mdbook). Decision log: `docs/src/decisions.md`.
Roadmap: `docs/src/roadmap.md`.
