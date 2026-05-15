# WGPU_RESEARCH.md — feasibility of replacing `ash` with `wgpu`

**Date:** 2026-05-15. Knowledge cutoff: January 2026.

## TL;DR

**Don't migrate right now.** The codebase's hot path is `VK_KHR_ray_query` (hardware-BVH ray queries inside compute shaders). `wgpu` 23.x–26.x exposes ray-tracing acceleration structures and a `rayQueryEXT`-equivalent **only under an unstable feature flag** (`RAY_QUERY` / `RAY_TRACING_ACCELERATION_STRUCTURE`) that is not yet in the published WebGPU spec; WGSL syntax for ray queries is still in flux. A migration today would be either:

1. A rewrite around software BVH traversal in WGSL (losing hardware acceleration — significant regression on RTX/RDNA2+ hardware), or
2. A bet on `wgpu`'s unstable RT feature locking in unchanged before our release — risky.

**Recommendation:** stay on `ash` for the `nano-vk` crate. Name the crate `nano-gpu` (or similar) rather than `nano-vk` so future migration doesn't force a rename. Track the wgpu RT issue trackers and re-evaluate once it lands in a stable release.

## What we depend on (hard requirements)

Inspected from `src/vk_runtime.rs` + the two compute shaders:

| Vulkan feature | GLSL extension | Usage |
|---|---|---|
| `VK_KHR_acceleration_structure` | `GL_EXT_ray_query` | Build BLAS/TLAS over scene triangles |
| `VK_KHR_ray_query` | `rayQueryEXT`, `rayQueryInitializeEXT`, `rayQueryProceedEXT`, `rayQueryGetIntersection*EXT` | The hot loop in both shaders |
| `VK_KHR_buffer_device_address` | (BDA) | Buffer references for AS build inputs |
| `VK_KHR_deferred_host_operations` | — | Required by acceleration_structure |
| 1.2 core | SPIR-V 1.4 | Shader feature set |

Everything else is "ordinary" compute Vulkan (storage buffers, uniform buffers, storage images, samplers) which `wgpu` supports trivially.

## wgpu state (Jan 2026)

### Available — stable

- `wgpu` ≥ 0.20 covers all the "ordinary" parts: SSBO/UBO as buffers, storage textures, compute pipelines, descriptor sets via bind groups.
- WGSL is the canonical shader language; GLSL is no longer first-class (the `naga-glsl` frontend is in maintenance mode and several uniform/array features lag behind WGSL).
- Cross-platform: DX12, Metal, Vulkan, GL, WebGPU.

### Unstable — gated

- `wgpu::Features::EXPERIMENTAL_RAY_QUERY` and `EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE` (names vary by minor release) — surfacing `VK_KHR_ray_query` and `VK_KHR_acceleration_structure` on Vulkan, with placeholder paths on DX12 (`DXR`) and Metal (`MTLAccelerationStructure`).
- WGSL `ray_query`/`acceleration_structure` syntax exists in `naga` behind a feature flag; spec is being actively negotiated in the W3C WebGPU repo, so source compatibility between wgpu minor releases is not guaranteed.
- Browser support: zero — WebGPU spec has no RT extension yet.

### What "experimental" actually means in practice

- The API surface can change between minor `wgpu` releases.
- Validation / portability layers may reject programs in subtle ways depending on driver vendor.
- DX12 backend support of the experimental RT feature lags behind Vulkan.
- Metal backend support is partial (Apple Silicon only, via `MTLAccelerationStructureGeometryDescriptor`).

## Migration cost estimate, if we did it

Items that would change (concrete, not hypothetical):

- `src/vk_runtime.rs` (~700 lines, all ash) → `wgpu::Device` + bind group layout descriptors. Loses all the manual descriptor / pool / memory bookkeeping (good) but requires re-expressing BLAS/TLAS build via the experimental API.
- Both shader strings (GLSL `#version 460` + `GL_EXT_ray_query`) → WGSL. The `rayQueryEXT` calls translate roughly 1:1 in spirit but the surrounding code (UBO layouts, struct padding, sampler bindings, storage buffer interface blocks) needs full rewrite.
- `shaderc` dependency disappears; replaced by `naga` (already a transitive dep of wgpu).
- `bytemuck::Pod` layouts stay (still need explicit `#[repr(C)]` for buffer-mapped structs).
- `ash` dependency goes away.
- Build: no more `$VULKAN_SDK` requirement on Windows — wgpu pulls everything it needs at compile time. **This is the real ergonomic win.**

Estimated effort: **1–2 weeks of focused work** including driver/validation hunting, with non-trivial perf risk until the experimental RT feature stabilises.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Stay on `ash` (current) | Hardware RT works. No surprises. | Bare-metal Vulkan API, lots of `unsafe`. Windows users must install Vulkan SDK to build `shaderc-sys`. |
| `vulkano` 0.34+ | Safer than ash, still Vulkan, ray tracing in `vulkano-shaders` works. | Smaller community than ash; another large rewrite. |
| `wgpu` with software BVH | No driver extension dependency; portable. | Big perf loss; need to also rewrite Scene→BVH on host. |
| `wgpu` with experimental RT | Cross-platform; modern API. | Unstable; API churn; weak DX12/Metal coverage. |

## Decision

For the upcoming workspace reorganisation, name the GPU-runtime crate generically (`nano-gpu` rather than `nano-vk`) and keep `ash` + `shaderc` as the implementation. Re-evaluate `wgpu` once **all** of the following hold:

1. `wgpu` ships `RAY_QUERY` outside an `EXPERIMENTAL_*` flag.
2. WGSL ray-query syntax is part of an accepted WebGPU spec PR.
3. DX12 backend reaches feature parity with Vulkan for AS build.

Tracking issues to revisit: `gfx-rs/wgpu` RT roadmap, W3C `gpuweb` issues tagged `ray-tracing`.
