# NanoTracer-RS

GPU path tracer + Gaussian-splat generator in Rust. Vulkan ray queries
(hardware-BVH), order-3 spherical-harmonics splat fitting, EXR
environment maps, glTF mesh import.

Originally a Rust port of [tinyraytracer](https://github.com/ssloy/tinyraytracer) (Dmitry V. Sokolov);
the CPU pipeline has since been replaced with a GPU path-tracer and a
forward-pass Gaussian-splat generator (geometry → splats, not the usual
images → splats).

![Screenshot](data/splat.jpg)

## Workspace layout

7 crates + binary at the workspace root:

```
nanotracer-rs/
├─ Cargo.toml          (workspace + bin)
├─ src/main.rs         (CLI)
├─ crates/
│  ├─ gpu-mem/         (zero-dep cross-platform VRAM / RAM query)
│  ├─ nano-core/       (scene, geometry, mesh, material, environment,
│  │                    colour helpers, CPU SH reference, LightSampling)
│  ├─ nano-io/         (glTF mesh loader, PNG framebuffer writer)
│  ├─ nano-shaders/    (PREAMBLE + HELPERS GLSL chunks shared by the
│  │                    image and splat shaders, plus `assemble()`)
│  ├─ nano-gpu/        (ash + shaderc: VkContext, BLAS/TLAS, buffers,
│  │                    scene -> GPU buffer marshalling)
│  ├─ nano-render/     (image renderer + path-trace compute shader)
│  ├─ nano-splat/      (splat generator + 3DGS-compatible PLY writer)
│  └─ splat-ref/       (the original CPU reference impl — kept as a
│                       browsable source tree, excluded from the
│                       workspace, does not compile)
└─ data/, test*.cmd    (assets and helper scripts)
```

See `AGENTS.md` / `DIAGRAMS.md` for dataflow and module-dependency maps,
`plan1.md` for the bug-hunt history, `WGPU_RESEARCH.md` for the
ash-vs-wgpu evaluation.

## Build prerequisites

- Rust **edition 2024** toolchain.
- Vulkan SDK for `shaderc-sys` (the build script reads `vk.xml` from it).
  Set `VULKAN_SDK` to the install directory, e.g.

  ```pwsh
  $env:VULKAN_SDK = "C:\Programs\VulkanSDK\1.4.341.1"
  ```

- A GPU with `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (any
  RTX 20-series / RDNA2+ / Arc / Apple GPU via MoltenVK should do).

## Quick start

```bash
cargo build --release

# Default render: procedural sky, 2x2 AA, ~200 random objects, output.png
cargo run --release

# Higher quality
cargo run --release -- -a 4 --mesh all -n 220 --seed 2025

# HDR environment lighting
cargo run --release -- -e data/studio.exr -x 0.15

# Gaussian-splat export (PLY for SuperSplat / Luma / bevy_gaussian_splatting)
cargo run --release -- -S scene.ply \
    --splat-density 512 --sh-samples 36 --splat-scale 0.03
```

On startup the binary prints `gpu-mem` info:

```
GPU: NVIDIA GeForce RTX 3080 Ti (12288 MiB VRAM, 10533 MiB free)
```

## CLI options

| Option | Description |
|---|---|
| `-a, --aa N` | Anti-aliasing samples NxN (default 2) |
| `-m, --max N` | Max recursion depth (default 32) |
| `-r, --refl N` | Reflection bounce limit (default 6) |
| `-f, --refr N` | Refraction bounce limit (default 16) |
| `-n, --num N` | Number of random scene objects (default 200) |
| `-s, --seed N` | Scene RNG seed (random if omitted) |
| `-e, --env FILE` | EXR HDR environment map |
| `-x, --exposure F` | HDR exposure multiplier (default 0.1) |
| `--sky` | Procedural sky gradient (default on) |
| `--mesh TYPE` | Add `cube` / `pyramid` / `torus` / `all` |
| `--glb FILE` | Load glTF/GLB mesh into scene |
| `--glb-scale F` | Scale applied to GLB mesh (default 1.0) |
| `--no-floor` | Disable the procedural checkerboard plane |
| `--no-spheres` | Mesh-only mode |
| `-t, --tonemap` | Reinhard tonemap before save (default on) |
| `-S, --splats FILE` | Export Gaussian splats to PLY |
| `--sh-samples N` | SH directions sampled per splat (default 64) |
| `--sh-glossy-mult F` | Extra-sample multiplier for glossy/refractive (default 1.5) |
| `--radiance-clamp F` | Per-sample luma clamp (0 disables, default 20) |
| `--light-sampling MODE` | `all` / `one` — renderer only; splat path always sums all |
| `--detail-boost F` | Adaptive density boost factor (default 1.5) |
| `--detail-boost-max F` | Max adaptive boost (default 3.0) |
| `--splat-density F` | Surface samples per unit area (default 100) |
| `--splat-scale F` | Override splat radius (auto from density when unset) |
| `--sh-keep-glossy` | Keep view-dependent SH on mirror/glass (off by default — DC fallback prevents the order-3-SH "rainbow" ringing) |

## Shading model

Both the image renderer and the splat fitter run the same shading code:

- **Diffuse** is Lambertian (`kd * diffuse_color * (N·L)` per light).
- **Specular** is GGX / Trowbridge–Reitz microfacet BRDF with Smith
  geometry and Schlick Fresnel (`f0 = ks`). Legacy `specular_exponent`
  is mapped to GGX roughness via `α = √(2/(n+2))`.
- **Reflection / refraction** rays are weighted by `kr` / `kt`. For
  dielectric materials with both > 0 (e.g. `GLASS`) a per-hit Schlick
  Fresnel rebalances the split — at grazing angles `kr` rises toward 1
  and the surface becomes mirror-like.
- **IBL diffuse** comes from a pre-convolved degree-2 SH of the env
  (Ramamoorthi–Hanrahan band factors `A_0=π, A_1=2π/3, A_2=π/4`).
  The 9 vec4 coefficients sit in the Params UBO and evaluate per
  surface normal at shade time.

## Materials

All material constants are energy-conserving (`kd + kr + kt ≤ 1`) with
albedo channels `[kd, ks, kr, kt]` — see `crates/nano-core/src/material.rs`.

`ks` is Fresnel F₀ for the GGX BRDF (≈ 0.04 for typical dielectrics).
The BRDF already absorbs F₀, so changing `specular_exponent` (now mapped
to GGX α) reshapes the lobe but not its integrated energy.

| Material | kd | ks (F₀) | kr | kt | n → α | Look |
|---|---|---|---|---|---|---|
| `IVORY` | 0.85 | 0.04 | 0.10 | 0.00 | 50 → 0.20 | Warm off-white, soft highlight |
| `GLASS` | 0.00 | 0.04 | 0.10 | 0.85 | 300 → 0.082 | Fresnel-blended dielectric |
| `RED_RUBBER` | 0.90 | 0.04 | 0.00 | 0.00 | 10 → 0.41 | Saturated matte red |
| `MIRROR` | 0.00 | 0.04 | 0.96 | 0.00 | 1500 → 0.037 | Near-perfect metallic mirror |
| `MATTE_*` | 0.95 | 0.04 | 0.00 | 0.00 | 20 → 0.30 | Pure matte family |

## Splat-fit notes

The splat path fits an order-3 SH per surface sample, with several
specific bias choices made for stability — see commit history and
`plan1.md` §1 for derivations:

- DC stored without an extra `/SH_C0`: viewers multiply back by `SH_C0`,
  so the LSQ-produced α₀ goes in directly.
- `shade_surface` always sums **all** lights regardless of
  `--light-sampling` — per-direction Monte-Carlo light pick shows up as
  "splotched" speckle in the fitted SH.
- Band-aware Tikhonov damping: λ rises with SH degree (1e-3 at ℓ=0,
  2e-2 at ℓ=3) to suppress hemisphere-only LSQ ringing.
- Reflective/refractive materials (`kr>0 \|\| kt>0`) drop view-dependent
  SH and keep DC only — order-3 SH cannot represent sharp specular on
  one hemisphere without rainbow ringing.
- Disc normal scale set to `splat_scale × 0.5` — thinner ratios left
  edge-on gaps at tight density.

## Dependencies

- `ash` + `shaderc` — Vulkan and GLSL → SPIR-V.
- `glam` — vector math.
- `bytemuck` — `#[repr(C)]` buffer marshalling.
- `exr` — HDR environment input.
- `gltf` — glTF/GLB mesh import.
- `image` — PNG framebuffer output.
- `indicatif` — progress bars.
- `clap` — CLI.
- `fastrand` — scene RNG.

`rtbvh` was dropped after the GPU refactor: ray intersection lives on
the GPU. See `plan1.md` for details. wgpu evaluated in `WGPU_RESEARCH.md`
(decision: stay on ash until wgpu ray-query stabilises).

## License

MIT.
