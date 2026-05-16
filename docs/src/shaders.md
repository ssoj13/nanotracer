# Shader catalog

Inventory of every GLSL chunk and WGSL kernel in the workspace, with
inputs/outputs and where the math comes from.

## GLSL (raytrace + splat-fit paths, `ash`)

GLSL lives in `crates/nano-shaders/src/lib.rs` as `&'static str`
chunks plus an `assemble(bindings, body)` helper. Per-shader files
in `nano-render/src/renderer.rs` and `nano-splat/src/generator.rs`
provide the `BINDINGS` and `BODY` for their respective compute
shaders.

### `nano-shaders::PREAMBLE`

Sits at the top of every shader (before any binding declarations).
No global dependencies.

| Declaration | Notes |
|-------------|-------|
| `Material` struct | Mirrors `nano-core::material::Material`. |
| `GpuLight` struct | 64-byte std430; mirrors `nano-gpu::gpu_scene::GpuLight`. |
| `LightSample` struct | Output of `sample_light`. |
| `LIGHT_POINT` … `LIGHT_ENV` constants | Match `LIGHT_KIND_*` in nano-gpu. |
| `SH_C0`, `SH_C1`, `SH_C2[5]`, `SH_C3[7]` | Condon–Shortley basis. |
| `EPS`, `MAX_STACK`, `PI`, `FLAG_CHECKER` | |
| `wang_hash`, `rand01` | Per-thread RNG. |
| `reflect_dir`, `refract_dir`, `offset_origin` | |
| `quat_rotate(q, v)` | (x, y, z, w) convention. |
| `phong_to_alpha(n)` | Legacy specular_exponent → GGX α. |
| `ggx_specular(N, V, L, α, F₀)` | GGX BRDF × cos. |
| `ggx_msc_boost(α, F₀)` | Turquin 2018 multi-scattering compensation. |
| `tonemap_reinhard`, `linear_to_srgb` | |
| `checker_color`, `max_component` | |

### `nano-shaders::HELPERS`

Sits after the per-shader bindings. Needs `topLevelAS`, `env_map`,
`params`, `lights`, `light_radiance` declared first.

| Function | Notes |
|----------|-------|
| `sample_environment(dir)` | Procedural sky vs HDR texture sample. |
| `trace_ray`, `shadow_ray` | Vulkan ray-query helpers. |
| `eval_env_irradiance(n)` | Pre-convolved degree-2 SH irradiance at the surface normal. |
| `sample_light(idx, hit_pos, hit_n, rand_uv)` | Returns `LightSample` for kind-`{Point, Rect, Sphere, Box, Env}`. |

### Per-shader bodies

| File | Workgroup | Purpose |
|------|-----------|---------|
| `nano-render/src/renderer.rs` `RENDERER_BODY` | 8×8 pixel tile | Pinhole pathtrace, halton-jittered AA, stack-based bounces, area-light MC via `sample_light`. |
| `nano-splat/src/generator.rs` `SPLAT_BODY` | 64-thread workgroup | Forward-fit splat generator: triangle-CDF pick, barycentric sample, hemisphere sampler, SH LSQ fit with band-aware Tikhonov. |

## WGSL (training + viewer, `wgpu`)

All WGSL files in `crates/nano-optimize/src/shaders/`:

### Forward rasteriser

| File | Bindings | Output |
|------|----------|--------|
| `project_gaussians.wgsl` | 6 splat SSBOs + Camera uniform | `ProjectedSplat[]` (48 B per splat) |
| `tile_count.wgsl` | `projected` (RO) + `counts` (RW) + atomic `total` + uniform | `counts[n]`, `total[0]` |
| `scan_block.wgsl` | `data` (RW) + `block_sums` (RW) + uniform | Per-block exclusive scan + block totals |
| `scan_add_offsets.wgsl` | `data` (RW) + `offsets` (RO) + uniform | `data[i] += offsets[wid]` |
| `tile_emit.wgsl` | `projected` + `offsets` + (`keys`, `payloads`) (RW) + uniform | `keys[]`, `payloads[]` |
| `bit_predicate.wgsl` | `keys` (RO) + `predicate` (RW) + uniform | `predicate[i] = 1 if bit-i of keys[i] is 0` |
| `bit_total_zeros.wgsl` | `keys`, `scan` (RO) + `total` (RW) + uniform | `total[0]` = count of zero-bit elements |
| `bit_scatter.wgsl` | `(keys_src, vals_src)` (RO), `(keys_dst, vals_dst)` (RW), `scan`, `total` (RO) + uniform | Stable split scatter |
| `tile_ranges.wgsl` | `sorted_keys` (RO) + `tile_ranges` (RW) + uniform | `tile_ranges[2·tile_id..]` = `[begin, end)` |
| `rasterize.wgsl` | `projected` (RO) + `sorted_payloads` (RO) + `tile_ranges` (RO) + `output` (RW) + uniform | `output[]` = `vec4<f32>` per pixel |

### Backward rasteriser

| File | Bindings | Output |
|------|----------|--------|
| `rasterize_backward.wgsl` | `projected` + `sorted_payloads` + `tile_ranges` + `forward_out` + `dL_dC` (all RO) + `projected_grad` (RW atomic) + uniform | Per-splat 2D-state gradient accumulator |
| `project_backward.wgsl` | Camera + 6 splat SSBOs + `projected_grad` (RO) + 6 grad SSBOs (RW) | Per-splat 3D parameter gradients |

### Viewer

| File | Bindings | Output |
|------|----------|--------|
| `tonemap.wgsl` | `source` (RO `vec4<f32>` buffer) + `output` (`rgba8unorm` storage texture) + uniform | sRGB-encoded `rgba8unorm` texture |

## Numerical-stability invariants

- `det(Σ_2D) ≥ 0.3 · 0.3 − ε` after 3-pixel low-pass dilation. The
  forward kernel still checks `det ≤ 0` as a safety net.
- Forward α-blend bails at `T < 1e-4`. Backward bails at `T > 1.0001`
  — same boundary, different direction.
- `1 - α` is clamped to `ONE_MINUS_ALPHA_EPS = 1e-4` in the backward
  divide.
- CAS-based `atomic_add_f32` in `rasterize_backward.wgsl` retries
  until the swap succeeds; under heavy contention this is the slow
  part of the kernel.

## Compatibility notes

- WGSL `atomic<u32>` works on `Rgba8Unorm` textures — bitcast for
  float atomics.
- `wgpu::RenderPassDescriptor` in wgpu 29 added `multiview_mask:
  Option<u32>` (None for non-multiview).
- `egui_wgpu::Renderer::new` in 0.34 takes `RendererOptions`, not
  the old 5-arg form.
- WGSL `select(false_value, true_value, condition)` argument order
  is the opposite of C `?:`.
