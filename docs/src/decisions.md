# Decision log

Every non-obvious "we picked this over that" call made during the
project, with the reasoning. Sorted roughly chronologically.

## Pre-history (tinyraytracer port)

- **Edition 2024.** Used for `let-else`, `if let chains`, and a
  cleaner closure syntax that helped during the GPU refactor.
- **`glam` over `nalgebra`.** Smaller, less generic; matches
  what most Rust GPU stacks (wgpu, bevy) use.
- **`fastrand` over `rand`.** Zero deps, deterministic seeding.

## CPU pipeline → GPU pipeline

- **Vulkan ray queries instead of compute-side BVH traversal.** The
  hardware path is ~10× faster on RTX 20-series+ / RDNA2+. Software
  BVH inside a WGSL/GLSL compute would have left a lot of throughput
  on the table for the reference-baking step.
- **`ash` (raw Vulkan) over `wgpu` for the raytrace side.** wgpu's
  `RAY_QUERY` extension is gated behind `EXPERIMENTAL_*` and the
  WGSL ray-query spec is still in flux (see `WGPU_RESEARCH.md`).
  Migrating once the experimental flag drops is parked.
- **`shaderc` over manual SPIR-V generation.** `shaderc-sys`
  requires `VULKAN_SDK` set, but cuts compile times for shader
  hot-reloads (not used yet) and keeps GLSL human-readable.

## Workspace reorganisation

- **Crate boundaries enforce data flow, not feature splits.**
  `nano-core` knows no GPU types; `nano-shaders` is pure constants;
  `nano-gpu` doesn't know about `nano-render` / `nano-splat`. Each
  upgrade path (e.g. wgpu migration) touches exactly the crates
  that need it.
- **Keep `splat-ref/` excluded from the workspace.** It references
  the pre-refactor `crate::renderer::*` namespace. Useful as
  browsable documentation of the CPU implementation; deleting it
  would discard a working SH-fit reference.
- **`nano-shaders` is a `&'static str` crate, not a build-script
  generator.** Concatenation happens at runtime in `assemble()`.
  Zero build-time deps; shader text remains readable in `cargo
  expand`-style debug.

## Shading model

- **GGX/Trowbridge-Reitz over Phong.** Energy-conserving by
  construction, better grazing-angle behaviour, what every modern
  PBR pipeline expects.
- **Multi-scattering compensation (Turquin 2018, fit after
  Heitz/Hill).** Single-scatter GGX leaks energy at high roughness.
  The boost factor recovers ~10–15% of "missing" light at α → 1
  without a real path tracer.
- **Materials as `[kd, ks, kr, kt]` with `Σ ≤ 1`.** Energy-
  conserving by construction. `ks` doubles as GGX F₀ — the BRDF
  already absorbs F₀, so changing `specular_exponent` (now mapped
  to GGX α) reshapes the lobe but not its integrated energy.

## Splat-fit biases (the F1–F5 fixes)

- **F1 — DC stored without `/SH_C0`.** Viewers reconstruct via
  `SH_C0 · f_dc`, so the LSQ-produced α₀ goes in directly. Earlier
  bug: dividing by `SH_C0` produced 4× brighter DC than viewers
  rendered.
- **F2 — Sum all lights inside the SH fitter.** Per-direction MC
  light pick shows up as "splotched colour" speckle. `--light-
  sampling all` is forced in the splat path regardless of CLI.
- **F3 — Band-aware Tikhonov damping.** λ rises with SH degree
  (1e-3 at ℓ=0, 2e-2 at ℓ=3). Hemisphere-only LSQ is rank-deficient
  for high-frequency modes; without damping, sample noise
  amplifies into visible speckle on flat surfaces.
- **F4 — DC-only for reflective / refractive splats.** Order-3 SH
  cannot represent sharp specular on one hemisphere without rainbow
  ringing. CLI `--sh-keep-glossy` re-enables full SH for stylised
  output.
- **F5 — Disc normal scale `splat_scale × 0.5`.** Thinner ratios
  (×0.3) left edge-on gaps at tight density.

## Area lights + IBL refactor

- **`Light` as an enum, not five structs.** One sample helper, one
  GPU layout, one CLI surface. Variants share a `radiance()`
  helper for the `color · intensity` pre-multiplication.
- **64-byte `GpuLight` + parallel `light_radiance` SSBO.** Avoids
  bit-packing colour into spare `W` components. Easy to extend
  (per-light SH probes, IES profiles → additional parallel SSBOs).
- **`Light::Env` instead of a global `ibl_strength` knob.** IBL is a
  light. The duplicate dial (`--ibl-strength` scaling SH pre-upload
  AND a separate intensity in the shader) collapsed into a single
  `intensity` field.
- **Auto-add `Light::Env { intensity: 1.0 }` when env map present
  but no explicit one.** Library callers (tests, training
  infrastructure) get IBL without having to remember the explicit
  add. The CPU `Scene` is not mutated — this is a `gpu_scene`
  build-time policy.
- **Point lights stay "unit-radiance, no falloff".** Demo-style
  convention, preserved for back-compat. A physically-correct
  `1/r²` point light is a follow-up `--point-light-physical` flag
  if needed.

## Phase A — Training

- **`wgpu` for the differentiable rasteriser.** Portable to DX12 /
  Metal / WebGPU; doesn't need ray queries. The training side and
  the reference baking side are two independent GPU contexts that
  communicate via `SplatBuffer` on the CPU.
- **Vec4-padded SSBO layout for `GpuSplatBuffer`.** WGSL `vec3` has
  16-byte alignment in `std430` anyway. Explicit padding makes the
  Rust-side struct layout trivial and removes a class of "off by
  one float" bugs.
- **`sh_rest` padded 45 → 48 floats per splat.** Same reason —
  divisible by 4 so it lays out as `vec4[12]` per splat. CPU readback
  strips the padding before AdamState consumes the slab.
- **Three-level prefix scan instead of recursion.** Iterative,
  supports up to 256³ = 16M elements in three explicit dispatch
  groups. Recursive would have been cleaner code but harder to
  reason about lifetime of intermediate buffers.
- **1-bit-at-a-time stable radix sort.** First attempt used 4-pass
  byte-radix with atomicAdd scatter; non-stable scatter breaks
  the radix-sort invariant. The 1-bit-stable version is slower
  but correct and uses only `PrefixScan` we already have.
- **CPU readback of `total_pairs` to size tile-key buffers exactly.**
  One u32 readback per iteration; cheap, deterministic. The
  alternative — pre-allocate worst-case (splats × tiles) — would
  cost GBs of VRAM for trained models.
- **CAS-based `atomic_add_f32`.** WGSL has no native float atomics.
  CAS-on-`u32`-bitcast is the standard workaround; the contention
  retry loop is fine at 256 pixels per workgroup.
- **`T > 1.0001` backward bail.** Forward early-outs at `T < 1e-4`;
  backward T-reconstruction blows up past that boundary. The clamp
  is one line but the bug took half a debug session to find on a
  real-world 5800-splat scene (CPU FD test on synthetic data
  passed).
- **Adam on CPU, splats sync via `queue.write_buffer`.** Cheaper
  than reallocating the `GpuSplatBuffer` every iter. Pure-GPU Adam
  is on the roadmap; CPU readback / upload bounces ~3–5 ms per iter
  but the constraint logic (quat re-norm, log-σ clamp) is easier
  to maintain on the CPU.
- **MSE loss for first cut, SSIM parked.** SSIM needs a 5-pass
  Gaussian-blur forward + analytic backward. MSE alone trains the
  splats; SSIM is a perceptual quality knob, not a correctness gate.
- **Densify takes deterministic random offsets.** LCG seeded by
  `iter`. A training run is reproducible given the scene + cfg.

## Phase B — Viewer

- **egui + egui_dock + egui_plot + egui_wgpu.** Standard Rust
  tooling stack. egui_dock buys dockable tab layout (Viewport +
  Inspector + Training) for free.
- **GPU-only presentation via `tonemap.wgsl` + egui native
  texture.** CPU readback for the surface blit (the original B1
  path) is wasteful per-frame bandwidth. The new path keeps the
  splat image on the GPU all the way to the swapchain.
- **Pre-load to memory before opening the viewer.** No async PLY
  load with a spinner — `--view-ply scene.ply` blocks main while
  reading. PLY parsing is fast (a few ms per million splats); the
  complexity of an async load doesn't pay off.
- **Worker thread for B2 instead of `pump_app_events`.** Training
  iterations take ~20 ms; densify takes ~50 ms. Interleaving them
  on the main thread caps the viewer's framerate at the training
  cadence. A worker thread keeps the viewer at full vsync regardless.
- **Both threads share the wgpu device (potential, currently
  separate).** wgpu's `Device` and `Queue` are `Send + Sync` so
  this is a future optimisation; currently each thread holds its
  own context for clearer ownership.

## Things explicitly *not* done

- **No async I/O.** Loading a PLY or baking references blocks the
  main thread. For a tool that's mostly used interactively or in
  batch, async pulls more weight than it gives.
- **No serde / JSON config.** CLI flags are the configuration
  surface. Scene description lives in Rust code (procedural).
  Adding a JSON layer would force a second source of truth.
- **No `Hash` / `Eq` impls on `SplatBuffer`.** Snapshots use
  `Clone` and the live preview compares monotonic version numbers,
  not buffer hashes.
- **No GPU-side splat sort by `tile_id, depth` packed into u64.**
  Our 32-bit packing (`tile_id << 16 | depth_u16`) limits scenes
  to ≤ 65k tiles and 16-bit depth quantisation. Upgrading to u64
  with native two-key sort is a future option if larger-format
  rendering becomes important.
- **No backward through SH-eval-of-view-direction.** SH coefficients
  depend on `to_cam = normalize(cam_pos − p_world)` which depends
  on `p_world`. Strictly correct backward would chain SH grads to
  position. We drop it (matches brush's reference); the
  contribution is small compared to the direct projection effect.
