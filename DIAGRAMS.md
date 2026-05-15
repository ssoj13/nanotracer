# DIAGRAMS.md — nanotracer-rs flow diagrams (mermaid)

Companion to `AGENTS.md`. Same diagrams in mermaid for GitHub / IDE
rendering. Updated for the 2026-05 7-crate workspace layout.

---

## 1. Top-level CLI dispatch

```mermaid
flowchart TD
    A[src/main.rs] -->|gpu_mem::query startup| B[print GPU info]
    A -->|build Scene| S[nano-core::Scene]
    S --> SPL{--splats FILE?}
    SPL -->|yes| RTS[nano-splat::generate_splats_gpu]
    SPL -->|no|  RTR[nano-render::render]
    RTS --> PLY[nano-splat::ply::write_ply]
    RTR --> PNG[nano-io::utils::save_image]
    PLY --> OUT1[(scene.ply)]
    PNG --> OUT2[(output.png)]
```

## 2. Scene → GPU buffers

```mermaid
flowchart LR
    SC[nano-core::Scene] --> BG[nano-gpu::gpu_scene::build_gpu_scene_with_detail_boost]
    BG --> V[vertices vec4]
    BG --> N[normals vec4]
    BG --> T[triangles uvec4]
    BG --> TM[tri_materials u32]
    BG --> CDF[tri_cdf f32 normalised]
    BG --> M[materials GpuMaterial]
    BG --> L[lights vec4]
    V & N & T & TM & CDF & M & L --> VK[Vulkan SSBO/UBO buffers]
    V & T --> AS[BLAS + TLAS via vk_runtime]
```

## 3. Splat generation pipeline (shader-level)

```mermaid
flowchart TD
    ID[gl_GlobalInvocationID.x] --> CDF_PICK[find_triangle on tri_cdf]
    CDF_PICK --> BARY[uniform barycentric sqrt-r1 / r2]
    BARY --> FETCH[interpolate pos, normal, fetch material]
    FETCH --> FRAME[build tangent frame]
    FRAME --> LOOP{for s in 0..&lpar;sh_samples + glossy_extra&rpar;}
    LOOP --> DIR[hemisphere sampler &mdash; idx stratified + golden ratio]
    DIR --> SHADE[shade_surface: all lights + reflect/refract trace_path]
    SHADE --> CLAMP[luma clamp]
    CLAMP --> TM[tonemap reinhard?]
    TM --> SRGB[linear_to_srgb]
    SRGB --> ACC[accumulate ATA, ATB for 16 SH basis funcs]
    ACC --> LOOP
    LOOP -->|done| TIK[band-aware Tikhonov &lambda;_l]
    TIK --> SOLVE[solve_linear x3 R/G/B]
    SOLVE --> DC[sh_dc = coeffs_0 - F1]
    DC --> GLOSSY{albedo.z &gt; 0 OR albedo.w &gt; 0}
    GLOSSY -->|yes| ZERO[zero coeffs 1..15 - F4]
    GLOSSY -->|no|  KEEP[keep full SH]
    ZERO --> PACK
    KEEP --> PACK
    PACK[pack GaussianOut: pos, normal, sh_dc, sh_rest planar, opacity logit, scale log F5, rot quat-from-normal] --> WRITE[gaussians ID]
```

## 4. Image renderer pipeline (shader-level)

```mermaid
flowchart TD
    GID[gl_GlobalInvocationID.xy] --> CULL{gid &lt; width,height}
    CULL --> AA[for s in 0..aa^2: halton 2,3 jitter]
    AA --> RAY[pinhole ray, origin=0, dir from fov]
    RAY --> STACK_INIT[push origin+dir, weight=1, depth=0]
    STACK_INIT --> POP{stack_size &gt; 0}
    POP -->|miss| SKY[sample_environment]
    SKY --> ACC[sample_color += weight*env]
    POP -->|hit| SHADE[diffuse + spec lights all/one]
    SHADE --> RR{depth &gt; 3?}
    RR -->|yes| RUSSIAN[Russian-roulette continue with prob max comp]
    RR -->|no| PUSH
    RUSSIAN --> PUSH
    PUSH[push reflection + refraction with weight *= albedo.z/.w] --> POP
    POP -->|empty| AVG[final_color += sample_color]
    AVG -->|next sample| AA
    AA -->|done| OUT[outImage gid = final_color/sample_count]
```

## 5. Crate dependency graph

```mermaid
flowchart TD
    bin[bin: nanotracer-rs] --> nc[nano-core]
    bin --> nio[nano-io]
    bin --> nr[nano-render]
    bin --> ns[nano-splat]
    bin --> gm[gpu-mem]
    bin --> glam_ext[glam]
    bin --> clap_ext[clap]
    bin --> fr[fastrand]

    nio --> nc
    nr  --> nc
    nr  --> ng[nano-gpu]
    nr  --> nshd[nano-shaders]
    ns  --> nc
    ns  --> ng
    ns  --> nshd
    ng  --> nc

    classDef pure fill:#1f4a1f,color:#fff,stroke:#0a0;
    classDef vendor fill:#2a2a4a,color:#fff,stroke:#88f;
    class nc,nio,nshd pure
    class gm vendor
```

`nano-shaders` is pure GLSL string constants (no runtime deps); `gpu-mem`
is `std`-only vendored from `vfx-rs`.

## 6. Shared GLSL split

```mermaid
flowchart LR
    subgraph PRE[PREAMBLE - no globals]
        S1[Material struct]
        S2[wang_hash / rand01 / max_component]
        S3[reflect_dir / refract_dir / offset_origin]
        S4[checker_color]
        S5[tonemap_reinhard / linear_to_srgb]
        S6[EPS / MAX_STACK / PI / FLAG_CHECKER]
    end

    subgraph HEL[HELPERS - need bindings]
        H1[sample_environment - uses params, env_map]
        H2[trace_ray / shadow_ray - uses topLevelAS]
    end

    PRE --> RB[RENDERER_BINDINGS]
    RB  --> HEL
    HEL --> RBO[RENDERER_BODY: halton + main]

    PRE --> SB[SPLAT_BINDINGS]
    SB  --> HEL
    HEL --> SBO[SPLAT_BODY: SH consts + sample_uniform_hemisphere + trace_path + shade_surface + sh_basis + solve_linear + quat_from_normal + find_triangle + main]
```

## 7. Quick reference: which crate owns what

| Concern | Crate | Key types / fns |
|---|---|---|
| Scene & geometry data | `nano-core` | `Scene`, `Object`, `Geometry`, `Mesh`, `Light` |
| Materials | `nano-core::material` | `Material`, `IVORY`, `GLASS`, `MIRROR`, `MATTE_*` |
| Environment IBL | `nano-core::environment` | `EnvironmentMap`, `EnvGpuData` |
| Colour pipeline | `nano-core::color` | `tonemap_reinhard`, `linear_to_srgb`, `apply_tonemap_srgb` |
| CPU SH reference | `nano-core::sh` | `sh_basis`, `fit_sh`, `eval_sh`, `fibonacci_hemisphere` |
| Light-sampling enum | `nano-core::LightSampling` | `All`, `One`, `.as_u32()` |
| glTF mesh loader | `nano-io::gltf_loader` | `load_glb_mesh` |
| PNG framebuffer writer | `nano-io::utils` | `save_image` |
| GLSL chunks | `nano-shaders` | `PREAMBLE`, `HELPERS`, `assemble` |
| Vulkan runtime | `nano-gpu::vk_runtime` | `VkContext`, `AccelResource`, `BufferResource`, `ImageResource` |
| Scene → GPU marshalling | `nano-gpu::gpu_scene` | `build_gpu_scene*`, `GpuMaterial`, `GpuTriangle` |
| Image renderer | `nano-render` | `render`, `RenderConfig` |
| Splat generator | `nano-splat::generator` | `generate_splats_gpu`, `SplatConfigGpu` |
| Splat PLY writer | `nano-splat::ply` | `write_ply`, `Gaussian` |
| VRAM / RAM probe | `gpu-mem` | `query`, `sys_mem`, `GpuMemInfo`, `SysMemInfo` |

---

See `AGENTS.md` for ASCII versions and the decision log, `plan1.md` for
the bug-hunt history.
