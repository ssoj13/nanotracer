# NanoTracer-RS

High-performance path tracer in Rust with **Gaussian Splatting (3DGS) export**.

Based on [tinyraytracer](https://github.com/ssloy/tinyraytracer) by Dmitry V. Sokolov.

![Screenshot](data/splat.jpg)

## Features

- **Path tracing** with reflection, refraction, soft shadows
- **Parallel rendering** via Rayon (scales to all CPU cores)
- **SIMD optimization** with `wide` crate for vectorized ray casting
- **Mesh primitives** (cube, pyramid, torus) with BVH acceleration
- **HDR environment maps** (.exr) or procedural sky
- **Adaptive anti-aliasing** with Halton quasi-Monte Carlo sampling
- **Gaussian splat export** to 3DGS-compatible PLY format

## Quick Start

```bash
cargo build --release

# Basic render
cargo run --release

# High quality with 4x AA and procedural sky
cargo run --release -- -a 4 --sky

# HDR environment lighting
cargo run --release -- -e data/studio.exr -x 0.15

# With mesh primitives
cargo run --release -- --mesh all -a 4
```

## Gaussian Splatting Export

Export scene as 3D Gaussian Splats for real-time rendering:

```bash
# Generate splats (view-independent SH0 color)
cargo run --release -- -S output.ply --splat-density 200

# Custom parameters
cargo run --release -- -S scene.ply \
    --splat-density 500 \
    --sh-samples 64 \
    --mesh torus
```

**Output format:** Standard 3DGS PLY compatible with SuperSplat, Luma AI, bevy_gaussian_splatting.

## CLI Options

| Option | Description |
|--------|-------------|
| `-a, --aa N` | Anti-aliasing samples NxN (default: 2) |
| `-m, --max N` | Max ray depth (default: 32) |
| `-r, --refl N` | Max reflection bounces (default: 6) |
| `-f, --refr N` | Max refraction bounces (default: 16) |
| `-n, --num N` | Number of random spheres (default: 200) |
| `-e, --env FILE` | HDR environment map (.exr) |
| `-x, --exposure F` | HDR exposure (default: 0.1) |
| `--sky` | Procedural sky gradient |
| `--mesh TYPE` | Add mesh: cube, pyramid, torus, all |
| `--no-floor` | Disable checkerboard plane |
| `--no-spheres` | Mesh-only mode |
| `-S, --splats FILE` | Export Gaussian splats to PLY |
| `--splat-density N` | Samples per unit area (default: 100) |
| `--sh-samples N` | SH sampling rays (default: 64) |
| `-t, --tonemap` | Apply Reinhard tonemapping |
| `--adaptive-aa` | Adaptive sampling (default: on) |

## Materials

| Material | Properties |
|----------|------------|
| Ivory | Diffuse with weak reflection |
| Glass | Transparent, refractive index 1.333 |
| Mirror | Highly reflective metallic |
| Red Rubber | Matte, no transparency |

## Architecture

```
src/
├── main.rs          # CLI, scene setup, render loop
├── renderer.rs      # Ray casting, RayConfig, lighting
├── simd_renderer.rs # SIMD-optimized Vec3x4, ray-sphere
├── scene.rs         # Scene graph, BVH traversal
├── geometry.rs      # Sphere, Object, intersection
├── mesh.rs          # Triangle mesh with BVH (rtbvh)
├── material.rs      # Material definitions
├── environment.rs   # HDR/procedural sky
├── color.rs         # Tonemapping, sRGB conversion
└── splat/           # Gaussian splatting module
    ├── sampler.rs   # Surface sampling (Fibonacci sphere)
    ├── sh.rs        # Spherical harmonics fitting
    └── ply.rs       # PLY file writer
```

## Dependencies

- **rayon** - parallel iteration
- **glam** - vector math
- **wide** - SIMD operations
- **rtbvh** - BVH acceleration
- **image** - PNG output
- **exr** - HDR environment loading
- **clap** - CLI parsing

## Performance

- 24 cores: ~1s for 50 objects at 2x AA
- SIMD sphere intersection (4 rays/batch)
- Auto tile sizing based on CPU cores
- BVH acceleration for meshes

## License

MIT
