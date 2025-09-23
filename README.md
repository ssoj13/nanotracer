# NanoTracer Rust

A port and extension of https://github.com/ssloy/tinyraytracer by Dmitry V. Sokolov.
Studio.exr file is taken from https://polyhaven.com/hdris.

A high-performance raytracer implementation in Rust, featuring reflection, refraction, and configurable ray tracing depths. This is a Rust port of the classic tinyraytracer project with modern improvements and parallelization.

## Features

- **Ray-sphere intersection** with multiple materials
- **Reflection and refraction** with separate depth controls
- **Phong shading model** with diffuse and specular lighting
- **Parallel rendering** using Rayon for multi-core performance
- **Configurable depth limits** via command line arguments
- **Material system** supporting ivory, glass, rubber, and mirror surfaces
- **Checkerboard ground plane** for visual reference

## Architecture

### Core Components

```
├── main.rs           # Scene setup and rendering loop
├── lib.rs            # Module declarations
├── vec3.rs           # Vector math using glam crate
├── material.rs       # Material definitions and presets
├── geometry.rs       # Sphere primitive and ray-sphere intersection
├── scene.rs          # Scene management and ray intersection
├── renderer.rs       # Ray casting and color computation
└── utils.rs          # Image saving utilities
```

### Data Flow

1. **Scene Setup** (`main.rs`)
   - Creates spheres with materials (ivory, glass, rubber, mirror)
   - Defines light sources
   - Parses command line arguments for depth limits

2. **Ray Generation** (`main.rs`)
   - For each pixel, calculates ray direction from camera
   - Calls `cast_ray_with_params()` with configurable depths

3. **Ray Casting** (`renderer.rs`)
   - `cast_ray_with_params()` → `cast_ray_with_separate_depths()`
   - Intersects ray with scene using `scene.intersect()`
   - Computes lighting, reflection, and refraction recursively

4. **Scene Intersection** (`scene.rs`)
   - Tests ray against all spheres and ground plane
   - Returns closest intersection with material properties

5. **Sphere Intersection** (`geometry.rs`)
   - Solves quadratic equation for ray-sphere intersection
   - Handles self-intersection prevention with 0.001 offset

6. **Material Properties** (`material.rs`)
   - Defines albedo coefficients for diffuse/specular/reflection/refraction
   - Refractive indices and surface properties

### Ray Tracing Algorithm

```rust
fn cast_ray_with_separate_depths(
    scene: &Scene,
    orig: Vector3,
    dir: Vector3,
    depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
    max_depth: i32,
    max_reflection_depth: i32,
    max_refraction_depth: i32
) -> Vector3
```

**The recursive ray casting process:**

1. **Depth Check**: Return background color if max depth exceeded
2. **Scene Intersection**: Find nearest object hit by ray
3. **Material Response**: Calculate surface properties at hit point
4. **Lighting**: Compute diffuse/specular from all light sources with shadow testing
5. **Reflection**: Cast reflection ray if `albedo[2] > 0` and within depth limit
6. **Refraction**: Cast refraction ray if `albedo[3] > 0` and within depth limit
7. **Color Mixing**: Combine all contributions using material albedo weights

**Final color formula:**
```rust
material.diffuse_color * diffuse_intensity * albedo[0] +     // Diffuse
Vector3::ONE * specular_intensity * albedo[1] +             // Specular
reflect_color * albedo[2] +                                 // Reflection
refract_color * albedo[3]                                   // Refraction
```

## Materials

### Predefined Materials

| Material | Refractive Index | Albedo [D,S,R,T] | Properties |
|----------|------------------|------------------|------------|
| **Ivory** | 1.0 | [0.9, 0.5, 0.1, 0.0] | Diffuse surface with weak reflection |
| **Glass** | 1.333 | [0.0, 0.9, 0.1, 0.8] | Transparent with strong refraction |
| **Red Rubber** | 1.0 | [1.4, 0.3, 0.0, 0.0] | Matte surface, no transparency |
| **Mirror** | 1.0 | [0.0, 16.0, 0.8, 0.0] | Highly reflective metallic surface |

**Albedo Components:**
- `[0]` - Diffuse contribution
- `[1]` - Specular highlight intensity
- `[2]` - Reflection strength
- `[3]` - Refraction/transmission strength

## Usage

### Build and Run

```bash
# Build in release mode for optimal performance
cargo build --release

# Run with default settings
cargo run --release

# Run with custom depth limits
cargo run --release -- -m 32 -r 6 -f 16
```

### Command Line Options

```
Usage: nanotracer [OPTIONS]

Options:
  -m, --max <MAX_DEPTH>          Maximum recursion depth [default: 32]
  -r, --refl <REFLECTION_DEPTH>  Maximum reflection depth [default: 6]
  -f, --refr <REFRACTION_DEPTH>  Maximum refraction depth [default: 16]
  -n, --env <ENV_PATH>           HDR environment map file (.exr format)
  -s, --sky                      Use procedural sky gradient instead of solid background
  -e, --exp <EXPOSURE>           Exposure adjustment for HDR environment maps [default: 0.1]
  -a, --aa <AA_SAMPLES>          Anti-aliasing samples per pixel [default: 1]
  -h, --help                     Print help
```

### Environment Options

```bash
# Default: solid blue background
cargo run --release

# Procedural sky gradient (short form)
cargo run --release -- -s

# HDR environment lighting with default exposure
cargo run --release -- -n data/studio.exr

# HDR with custom exposure adjustment (short form)
cargo run --release -- -n data/studio.exr -e 0.05  # Darker
cargo run --release -- -n data/studio.exr -e 0.2   # Brighter

# Combine options with anti-aliasing
cargo run --release -- -s -a 4
cargo run --release -- -n data/studio.exr -e 0.15 -a 4
```

### Performance Tips

- Use `--release` for 10x+ speedup over debug builds
- Higher depth values increase quality but reduce performance exponentially
- Reflection depth affects mirror and metallic surfaces
- Refraction depth affects glass and transparent materials

## Scene Configuration

The scene is hardcoded in `main.rs` but easily modifiable:

```rust
// Add a new sphere
scene.add_sphere(Sphere::new(
    Vector3::new(x, y, z),    // Position
    radius,                   // Size
    MATERIAL,                 // Material preset
));

// Add a light source
scene.add_light(Light {
    position: Vector3::new(x, y, z),
});
```

## Technical Details

### Ray-Sphere Intersection

Uses analytical solution to quadratic equation:
```rust
let l = sphere.center - ray_origin;
let tca = l.dot(ray_direction);
let d2 = l.dot(l) - tca * tca;
if d2 > sphere.radius² { return miss; }
let thc = sqrt(sphere.radius² - d2);
let t = tca ± thc;  // Two intersection points
```

### Reflection Formula

```rust
reflect_dir = incident - normal * 2.0 * incident.dot(normal)
```

### Refraction (Snell's Law)

```rust
eta = eta_incident / eta_transmitted;
cos_i = -incident.dot(normal);
k = 1 - eta² * (1 - cos_i²);
if k < 0 { total_internal_reflection; }
refract_dir = incident * eta + normal * (eta * cos_i - sqrt(k));
```

### Parallelization

Uses Rayon for pixel-parallel rendering:
```rust
framebuffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
    // Ray casting per pixel happens in parallel
});
```

## Dependencies

- **rayon** - Data parallelism for multi-core rendering
- **glam** - Fast SIMD vector math library
- **image** - PNG/JPEG image saving
- **clap** - Command line argument parsing
- **exr** - HDR environment map loading (OpenEXR format)

## Output

Renders to `output.png` in the current directory at 1024×768 resolution.

## License

MIT License - Feel free to use and modify for educational and commercial purposes.