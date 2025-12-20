use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use glam::{Quat, Vec3};
use indicatif::{ProgressBar, ProgressStyle};
use nanotracer_rs::environment::EnvironmentMap;
use nanotracer_rs::geometry::Object;
use nanotracer_rs::material::{
    GLASS, IVORY, MATTE_BLUE, MATTE_GREEN, MATTE_RED, MIRROR, RED_RUBBER,
};
use nanotracer_rs::mesh::{Mesh, cube, pyramid, torus};
use nanotracer_rs::renderer::cast_ray_with_params;
use nanotracer_rs::scene::{Light, Scene};
use nanotracer_rs::splat::{SplatConfig, ply::write_ply, sampler::generate_splats};
use nanotracer_rs::utils::save_image;
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "nanotracer")]
#[command(about = "A path tracer with Gaussian Splatting export")]
#[command(after_help = r#"EXAMPLES:
  Rendering:
    nanotracer-rs                           # Basic render, output.png
    nanotracer-rs -a 4                       # 4x anti-aliasing
    nanotracer-rs -a 8 --sky                 # 8x AA + procedural sky
    nanotracer-rs -e hdr.exr -x 0.5          # HDR environment, exposure 0.5
    nanotracer-rs -a 4 -m 64 -r 8 -f 20      # High quality: more bounces

  Controlling scene:
    nanotracer-rs -n 400 --seed 1234         # reproducible object set

  With meshes:
    nanotracer-rs --mesh cube                # Add a cube mesh
    nanotracer-rs --mesh torus               # Add a torus mesh
    nanotracer-rs --mesh all                 # Add all mesh primitives

  Gaussian Splats:
    nanotracer-rs -S scene.ply --splat-density 200 --sh-samples 64

  Disable tonemapping (linear colors):
    nanotracer-rs --tonemap false
"#)]
struct Args {
    /// Number of objects to generate for the scene
    #[arg(short = 'n', long = "num", default_value_t = 200)]
    object_count: usize,

    /// Random seed for scene generation (optional)
    #[arg(short = 's', long = "seed")]
    scene_seed: Option<u64>,

    /// Maximum recursion depth
    #[arg(short = 'm', long = "max", default_value_t = 32)]
    max_depth: i32,

    /// Maximum reflection depth
    #[arg(short = 'r', long = "refl", default_value_t = 6)]
    reflection_depth: i32,

    /// Maximum refraction depth
    #[arg(short = 'f', long = "refr", default_value_t = 16)]
    refraction_depth: i32,

    /// Anti-aliasing samples per pixel
    #[arg(short = 'a', long = "aa", default_value_t = 2)]
    aa_samples: u32,

    /// HDR environment map file (.exr format)
    #[arg(short = 'e', long = "env")]
    env_path: Option<String>,

    /// Exposure adjustment for HDR environment maps
    #[arg(short = 'x', long = "exposure", default_value_t = 0.1)]
    exposure: f32,

    /// Use procedural sky gradient instead of solid background
    #[arg(long = "sky", default_value_t = true)]
    use_sky: bool,

    /// Tile size for parallel render batches (default: auto-select based on CPU cores)
    #[arg(long = "tilesize", default_value_t = 32)]
    tile_size: usize,

    /// Apply tonemapping (Reinhard) before writing colors
    #[arg(short = 't', long = "tonemap", default_value_t = true)]
    tonemap: bool,

    /// Export Gaussian splats to PLY file (skips image rendering)
    #[arg(short = 'S', long = "splats")]
    splat_output: Option<String>,

    /// SH sampling directions per splat (default: 64)
    #[arg(long = "sh-samples", default_value_t = 64)]
    sh_samples: usize,

    /// Surface samples per unit area (default: 100)
    #[arg(long = "splat-density", default_value_t = 100.0)]
    splat_density: f32,

    /// Override splat scale (radius). Auto-calculated from density if not set
    #[arg(long = "splat-scale")]
    splat_scale: Option<f32>,

    /// Add mesh primitives: cube, pyramid, torus, all
    #[arg(long = "mesh")]
    mesh: Option<String>,

    /// Disable checkerboard plane
    #[arg(long = "no-floor")]
    no_floor: bool,

    /// Disable default spheres (mesh-only mode)
    #[arg(long = "no-spheres")]
    no_spheres: bool,

    /// Enable adaptive anti-aliasing based on pixel contrast
    #[arg(long = "adaptive-aa", default_value_t = true)]
    adaptive_aa: bool,
}

#[derive(Debug)]
struct Tile {
    x_start: usize,
    y_start: usize,
    width: usize,
    height: usize,
}

struct FramebufferPtr(*mut Vec3);

unsafe impl Sync for FramebufferPtr {}
unsafe impl Send for FramebufferPtr {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 768;
    const FOV: f32 = 1.05;

    let mut scene = Scene::new();
    let scene_seed = args
        .scene_seed
        .unwrap_or_else(|| fastrand::u64(..=u64::MAX));
    println!("Scene RNG seed: {}", scene_seed);

    // Environment setup
    if let Some(env_path) = &args.env_path {
        println!(
            "Loading HDR environment map: {} (exposure: {})",
            env_path, args.exposure
        );
        match EnvironmentMap::from_exr(env_path, args.exposure) {
            Ok(env_map) => {
                println!(
                    "Loaded {}x{} HDR environment map",
                    env_map.width(),
                    env_map.height()
                );
                scene.set_environment(env_map);
            }
            Err(e) => {
                eprintln!("Warning: Failed to load HDR environment map: {}", e);
                eprintln!("Continuing with default sky color...");
            }
        }
    } else if args.use_sky {
        println!("Using procedural sky gradient");
        scene.set_environment(EnvironmentMap::procedural_sky());
    }

    // Disable checkerboard if requested
    if args.no_floor {
        scene.checkerboard_enabled = false;
    }

    // Add randomized scene objects
    let mesh_type = args.mesh.as_deref().unwrap_or("all");
    if mesh_type != "none" || !args.no_spheres {
        add_random_objects(
            &mut scene,
            mesh_type,
            !args.no_spheres,
            args.object_count,
            Some(scene_seed),
        );
    }

    scene.rebuild_scene_bvh();

    // Add lights
    scene.add_light(Light {
        position: Vec3::new(-20.0, 20.0, 20.0),
    });
    scene.add_light(Light {
        position: Vec3::new(30.0, 50.0, -25.0),
    });
    scene.add_light(Light {
        position: Vec3::new(30.0, 20.0, 30.0),
    });

    // Display rendering statistics
    if args.splat_output.is_none() { // Only for image rendering mode
        let tile_size = if args.tile_size == 0 {
            // Auto-select tile size based on CPU cores
            let cpu_cores = num_cpus::get();
            match cpu_cores {
                0..=2 => 32,
                3..=4 => 24,
                5..=8 => 16,
                9..=16 => 12,
                _ => 8,  // More than 16 cores
            }
        } else {
            args.tile_size.clamp(1, WIDTH.min(HEIGHT))
        };

        let tiles_count = if tile_size > 0 {
            ((WIDTH + tile_size - 1) / tile_size) * ((HEIGHT + tile_size - 1) / tile_size)
        } else {
            1  // fallback value
        };

        println!("┌─────────────────────────────────────────┐");
        println!("│            RENDERING STATISTICS         │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Resolution:      {:>6} x {:<6}        │", WIDTH, HEIGHT);
        println!("│ Objects:         {:>6}                  │", args.object_count);
        println!("│ Max depth:       {:>6}                  │", args.max_depth);
        println!("│ Reflection:      {:>6}                  │", args.reflection_depth);
        println!("│ Refraction:      {:>6}                  │", args.refraction_depth);
        println!("│ AA samples:      {:>6}x{:<5}            │", args.aa_samples, args.aa_samples);
        println!("│ Tile size:       {:>6}                  │", tile_size);
        println!("│ Tiles count:     {:>6}                  │", tiles_count);
        println!("│ CPU cores:       {:>6}                  │", num_cpus::get());
        println!("│ Adaptive AA:     {:>6}                  │", if args.adaptive_aa { "YES" } else { "NO" });
        println!("│ Environment:     {:>6}                  │", if scene.environment.is_some() { "YES" } else { "NO" });
        println!("│ Output file:     {:>6}                  │", "output.png");
        println!("│ Tonemapping:     {:>6}                  │", if args.tonemap { "YES" } else { "NO" });
        println!("└─────────────────────────────────────────┘");
    }

    // Splat generation mode
    if let Some(splat_path) = &args.splat_output {
        // Display splat generation statistics
        println!("┌─────────────────────────────────────────┐");
        println!("│         SPLAT GENERATION STATISTICS     │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Objects:         {:>6}                  │", args.object_count);
        println!("│ Max depth:       {:>6}                  │", args.max_depth);
        println!("│ Reflection:      {:>6}                  │", args.reflection_depth);
        println!("│ Refraction:      {:>6}                  │", args.refraction_depth);
        println!("│ Splat density:   {:>6}                  │", args.splat_density);
        println!("│ SH samples:      {:>6}                  │", args.sh_samples);
        println!("│ CPU cores:       {:>6}                  │", num_cpus::get());
        println!("│ Environment:     {:>6}                  │", if scene.environment.is_some() { "YES" } else { "NO" });
        println!("│ Output file:     {:>6}                  │", splat_path);
        println!("│ Tonemapping:     {:>6}                  │", if args.tonemap { "YES" } else { "NO" });
        println!("└─────────────────────────────────────────┘");

        println!("Generating Gaussian splats...");
        let config = SplatConfig {
            density: args.splat_density,
            sh_samples: args.sh_samples,
            max_depth: args.max_depth,
            reflection_depth: args.reflection_depth,
            refraction_depth: args.refraction_depth,
            scale_override: args.splat_scale,
            tonemap: args.tonemap,
        };

        let gaussians = generate_splats(&scene, &config);

        let path = Path::new(splat_path);
        println!("Writing {} gaussians to {}...", gaussians.len(), splat_path);
        write_ply(path, &gaussians)?;

        let file_size = std::fs::metadata(path)?.len();
        println!("Splat generation complete!");
        println!("  Output: {}", splat_path);
        println!("  Gaussians: {}", gaussians.len());
        println!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0);

        return Ok(());
    }

    // Render mode
    let mut framebuffer = vec![Vec3::ZERO; WIDTH * HEIGHT];

    let aa_samples = args.aa_samples;
    let samples_per_pixel = (aa_samples * aa_samples) as f32;
    let inv_spp = 1.0 / samples_per_pixel;
    let half_width = WIDTH as f32 * 0.5;
    let half_height = HEIGHT as f32 * 0.5;
    let fov_scale = (FOV / 2.0).tan();
    let dir_z = -(HEIGHT as f32) / (2.0 * fov_scale);

    let tile_size = if args.tile_size == 0 {
        // Auto-select tile size based on CPU cores
        let cpu_cores = num_cpus::get();
        match cpu_cores {
            0..=2 => 32,
            3..=4 => 24,
            5..=8 => 16,
            9..=16 => 12,
            _ => 8,  // More than 16 cores
        }
    } else {
        args.tile_size.clamp(1, WIDTH.min(HEIGHT))
    };

    let tiles: Vec<Tile> = (0..HEIGHT)
        .step_by(tile_size)
        .flat_map(|y| {
            (0..WIDTH).step_by(tile_size).map(move |x| Tile {
                x_start: x,
                y_start: y,
                width: tile_size.min(WIDTH - x),
                height: tile_size.min(HEIGHT - y),
            })
        })
        .collect();

    if args.aa_samples > 1 {
        println!("Rendering scene with {}x anti-aliasing...", args.aa_samples);
    } else {
        println!("Rendering scene...");
    }

    let framebuffer_ptr = Arc::new(FramebufferPtr(framebuffer.as_mut_ptr()));
    let pb = Arc::new({
        let pb = ProgressBar::new(tiles.len() as u64);
        let style =
            ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})")
                .unwrap()
                .progress_chars("=>-");
        pb.set_style(style);
        pb.set_message("Rendering");
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    });
    tiles.par_iter().for_each(|tile| {
        let framebuffer_ptr = framebuffer_ptr.clone();
        let pb = pb.clone();
        for y in tile.y_start..(tile.y_start + tile.height) {
            let row_offset = y * WIDTH;
            for x in tile.x_start..(tile.x_start + tile.width) {
                let mut color = Vec3::ZERO;

                // Use adaptive sampling if enabled
                if args.adaptive_aa && aa_samples > 1 {
                    color = adaptive_sample_pixel(
                        &scene,
                        x, y,
                        aa_samples,
                        half_width, half_height, dir_z,
                        args.max_depth,
                        args.reflection_depth,
                        args.refraction_depth
                    );
                } else {
                    // Use Quasi-Monte Carlo sampling instead of regular grid
                    for sample in 0..(aa_samples * aa_samples) {
                        // Generate Halton sequence for better distribution
                        let jitter_x = halton_sequence(sample, 2); // base 2
                        let jitter_y = halton_sequence(sample, 3); // base 3

                        let dir_x = (x as f32 + jitter_x) - half_width;
                        let dir_y = -(y as f32 + jitter_y) + half_height;
                        let direction = Vec3::new(dir_x, dir_y, dir_z).normalize();
                        color += cast_ray_with_params(
                            &scene,
                            Vec3::ZERO,
                            direction,
                            0,
                            0,
                            0,
                            args.max_depth,
                            args.reflection_depth,
                            args.refraction_depth,
                        );
                    }
                    color *= inv_spp;
                }

                let idx = row_offset + x;
                unsafe {
                    *framebuffer_ptr.0.add(idx) = color;
                }
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("Rendering complete");

    println!("Saving image...");
    save_image(
        &framebuffer,
        WIDTH as u32,
        HEIGHT as u32,
        "output.png",
        args.tonemap,
    )?;

    println!("Render complete! Image saved as output.png");
    Ok(())
}

/// Add randomized objects to the scene.
fn add_random_objects(
    scene: &mut Scene,
    mesh_type: &str,
    include_spheres: bool,
    count: usize,
    seed: Option<u64>,
) {
    #[derive(Clone, Copy)]
    enum MeshKind {
        Cube,
        Pyramid,
        Torus,
    }

    #[derive(Clone, Copy)]
    enum ObjectKind {
        Sphere,
        Mesh(MeshKind),
    }

    fn random_unit_vec(rng: &mut fastrand::Rng) -> Vec3 {
        loop {
            let x = rng.f32() * 2.0 - 1.0;
            let y = rng.f32() * 2.0 - 1.0;
            let z = rng.f32() * 2.0 - 1.0;
            let v = Vec3::new(x, y, z);
            let len = v.length();
            if len > 1e-6 && len <= 1.0 {
                return v / len;
            }
        }
    }

    fn transform_mesh(base: &Mesh, scale: f32, rotation: Quat, translation: Vec3) -> Mesh {
        let vertices: Vec<Vec3> = base
            .vertices
            .iter()
            .map(|v| translation + rotation * (*v * scale))
            .collect();

        let normals: Vec<Vec3> = if base.normals.len() == base.vertices.len() {
            base.normals
                .iter()
                .map(|n| (rotation * *n).normalize())
                .collect()
        } else {
            base.normals.clone()
        };

        Mesh::with_normals(vertices, base.indices.clone(), normals)
    }

    let mut mesh_kinds = Vec::new();
    match mesh_type.to_lowercase().as_str() {
        "cube" => mesh_kinds.push(MeshKind::Cube),
        "pyramid" => mesh_kinds.push(MeshKind::Pyramid),
        "torus" => mesh_kinds.push(MeshKind::Torus),
        "all" => {
            mesh_kinds.push(MeshKind::Cube);
            mesh_kinds.push(MeshKind::Pyramid);
            mesh_kinds.push(MeshKind::Torus);
        }
        "none" => {}
        _ => {
            eprintln!(
                "Unknown mesh type: {}. Use: cube, pyramid, torus, all, none",
                mesh_type
            );
        }
    }

    let mut kinds = Vec::new();
    if include_spheres {
        kinds.push(ObjectKind::Sphere);
    }
    for kind in &mesh_kinds {
        kinds.push(ObjectKind::Mesh(*kind));
    }

    if kinds.is_empty() {
        return;
    }

    let base_cube = cube(1.0);
    let base_pyramid = pyramid(1.0, 1.25);
    let base_torus = torus(1.0, 0.35, 24, 12);

    let materials = [
        MATTE_RED,
        MATTE_GREEN,
        MATTE_BLUE,
        IVORY,
        RED_RUBBER,
        GLASS,
        MIRROR,
    ];

    let mut rng = match seed {
        Some(value) => fastrand::Rng::with_seed(value),
        None => fastrand::Rng::new(),
    };
    let per_kind = count / kinds.len();
    let mut pool = Vec::with_capacity(count);

    for kind in &kinds {
        for _ in 0..per_kind {
            pool.push(*kind);
        }
    }
    while pool.len() < count {
        pool.push(kinds[rng.usize(0..kinds.len())]);
    }

    for i in (1..pool.len()).rev() {
        let j = rng.usize(0..=i);
        pool.swap(i, j);
    }

    for kind in pool {
        let pos = Vec3::new(
            rng.f32() * 24.0 - 12.0,
            rng.f32() * 8.0 - 4.0,
            -6.0 - rng.f32() * 28.0,
        );

        let mat = materials[rng.usize(0..materials.len())];

        match kind {
            ObjectKind::Sphere => {
                let radius = 0.4 + rng.f32() * 2.4;
                scene.add_object(Object::sphere(pos, radius, mat));
            }
            ObjectKind::Mesh(mesh_kind) => {
                let scale = 0.6 + rng.f32() * 2.6;
                let axis = random_unit_vec(&mut rng);
                let angle = rng.f32() * std::f32::consts::TAU;
                let rotation = Quat::from_axis_angle(axis, angle);

                let mesh = match mesh_kind {
                    MeshKind::Cube => transform_mesh(&base_cube, scale, rotation, pos),
                    MeshKind::Pyramid => transform_mesh(&base_pyramid, scale, rotation, pos),
                    MeshKind::Torus => transform_mesh(&base_torus, scale, rotation, pos),
                };

                scene.add_object(Object::mesh(mesh, mat));
            }
        }
    }
}

/// Generate Halton sequence for Quasi-Monte Carlo sampling
fn halton_sequence(index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f32;
    let mut i = index;

    while i > 0 {
        result += (i % base) as f32 * fraction;
        i /= base;
        fraction /= base as f32;
    }

    result
}

/// Adaptive sampling based on pixel contrast
fn adaptive_sample_pixel(
    scene: &Scene,
    x: usize,
    y: usize,
    max_aa_samples: u32,
    half_width: f32,
    half_height: f32,
    dir_z: f32,
    max_depth: i32,
    reflection_depth: i32,
    refraction_depth: i32,
) -> Vec3 {
    // Calculate the number of samples based on contrast with neighbors
    let spp = max_aa_samples * max_aa_samples;
    let inv_spp = 1.0 / spp as f32;

    let mut color = Vec3::ZERO;
    for sample in 0..spp {
        let jitter_x = halton_sequence(sample, 2);
        let jitter_y = halton_sequence(sample, 3);

        let dir_x = (x as f32 + jitter_x) - half_width;
        let dir_y = -(y as f32 + jitter_y) + half_height;
        let direction = Vec3::new(dir_x, dir_y, dir_z).normalize();
        color += cast_ray_with_params(
            scene,
            Vec3::ZERO,
            direction,
            0,
            0,
            0,
            max_depth,
            reflection_depth,
            refraction_depth,
        );
    }

    color * inv_spp
}
