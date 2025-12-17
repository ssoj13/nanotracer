use std::path::Path;

use clap::Parser;
use glam::Vec3;
use nanotracer_rs::environment::EnvironmentMap;
use nanotracer_rs::geometry::{Object, Sphere};
use nanotracer_rs::material::{GLASS, IVORY, MATTE_BLUE, MATTE_GREEN, MATTE_RED, MIRROR, RED_RUBBER};
use nanotracer_rs::mesh::{cube, pyramid, torus};
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
    nanotracer-rs -a 8 -s                    # 8x AA + procedural sky
    nanotracer-rs -n hdr.exr -e 0.5          # HDR environment, exposure 0.5
    nanotracer-rs -a 4 -m 64 -r 8 -f 20      # High quality: more bounces

  With meshes:
    nanotracer-rs --mesh cube               # Add a cube mesh
    nanotracer-rs --mesh torus              # Add a torus mesh
    nanotracer-rs --mesh all                # Add all mesh primitives

  Gaussian Splats (fast preview):
    nanotracer-rs -S test.ply --splat-density 50 --sh-samples 16

  Gaussian Splats (balanced):
    nanotracer-rs -S scene.ply --splat-density 200 --sh-samples 64

  Gaussian Splats (high quality):
    nanotracer-rs -S hq.ply --splat-density 500 --sh-samples 128
"#)]
struct Args {
    /// Maximum recursion depth
    #[arg(short = 'm', long = "max", default_value_t = 32)]
    max_depth: i32,

    /// Maximum reflection depth
    #[arg(short = 'r', long = "refl", default_value_t = 6)]
    reflection_depth: i32,

    /// Maximum refraction depth
    #[arg(short = 'f', long = "refr", default_value_t = 16)]
    refraction_depth: i32,

    /// HDR environment map file (.exr format)
    #[arg(short = 'n', long = "env")]
    env_path: Option<String>,

    /// Use procedural sky gradient instead of solid background
    #[arg(short = 's', long = "sky")]
    use_sky: bool,

    /// Exposure adjustment for HDR environment maps
    #[arg(short = 'e', long = "exp", default_value_t = 0.1)]
    exposure: f32,

    /// Anti-aliasing samples per pixel
    #[arg(short = 'a', long = "aa", default_value_t = 1)]
    aa_samples: u32,

    /// Export Gaussian splats to PLY file (skips image rendering)
    #[arg(short = 'S', long = "splats")]
    splat_output: Option<String>,

    /// SH sampling directions per splat (default: 64)
    #[arg(long = "sh-samples", default_value_t = 64)]
    sh_samples: usize,

    /// Surface samples per unit area (default: 100)
    #[arg(long = "splat-density", default_value_t = 100.0)]
    splat_density: f32,

    /// Maximum SH degree 0-3 (default: 3)
    #[arg(long = "sh-degree", default_value_t = 3)]
    sh_degree: u32,

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 768;
    const FOV: f32 = 1.05;

    let mut scene = Scene::new();

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

    // Add default spheres unless disabled
    if !args.no_spheres {
        scene.add_sphere(Sphere::new(Vec3::new(-3.0, 0.0, -16.0), 2.0, IVORY));
        scene.add_sphere(Sphere::new(Vec3::new(-1.0, -1.5, -12.0), 2.0, GLASS));
        scene.add_sphere(Sphere::new(Vec3::new(1.5, -0.5, -18.0), 3.0, RED_RUBBER));
        scene.add_sphere(Sphere::new(Vec3::new(7.0, 5.0, -18.0), 4.0, MIRROR));
        scene.add_sphere(Sphere::new(Vec3::new(-2.0, 1.0, -6.0), 1.5, MIRROR));
        scene.add_sphere(Sphere::new(Vec3::new(2.5, -1.0, -7.5), 1.2, GLASS));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 2.5, -8.0), 1.0, IVORY));
        scene.add_sphere(Sphere::new(Vec3::new(-4.0, -2.0, -9.0), 1.8, RED_RUBBER));
        scene.add_sphere(Sphere::new(Vec3::new(3.0, 0.5, -5.5), 1.0, MIRROR));
    }

    // Add mesh primitives (default: all, or specified type)
    let mesh_type = args.mesh.as_deref().unwrap_or("all");
    if mesh_type != "none" {
        add_meshes(&mut scene, mesh_type);
    }

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

    // Splat generation mode
    if let Some(splat_path) = &args.splat_output {
        println!("Generating Gaussian splats...");
        println!("  Density: {} samples/unit^2", args.splat_density);
        println!("  SH samples: {}", args.sh_samples);
        println!("  SH degree: {}", args.sh_degree);

        let config = SplatConfig {
            density: args.splat_density,
            sh_samples: args.sh_samples,
            sh_degree: args.sh_degree.min(3),
            max_depth: args.max_depth,
            reflection_depth: args.reflection_depth,
            refraction_depth: args.refraction_depth,
            scale_override: args.splat_scale,
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

    if args.aa_samples > 1 {
        println!("Rendering scene with {}x anti-aliasing...", args.aa_samples);
    } else {
        println!("Rendering scene...");
    }

    framebuffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, pixel)| {
            let x = i % WIDTH;
            let y = i / WIDTH;

            let mut color = Vec3::ZERO;

            for sample_y in 0..args.aa_samples {
                for sample_x in 0..args.aa_samples {
                    let jitter_x = (sample_x as f32 + 0.5) / args.aa_samples as f32;
                    let jitter_y = (sample_y as f32 + 0.5) / args.aa_samples as f32;

                    let dir_x = (x as f32 + jitter_x) - WIDTH as f32 / 2.0;
                    let dir_y = -(y as f32 + jitter_y) + HEIGHT as f32 / 2.0;
                    let dir_z = -(HEIGHT as f32) / (2.0 * (FOV / 2.0).tan());

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
            }

            *pixel = color / (args.aa_samples * args.aa_samples) as f32;
        });

    println!("Saving image...");
    save_image(&framebuffer, WIDTH as u32, HEIGHT as u32, "output.png")?;

    println!("Render complete! Image saved as output.png");
    Ok(())
}

/// Add mesh primitives to the scene
fn add_meshes(scene: &mut Scene, mesh_type: &str) {
    match mesh_type.to_lowercase().as_str() {
        "cube" => {
            println!("Adding cube mesh...");
            let mesh = cube(2.0);
            println!("  {} triangles", mesh.tri_count());
            scene.add_object(Object::mesh(mesh, MATTE_RED));
        }
        "pyramid" => {
            println!("Adding pyramid mesh...");
            let mesh = pyramid(2.0, 2.5);
            println!("  {} triangles", mesh.tri_count());
            let verts: Vec<Vec3> = mesh.vertices.iter().map(|v| *v + Vec3::new(4.0, -4.0, -12.0)).collect();
            let shifted = nanotracer_rs::mesh::Mesh::with_normals(verts, mesh.indices.clone(), mesh.normals.clone());
            scene.add_object(Object::mesh(shifted, MATTE_GREEN));
        }
        "torus" => {
            println!("Adding torus mesh...");
            let mesh = torus(1.5, 0.5, 32, 16);
            println!("  {} triangles", mesh.tri_count());
            let verts: Vec<Vec3> = mesh.vertices.iter().map(|v| {
                let rotated = Vec3::new(v.x, v.z, -v.y);
                rotated + Vec3::new(-4.0, -2.5, -10.0)
            }).collect();
            let shifted = nanotracer_rs::mesh::Mesh::with_normals(verts, mesh.indices.clone(), mesh.normals.iter().map(|n| Vec3::new(n.x, n.z, -n.y)).collect());
            scene.add_object(Object::mesh(shifted, MATTE_BLUE));
        }
        "all" => {
            println!("Adding 5 random mesh primitives...");
            let mut rng = fastrand::Rng::new();
            
            let materials = [MATTE_RED, MATTE_GREEN, MATTE_BLUE, IVORY, RED_RUBBER];
            
            for i in 0..5 {
                // Random position in view
                let x = rng.f32() * 12.0 - 6.0;  // -6 to 6
                let y = rng.f32() * 4.0 - 4.0;   // -4 to 0 (on/above floor)
                let z = -8.0 - rng.f32() * 10.0; // -8 to -18
                let pos = Vec3::new(x, y, z);
                
                // Random size (doubled)
                let size = 1.6 + rng.f32() * 2.4; // 1.6 to 4.0
                
                // Random material
                let mat = materials[rng.usize(0..materials.len())];
                
                // Random primitive type
                let prim_type = rng.u32(0..3);
                
                let mesh = match prim_type {
                    0 => {
                        // Cube
                        let m = cube(size);
                        let verts: Vec<Vec3> = m.vertices.iter().map(|v| *v + pos).collect();
                        nanotracer_rs::mesh::Mesh::with_normals(verts, m.indices.clone(), m.normals.clone())
                    }
                    1 => {
                        // Pyramid
                        let m = pyramid(size, size * 1.3);
                        let verts: Vec<Vec3> = m.vertices.iter().map(|v| *v + pos).collect();
                        nanotracer_rs::mesh::Mesh::with_normals(verts, m.indices.clone(), m.normals.clone())
                    }
                    _ => {
                        // Torus (tilted randomly)
                        let m = torus(size * 0.6, size * 0.2, 24, 12);
                        let angle = rng.f32() * std::f32::consts::PI;
                        let cos_a = angle.cos();
                        let sin_a = angle.sin();
                        let verts: Vec<Vec3> = m.vertices.iter().map(|v| {
                            let rotated = Vec3::new(v.x, v.y * cos_a - v.z * sin_a, v.y * sin_a + v.z * cos_a);
                            rotated + pos
                        }).collect();
                        let normals: Vec<Vec3> = m.normals.iter().map(|n| {
                            Vec3::new(n.x, n.y * cos_a - n.z * sin_a, n.y * sin_a + n.z * cos_a).normalize()
                        }).collect();
                        nanotracer_rs::mesh::Mesh::with_normals(verts, m.indices.clone(), normals)
                    }
                };
                
                println!("  Mesh {}: {} tris at ({:.1}, {:.1}, {:.1})", 
                    i + 1, mesh.tri_count(), pos.x, pos.y, pos.z);
                scene.add_object(Object::mesh(mesh, mat));
            }
        }
        "none" => {}
        _ => {
            eprintln!("Unknown mesh type: {}. Use: cube, pyramid, torus, all, none", mesh_type);
        }
    }
}
