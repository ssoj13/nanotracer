use clap::Parser;
use glam::{Quat, Vec3};
use nanotracer_rs::environment::EnvironmentMap;
use nanotracer_rs::geometry::Object;
use nanotracer_rs::material::{
    GLASS, IVORY, MATTE_BLUE, MATTE_GREEN, MATTE_RED, MIRROR, RED_RUBBER,
};
use nanotracer_rs::mesh::{cube, pyramid, torus, Mesh};
use nanotracer_rs::rt_renderer::{render, RenderConfig};
use nanotracer_rs::rt_splats::{generate_splats_gpu, SplatConfigGpu};
use nanotracer_rs::scene::{Light, Scene};
use nanotracer_rs::splat_gpu::write_ply;
use nanotracer_rs::utils::save_image;

#[derive(Parser)]
#[command(name = "nanotracer")]
#[command(about = "A path tracer with GPU ray query (Vulkan)")]
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

  Gaussian Splats (GPU):
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

    /// Apply tonemapping (Reinhard) before writing colors
    #[arg(short = 't', long = "tonemap", default_value_t = true)]
    tonemap: bool,

    /// Export Gaussian splats to PLY file (GPU mode)
    #[arg(short = 'S', long = "splats")]
    splat_output: Option<String>,

    /// SH sampling directions per splat (default: 64)
    #[arg(long = "sh-samples", default_value_t = 64)]
    sh_samples: u32,

    /// Multiplier for SH samples on glossy/refractive materials (default: 1.5)
    #[arg(long = "sh-glossy-mult", default_value_t = 1.5)]
    sh_glossy_mult: f32,

    /// Clamp radiance before tonemapping (0 disables, default: 20)
    #[arg(long = "radiance-clamp", default_value_t = 20.0)]
    radiance_clamp: f32,

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
}

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

    if args.no_floor {
        scene.checkerboard_enabled = false;
    }

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

    if let Some(splat_path) = &args.splat_output {
        println!("GPU splat generation...");
        let config = SplatConfigGpu {
            density: args.splat_density,
            sh_samples: args.sh_samples,
            max_depth: args.max_depth,
            reflection_depth: args.reflection_depth,
            refraction_depth: args.refraction_depth,
            scale_override: args.splat_scale,
            tonemap: args.tonemap,
            glossy_mult: args.sh_glossy_mult,
            radiance_clamp: args.radiance_clamp,
            seed: scene_seed,
        };
        let gaussians = generate_splats_gpu(&scene, &config)?;
        let path = std::path::Path::new(splat_path);
        println!("Writing {} gaussians to {}...", gaussians.len(), splat_path);
        write_ply(path, &gaussians)?;
        println!("Splat generation complete");
    } else {
        println!("GPU renderer: Vulkan ray query");
        println!("Resolution: {}x{}", WIDTH, HEIGHT);
        println!("AA: {}x{}", args.aa_samples, args.aa_samples);

        let config = RenderConfig {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            fov: FOV,
            aa_samples: args.aa_samples,
            max_depth: args.max_depth,
            reflection_depth: args.reflection_depth,
            refraction_depth: args.refraction_depth,
            tonemap: args.tonemap,
        };

        println!("Rendering on GPU...");
        let framebuffer = render(&scene, &config)?;

        println!("Saving image...");
        save_image(&framebuffer, WIDTH as u32, HEIGHT as u32, "output.png", args.tonemap)?;

        println!("Image saved as output.png");
    }
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
