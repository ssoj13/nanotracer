use clap::Parser;
use glam::{Quat, Vec3};

use nano_core::LightSampling;
use nano_core::environment::EnvironmentMap;
use nano_core::geometry::Object;
use nano_core::material::{
    GLASS, IVORY, MATTE_BLUE, MATTE_GREEN, MATTE_RED, MATTE_WHITE, MIRROR, RED_RUBBER,
};
use nano_core::mesh::{Mesh, cube, pyramid, torus};
use nano_core::scene::{Light, Scene};
use nano_io::gltf_loader::load_glb_mesh;
use nano_io::utils::save_image;
use nano_render::{RenderConfig, render};
use nano_optimize::{TrainConfig, reference::BakeConfig, train as run_training};
use nano_optimize::adam::AdamConfig;
use nano_splat::{SplatConfigGpu, generate_splats_gpu, write_ply};

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

    /// Light sampling: all or one (default: one)
    #[arg(long = "light-sampling", default_value = "one")]
    light_sampling: String,

    /// Detail boost factor for adaptive splat density (default: 1.5, 0 disables)
    #[arg(long = "detail-boost", default_value_t = 1.5)]
    detail_boost: f32,

    /// Max detail boost factor (default: 3.0)
    #[arg(long = "detail-boost-max", default_value_t = 3.0)]
    detail_boost_max: f32,

    /// Surface samples per unit area (default: 100)
    #[arg(long = "splat-density", default_value_t = 100.0)]
    splat_density: f32,

    /// Override splat scale (radius). Auto-calculated from density if not set
    #[arg(long = "splat-scale")]
    splat_scale: Option<f32>,

    /// Keep view-dependent SH coefficients for reflective/refractive materials.
    /// By default mirror/glass splats use DC only to avoid order-3 SH ringing;
    /// turning this on reintroduces the ringing but preserves some directional
    /// reflection cues. Useful for stylised splat output.
    #[arg(long = "sh-keep-glossy", default_value_t = false)]
    sh_keep_glossy: bool,

    /// Env-map IBL intensity, applied to the implicit `Light::Env`
    /// when an environment is loaded. 0 disables IBL; 1.0 is the full
    /// Lambertian-convolved physical value (tends to over-fill shadows
    /// for our non-physical "unit-radiance" direct-light convention).
    /// 0.15–0.3 reads well as a soft ambient.
    #[arg(long = "env-light", default_value_t = 0.15)]
    env_light: f32,

    /// Add a point light. Repeat for multiple lights.
    /// Format: `x,y,z,r,g,b,intensity` — 7 floats.
    #[arg(long = "point-light", value_name = "x,y,z,r,g,b,i")]
    point_lights: Vec<String>,

    /// Add a rectangle area light. `u` and `v` are world-space
    /// half-extent vectors (a 4×6 rect with u along +X, v along +Y is
    /// `u=2,0,0  v=0,3,0`). Optional `two_sided` suffix `0`/`1`.
    /// Format: `cx,cy,cz,ux,uy,uz,vx,vy,vz,r,g,b,intensity[,two_sided]`.
    #[arg(long = "rect-light", value_name = "cx,cy,cz,ux,uy,uz,vx,vy,vz,r,g,b,i[,two]")]
    rect_lights: Vec<String>,

    /// Add a sphere area light.
    /// Format: `cx,cy,cz,radius,r,g,b,intensity` — 8 floats.
    #[arg(long = "sphere-light", value_name = "cx,cy,cz,radius,r,g,b,i")]
    sphere_lights: Vec<String>,

    /// Add an oriented-box area light. Rotation is given as a unit
    /// quaternion `(x, y, z, w)`. Half-extents are along the box's
    /// local +X/+Y/+Z before rotation.
    /// Format: `cx,cy,cz,hx,hy,hz,qx,qy,qz,qw,r,g,b,intensity` — 14 floats.
    #[arg(long = "box-light", value_name = "cx,cy,cz,hx,hy,hz,qx,qy,qz,qw,r,g,b,i")]
    box_lights: Vec<String>,

    /// Run the gradient-based splat optimiser after the forward fit.
    /// Requires `-S/--splats FILE` for the output path. Phase A1 — wires
    /// scaffolding only; forward / backward rasteriser and Adam updates
    /// land in later phases.
    #[arg(long = "train", default_value_t = false)]
    train: bool,

    /// Number of optimiser iterations.
    #[arg(long = "train-iters", default_value_t = 30000)]
    train_iters: u32,

    /// Number of reference frames baked at training start.
    #[arg(long = "train-views", default_value_t = 50)]
    train_views: u32,

    /// Reference-frame width / height for training (smaller = faster bake).
    #[arg(long = "train-width", default_value_t = 512)]
    train_width: u32,
    #[arg(long = "train-height", default_value_t = 384)]
    train_height: u32,

    /// Hard cap on optimiser splat count (densify won't exceed this).
    #[arg(long = "train-max-splats", default_value_t = 5_000_000)]
    train_max_splats: usize,

    /// SSIM blend factor for combined loss
    /// `L = (1 − λ) · MSE + λ · (1 − SSIM)`. 0.0 = pure MSE, 1.0 = pure
    /// SSIM. Inria 3DGS default is 0.2. Range-checked at parse time so
    /// out-of-range values fail loudly instead of silently clamping.
    #[arg(long = "ssim-lambda", default_value_t = 0.2, value_parser = parse_unit_interval)]
    ssim_lambda: f32,

    /// Add mesh primitives: cube, pyramid, torus, all
    #[arg(long = "mesh")]
    mesh: Option<String>,

    /// Load a glTF/GLB mesh and add to the scene
    #[arg(long = "glb")]
    glb_path: Option<String>,

    /// Scale applied to the loaded GLB mesh (default: 1.0)
    #[arg(long = "glb-scale", default_value_t = 1.0)]
    glb_scale: f32,

    /// Disable checkerboard plane
    #[arg(long = "no-floor")]
    no_floor: bool,

    /// Disable default spheres (mesh-only mode)
    #[arg(long = "no-spheres")]
    no_spheres: bool,

    /// Open the interactive splat viewer after generating / loading
    /// the splats (via `--splats` or `--train`). Reuses the WGSL
    /// rasteriser; orbit-cam (LMB drag + scroll zoom), ESC to quit.
    #[arg(long = "view", default_value_t = false)]
    view: bool,

    /// Load a 3DGS PLY from disk and open the interactive viewer.
    /// Bypasses scene generation / training entirely.
    #[arg(long = "view-ply")]
    view_ply: Option<String>,

    /// Open the viewer with a live training preview. Spawns the
    /// `train()` loop on a worker thread; the viewer polls a shared
    /// snapshot every frame and updates the splat texture as Adam +
    /// densify reshape the cloud. Implies `--train`.
    #[arg(long = "view-training", default_value_t = false)]
    view_training: bool,
}

/// Clap value-parser that accepts only `[0.0, 1.0]` — used by
/// `--ssim-lambda` so out-of-range values fail loudly at parse time
/// rather than getting silently clamped inside the loss function.
fn parse_unit_interval(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("not a float: {e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be in [0.0, 1.0], got {v}"))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Fast path: open the viewer directly on a PLY file. Skips scene
    // generation, raytracing, and forward-fit entirely.
    if let Some(ply_path) = &args.view_ply {
        println!("Loading PLY: {}", ply_path);
        let gaussians = nano_splat::read_ply(std::path::Path::new(ply_path))?;
        println!("Loaded {} gaussians. Opening viewer (ESC to quit)...", gaussians.len());
        let buf = nano_optimize::SplatBuffer::from_gaussians(&gaussians);
        nano_view::run(buf)?;
        return Ok(());
    }

    if let Some(info) = gpu_mem::query() {
        let mib = |b: u64| b / (1024 * 1024);
        println!(
            "GPU: {} ({} MiB VRAM, {} MiB free){}",
            info.name,
            mib(info.dedicated_vram),
            mib(info.free_vram),
            if info.unified { ", unified" } else { "" },
        );
    }

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

    // Add the implicit env-light whenever an environment is bound, at
    // the user-controlled intensity. nano-gpu's build also auto-adds one
    // at intensity 1.0 if neither this nor an explicit `Light::Env` is
    // present — that path covers library callers (tests, training).
    if scene.environment.is_some() && args.env_light > 0.0 {
        scene.add_light(Light::Env {
            intensity: args.env_light,
        });
    }

    if args.no_floor {
        scene.checkerboard_enabled = false;
    }

    if let Some(glb_path) = &args.glb_path {
        let mesh_path = std::path::Path::new(glb_path);
        println!("Loading GLB mesh: {} (scale {})", glb_path, args.glb_scale);
        match load_glb_mesh(mesh_path, args.glb_scale) {
            Ok(mesh) => {
                scene.add_object(Object::mesh(mesh, MATTE_WHITE));
            }
            Err(err) => {
                eprintln!("Warning: Failed to load GLB mesh: {}", err);
            }
        }
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

    // Add lights — `Light::point` produces the legacy unit-radiance
    // demo light. Per-light colour/intensity are available via the
    // explicit `Light::Point { .. }` constructor or the `--*-light`
    // CLI flags below.
    scene.add_light(Light::point(Vec3::new(-20.0, 20.0, 20.0)));
    scene.add_light(Light::point(Vec3::new(30.0, 50.0, -25.0)));
    scene.add_light(Light::point(Vec3::new(30.0, 20.0, 30.0)));

    for s in &args.point_lights {
        let v = parse_floats(s, "point-light", &[7]);
        scene.add_light(Light::Point {
            position: Vec3::new(v[0], v[1], v[2]),
            color: Vec3::new(v[3], v[4], v[5]),
            intensity: v[6],
        });
    }
    for s in &args.rect_lights {
        let v = parse_floats(s, "rect-light", &[13, 14]);
        let two_sided = v.get(13).map(|x| *x != 0.0).unwrap_or(false);
        scene.add_light(Light::Rect {
            center: Vec3::new(v[0], v[1], v[2]),
            u: Vec3::new(v[3], v[4], v[5]),
            v: Vec3::new(v[6], v[7], v[8]),
            color: Vec3::new(v[9], v[10], v[11]),
            intensity: v[12],
            two_sided,
        });
    }
    for s in &args.sphere_lights {
        let v = parse_floats(s, "sphere-light", &[8]);
        scene.add_light(Light::Sphere {
            center: Vec3::new(v[0], v[1], v[2]),
            radius: v[3],
            color: Vec3::new(v[4], v[5], v[6]),
            intensity: v[7],
        });
    }
    for s in &args.box_lights {
        let v = parse_floats(s, "box-light", &[14]);
        scene.add_light(Light::Box {
            center: Vec3::new(v[0], v[1], v[2]),
            half_extents: Vec3::new(v[3], v[4], v[5]),
            // Normalise the input quaternion defensively — small
            // floating-point drift in user-supplied rotations would
            // otherwise show up as a slow shear of the box.
            rotation: Quat::from_xyzw(v[6], v[7], v[8], v[9]).normalize(),
            color: Vec3::new(v[10], v[11], v[12]),
            intensity: v[13],
        });
    }

    if let Some(splat_path) = &args.splat_output {
        println!("GPU splat generation...");
        // Note: `--light-sampling` is intentionally ignored on the splat path —
        // the SH fitter inside the shader always sums all lights to keep the
        // hemisphere LSQ low-variance (per-direction light pick would show up
        // as 'splotched colour' speckle in the fitted SH).
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
            detail_boost: args.detail_boost,
            detail_boost_max: args.detail_boost_max,
            keep_glossy_sh: args.sh_keep_glossy,
            seed: scene_seed,
        };

        let gaussians = if args.train || args.view_training {
            println!(
                "Training enabled: {} iterations, {} reference views at {}×{}",
                args.train_iters, args.train_views, args.train_width, args.train_height,
            );
            let train_cfg = TrainConfig {
                iterations: args.train_iters,
                max_splats: args.train_max_splats,
                reference: BakeConfig {
                    views: args.train_views,
                    width: args.train_width,
                    height: args.train_height,
                    ..BakeConfig::default()
                },
                seed: config,
                // Inria-style per-attribute lrs — position needs a noticeably
                // higher rate than the other channels to escape its forward-fit
                // start; SH / opacity / scale stay conservative.
                adam_pos: AdamConfig {
                    lr: 1.6e-4,
                    ..AdamConfig::default()
                },
                adam_attr: AdamConfig::default(),
                ssim_lambda: args.ssim_lambda,
            };
            if args.view_training {
                println!(
                    "Opening live-training viewer (ESC to quit; closes detaches worker)..."
                );
                // run_with_training blocks until window close; the
                // background thread runs train() to completion (or is
                // detached when the window closes). On exit we don't
                // try to write a PLY — that's the trainer's own
                // responsibility if it finishes first.
                nano_view::run_with_training(scene, train_cfg)?;
                return Ok(());
            }
            let splats = run_training(&scene, &train_cfg, |_, _, _| {})?;
            splats.to_gaussians()
        } else {
            generate_splats_gpu(&scene, &config)?
        };
        let path = std::path::Path::new(splat_path);
        println!("Writing {} gaussians to {}...", gaussians.len(), splat_path);
        write_ply(path, &gaussians)?;
        println!("Splat generation complete");

        if args.view {
            println!("Opening interactive viewer (ESC to quit)...");
            let buf = nano_optimize::SplatBuffer::from_gaussians(&gaussians);
            nano_view::run(buf)?;
        }
    } else {
        println!("GPU renderer: Vulkan ray query");
        println!("Resolution: {}x{}", WIDTH, HEIGHT);
        println!("AA: {}x{}", args.aa_samples, args.aa_samples);
        let light_sampling = match args.light_sampling.as_str() {
            "all" => LightSampling::All,
            _ => LightSampling::One,
        };

        let (camera_pos, camera_target, camera_up) = RenderConfig::default_camera();
        let config = RenderConfig {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            fov: FOV,
            camera_pos,
            camera_target,
            camera_up,
            aa_samples: args.aa_samples,
            max_depth: args.max_depth,
            reflection_depth: args.reflection_depth,
            refraction_depth: args.refraction_depth,
            tonemap: args.tonemap,
            light_sampling,
        };

        println!("Rendering on GPU...");
        let framebuffer = render(&scene, &config)?;

        println!("Saving image...");
        save_image(&framebuffer, WIDTH as u32, HEIGHT as u32, "output.png", args.tonemap)?;

        println!("Image saved as output.png");
    }
    Ok(())
}

/// Parse a comma-separated list of floats. `allowed_lens` is the set of
/// acceptable token counts — multiple values let an optional trailing
/// boolean flag (encoded as `0`/`1`) coexist with a fixed prefix.
/// Bails out with a clear panic message so CLI typos surface loudly
/// instead of silently picking a wrong default.
fn parse_floats(s: &str, flag: &str, allowed_lens: &[usize]) -> Vec<f32> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|t| {
            t.trim().parse::<f32>().unwrap_or_else(|_| {
                panic!("--{flag}: cannot parse '{t}' as float (input: {s:?})")
            })
        })
        .collect();
    if !allowed_lens.contains(&parts.len()) {
        panic!(
            "--{flag}: expected {} float tokens, got {} (input: {:?})",
            allowed_lens
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/"),
            parts.len(),
            s
        );
    }
    parts
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
