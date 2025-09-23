use rayon::prelude::*;
use clap::Parser;
use nanotracer::vec3::{Vector3, Vec3Ext};
use nanotracer::material::{IVORY, GLASS, RED_RUBBER, MIRROR};
use nanotracer::geometry::Sphere;
use nanotracer::scene::{Scene, Light};
use nanotracer::renderer::cast_ray_with_params;
use nanotracer::utils::save_image;
use nanotracer::environment::EnvironmentMap;

#[derive(Parser)]
#[command(name = "nanotracer")]
#[command(about = "A simple raytracer")]
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 768;
    const FOV: f32 = 1.05; // 60 degrees field of view in radians
    
    // Create scene
    let mut scene = Scene::new();

    // Set up environment based on arguments
    if let Some(env_path) = &args.env_path {
        println!("Loading HDR environment map: {} (exposure: {})", env_path, args.exposure);
        match EnvironmentMap::from_exr(env_path, args.exposure) {
            Ok(env_map) => {
                println!("Loaded {}x{} HDR environment map", env_map.width(), env_map.height());
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
    
    // Add spheres with different materials
    scene.add_sphere(Sphere::new(
        Vector3::new(-3.0, 0.0, -16.0),
        2.0,
        IVORY,
    ));
    
    scene.add_sphere(Sphere::new(
        Vector3::new(-1.0, -1.5, -12.0),
        2.0,
        GLASS,
    ));
    
    scene.add_sphere(Sphere::new(
        Vector3::new(1.5, -0.5, -18.0),
        3.0,
        RED_RUBBER,
    ));
    
    scene.add_sphere(Sphere::new(
        Vector3::new(7.0, 5.0, -18.0),
        4.0,
        MIRROR,
    ));

    // Add more spheres closer to camera (5-10 units away)
    scene.add_sphere(Sphere::new(
        Vector3::new(-2.0, 1.0, -6.0),
        1.5,
        MIRROR,
    ));

    scene.add_sphere(Sphere::new(
        Vector3::new(2.5, -1.0, -7.5),
        1.2,
        GLASS,
    ));

    scene.add_sphere(Sphere::new(
        Vector3::new(0.0, 2.5, -8.0),
        1.0,
        IVORY,
    ));

    scene.add_sphere(Sphere::new(
        Vector3::new(-4.0, -2.0, -9.0),
        1.8,
        RED_RUBBER,
    ));

    scene.add_sphere(Sphere::new(
        Vector3::new(3.0, 0.5, -5.5),
        1.0,
        MIRROR,
    ));
    
    // Add lights
    scene.add_light(Light {
        position: Vector3::new(-20.0, 20.0, 20.0),
    });
    
    scene.add_light(Light {
        position: Vector3::new(30.0, 50.0, -25.0),
    });
    
    scene.add_light(Light {
        position: Vector3::new(30.0, 20.0, 30.0),
    });
    
    // Create framebuffer
    let mut framebuffer = vec![Vector3::ZERO; WIDTH * HEIGHT];
    
    // Render scene in parallel with anti-aliasing
    if args.aa_samples > 1 {
        println!("Rendering scene with {}x anti-aliasing...", args.aa_samples);
    } else {
        println!("Rendering scene...");
    }

    framebuffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let x = i % WIDTH;
        let y = i / WIDTH;

        let mut color = Vector3::ZERO;

        // Anti-aliasing: sample multiple sub-pixels
        for sample_y in 0..args.aa_samples {
            for sample_x in 0..args.aa_samples {
                // Jittered sub-pixel sampling
                let jitter_x = (sample_x as f32 + 0.5) / args.aa_samples as f32;
                let jitter_y = (sample_y as f32 + 0.5) / args.aa_samples as f32;

                let dir_x = (x as f32 + jitter_x) - WIDTH as f32 / 2.0;
                let dir_y = -(y as f32 + jitter_y) + HEIGHT as f32 / 2.0; // Flip image
                let dir_z = -(HEIGHT as f32) / (2.0 * (FOV / 2.0).tan());

                let direction = Vector3::new(dir_x, dir_y, dir_z).normalized();
                color += cast_ray_with_params(&scene, Vector3::ZERO, direction, 0, 0, 0, args.max_depth, args.reflection_depth, args.refraction_depth);
            }
        }

        // Average all samples
        *pixel = color / (args.aa_samples * args.aa_samples) as f32;
    });
    
    // Save image
    println!("Saving image...");
    save_image(&framebuffer, WIDTH as u32, HEIGHT as u32, "output.png")?;
    
    println!("Render complete! Image saved as output.png");
    Ok(())
}
