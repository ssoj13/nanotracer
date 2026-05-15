//! Physically-plausible material library for `nanotracer-rs`.
//!
//! # Albedo channel convention
//!
//! `albedo` is laid out as `[kd, ks, kr, kt]`:
//!
//! | Index | Symbol | Meaning |
//! |-------|--------|---------|
//! | `albedo[0]` | `kd` | Lambertian diffuse weight (≤ 1)            |
//! | `albedo[1]` | `ks` | Specular highlight weight (Phong term)      |
//! | `albedo[2]` | `kr` | Recursive mirror reflection weight (≤ 1)    |
//! | `albedo[3]` | `kt` | Recursive refraction / transmission weight  |
//!
//! Energy conservation rule of thumb: `kd + kr + kt ≤ 1`. `ks` is a
//! highlight intensity that scales the unnormalised Phong lobe; for ideal
//! mirrors / dielectrics the visible "shine" comes from `kr` and `kt` so
//! `ks` is small or zero.
//!
//! All `diffuse_color` values are in linear sRGB primaries in `[0, 1]`.
//!
//! These values differ from the tinyraytracer originals (which had
//! `albedo[1] = 16` on `MIRROR` etc.) — the new constants are
//! energy-conserving and stay well behaved under tonemapping. The shading
//! model in both the image renderer and the splat fitter is still
//! unnormalised Phong; future PBR upgrades (GGX / Cook–Torrance) can change
//! the *interpretation* of these constants without changing the storage
//! layout.

use glam::Vec3;

/// Surface material consumed by the GPU shaders.
///
/// Layout is kept identical to the GPU-side `Material` struct (see
/// `nano-gpu::gpu_scene::GpuMaterial`) so it can be marshalled with no
/// reinterpretation.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    /// Refractive index used by the refraction path.
    pub refractive_index: f32,
    /// `[kd, ks, kr, kt]` — see module docs.
    pub albedo: [f32; 4],
    /// Linear sRGB diffuse colour.
    pub diffuse_color: Vec3,
    /// Phong specular exponent (highlight tightness).
    pub specular_exponent: f32,
}

// ── Stock materials ─────────────────────────────────────────────────────────
//
// All `ks` values are set to ≈ F₀ for a typical dielectric (≈ 0.04). Both
// shaders apply the normalised-Phong factor `(n+2)/(2π)` so the integrated
// energy in the specular lobe equals `ks` regardless of `specular_exponent`
// — i.e. `ks` is now an honest energy weight, not a peak-height knob.

/// F₀ for a typical dielectric (water/glass/most paints) at normal incidence.
const F0_DIELECTRIC: f32 = 0.04;

/// Warm off-white with a soft highlight and a hint of reflection.
pub const IVORY: Material = Material {
    refractive_index: 1.0,
    albedo: [0.85, F0_DIELECTRIC, 0.10, 0.00],
    diffuse_color: Vec3::new(0.85, 0.85, 0.75),
    specular_exponent: 50.0,
};

/// Clear glass: Fresnel-blended reflection/refraction (handled in the shader).
/// `kr` + `kt` give the *total* dielectric contribution (≈ 0.95); the actual
/// reflection/refraction split is angle-dependent via Schlick at runtime.
pub const GLASS: Material = Material {
    refractive_index: 1.5,
    albedo: [0.00, F0_DIELECTRIC, 0.10, 0.85],
    diffuse_color: Vec3::new(0.95, 0.97, 1.00),
    specular_exponent: 300.0,
};

/// Saturated red rubber — high diffuse, small highlight, no reflection.
pub const RED_RUBBER: Material = Material {
    refractive_index: 1.0,
    albedo: [0.90, F0_DIELECTRIC, 0.00, 0.00],
    diffuse_color: Vec3::new(0.65, 0.20, 0.20),
    specular_exponent: 10.0,
};

/// Near-perfect metallic mirror — reflection carries the look.
pub const MIRROR: Material = Material {
    refractive_index: 1.0,
    albedo: [0.00, F0_DIELECTRIC, 0.96, 0.00],
    diffuse_color: Vec3::new(1.0, 1.0, 1.0),
    specular_exponent: 1500.0,
};

// ── Matte family (shared kd / ks, varying colour) ──────────────────────────

const MATTE_ALBEDO: [f32; 4] = [0.95, F0_DIELECTRIC, 0.00, 0.00];
const MATTE_EXP: f32 = 20.0;

pub const MATTE_WHITE: Material = Material {
    refractive_index: 1.0,
    albedo: MATTE_ALBEDO,
    diffuse_color: Vec3::new(0.85, 0.85, 0.85),
    specular_exponent: MATTE_EXP,
};

pub const MATTE_RED: Material = Material {
    refractive_index: 1.0,
    albedo: MATTE_ALBEDO,
    diffuse_color: Vec3::new(0.85, 0.20, 0.20),
    specular_exponent: MATTE_EXP,
};

pub const MATTE_BLUE: Material = Material {
    refractive_index: 1.0,
    albedo: MATTE_ALBEDO,
    diffuse_color: Vec3::new(0.20, 0.30, 0.85),
    specular_exponent: MATTE_EXP,
};

pub const MATTE_GREEN: Material = Material {
    refractive_index: 1.0,
    albedo: MATTE_ALBEDO,
    diffuse_color: Vec3::new(0.20, 0.85, 0.30),
    specular_exponent: MATTE_EXP,
};

/// Build a matte checkerboard tile with the given diffuse colour.
///
/// The `MATERIAL_FLAG_CHECKERBOARD` bit set by the GPU side causes the
/// shader to override `diffuse_color` from a procedural pattern, so the
/// colour passed here only affects the fallback.
pub fn checkerboard_material(diffuse_color: Vec3) -> Material {
    Material {
        refractive_index: 1.0,
        albedo: [0.9, 0.1, 0.0, 0.0],
        diffuse_color,
        specular_exponent: 10.0,
    }
}
