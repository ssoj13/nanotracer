//! Scene description and CPU-only data model for `nanotracer-rs`.
//!
//! This crate contains no GPU code and no I/O. It owns the canonical types
//! that the rest of the workspace builds on: [`scene::Scene`],
//! [`geometry::Object`], [`mesh::Mesh`], [`material::Material`],
//! [`environment::EnvironmentMap`], plus colour-space helpers and the
//! shared [`LightSampling`] knob.

pub mod color;
pub mod environment;
pub mod geometry;
pub mod material;
pub mod mesh;
pub mod scene;
pub mod sh;

/// Lighting strategy used by the GPU shading code paths.
///
/// `All` sums contributions from every scene light per shade evaluation.
/// `One` picks a single random light per evaluation and weighs its
/// contribution by the light count (unbiased Monte-Carlo estimator).
///
/// `All` is the right choice for low-variance estimates and is the only mode
/// the splat fitter uses (its LSQ over hemisphere samples is sensitive to
/// per-direction noise). `One` is a path-trace optimisation for the image
/// renderer where variance averages across pixel samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightSampling {
    All,
    One,
}

impl LightSampling {
    /// Encode as a u32 for the GPU uniform buffer (0 = All, 1 = One).
    pub fn as_u32(self) -> u32 {
        match self {
            LightSampling::All => 0,
            LightSampling::One => 1,
        }
    }
}
