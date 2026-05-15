//! GPU Gaussian-splat generator and 3DGS-compatible binary PLY writer.
//!
//! - [`generator::generate_splats_gpu`] — fits an order-3 SH at each surface
//!   sample using a Vulkan compute shader.
//! - [`ply::write_ply`] — writes the result in the field order expected by
//!   SuperSplat / Luma / bevy_gaussian_splatting.

pub mod generator;
pub mod ply;

pub use generator::{SplatConfigGpu, generate_splats_gpu};
pub use ply::{Gaussian, write_ply};
