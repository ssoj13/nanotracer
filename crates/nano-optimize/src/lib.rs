//! Gradient-based 3D Gaussian-splat optimizer for `nanotracer-rs`.
//!
//! Mirrors the architecture of the Inria 3DGS training pipeline (Kerbl et
//! al. 2023) and the Rust `brush` project — both are linked from
//! `WGPU_RESEARCH.md` and `plan1.md`. The split is intentional:
//!
//! - **Reference baking** ([`reference`]) reuses `nano-render` (ash + Vulkan
//!   ray queries) to produce ground-truth radiance frames from many camera
//!   angles. This stays on Vulkan because the reference path needs stable
//!   hardware-BVH ray queries.
//! - **Optimisation** ([`adam`], [`splat_store`], [`train`]) runs entirely
//!   on **wgpu** so the differentiable splat rasteriser (Phase A2) can
//!   target Vulkan / DX12 / Metal / WebGPU uniformly and so the canonical
//!   WGSL kernels from `brush` are portable.
//!
//! Phase A1 (this commit) is **scaffolding only**: types, file layout,
//! Adam state, training-loop skeleton, multi-view reference baking. The
//! differentiable forward and backward rasteriser, loss, and
//! densify-and-prune land in subsequent phases.

pub mod adam;
pub mod gpu;
pub mod prefix_scan;
pub mod radix_sort;
pub mod raster;
pub mod reference;
pub mod splat_gpu;
pub mod splat_store;
pub mod train;

pub use adam::AdamState;
pub use gpu::WgpuCtx;
pub use prefix_scan::PrefixScan;
pub use radix_sort::RadixSort;
pub use raster::{CameraUniform, ProjectedSplat, Rasterizer};
pub use reference::{ReferenceView, bake_references};
pub use splat_gpu::GpuSplatBuffer;
pub use splat_store::SplatBuffer;
pub use train::{TrainConfig, train};
