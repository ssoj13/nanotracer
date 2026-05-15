//! Asset I/O for `nanotracer-rs`.
//!
//! - [`gltf_loader::load_glb_mesh`] — pull a triangle mesh out of a `.glb`.
//! - [`utils::save_image`] — write a Vec3 framebuffer to PNG (tonemap + sRGB).

pub mod gltf_loader;
pub mod utils;
