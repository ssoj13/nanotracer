//! GPU path-tracing image renderer for `nanotracer-rs`.
//!
//! Drives a Vulkan compute shader that path-traces the scene via ray queries
//! and writes a linear-RGB framebuffer back to the host. Reuses
//! `nano-shaders::PREAMBLE` + `HELPERS` for the chunks of GLSL that are
//! identical to the splat generator.

pub mod renderer;

pub use renderer::{RenderConfig, render};
