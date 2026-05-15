//! Vulkan runtime + Scene -> GPU buffer marshalling for `nanotracer-rs`.
//!
//! [`vk_runtime::VkContext`] owns the ash device/queue/command-pool/AS-loader
//! and provides helpers for buffer/image/AS allocation and shader-module
//! creation via shaderc. [`gpu_scene::build_gpu_scene_with_detail_boost`]
//! flattens a `nano_core::scene::Scene` into the SSBO/UBO layout the
//! compute shaders expect.

pub mod gpu_scene;
pub mod vk_runtime;
