//! wgpu context + low-level buffer/readback helpers for `nano-optimize`.
//!
//! Phase A2 onward, every per-iteration tensor lives on the GPU (forward
//! rasteriser, gradients, Adam moments). This module is the thin layer
//! between bytemuck-flat host data and wgpu storage buffers — no math,
//! no kernels, just plumbing.
//!
//! The crate's WGSL kernels target a single [`WgpuCtx`] for the whole
//! training run; recreating the device per iteration would dwarf the
//! actual work. `train()` builds the context once and threads `&ctx`
//! into every pass.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// Bundled wgpu device handles. Hold one per training run.
///
/// The context is **GPU-only** — if no compatible adapter is available,
/// construction fails and the caller bails out. No CPU fallback path
/// exists by design: the rasteriser only makes sense at GPU speed.
pub struct WgpuCtx {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuCtx {
    /// Block on async init. Picks the highest-power adapter available;
    /// fails if none is found.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, Box<dyn std::error::Error>> {
        // Headless context — no display handle needed for compute work.
        // The env-aware helper still honours `WGPU_BACKEND`, validation
        // flags, and other tunables a user might want to override.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no compatible GPU adapter: {e}"))?;
        let info = adapter.get_info();
        eprintln!(
            "[nano-optimize] wgpu adapter: {} ({:?}, {:?})",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nano-optimize device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Storage buffer initialised from CPU data. Usage includes `COPY_SRC`
    /// so the buffer can be read back via [`Self::readback`].
    pub fn storage_buffer<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Empty storage buffer of the given byte size (cleared to zero).
    pub fn storage_buffer_zeroed(&self, label: &str, size_bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Uniform buffer initialised from a single `T`. Usage includes
    /// `COPY_DST` so `train()` can refresh the contents per iteration.
    pub fn uniform_buffer<T: Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Read a storage buffer back to the host as a `Vec<T>`. Uses a
    /// staging buffer because storage memory isn't host-mappable; safe
    /// to call repeatedly but expensive — only use during readback or
    /// for tests, not in the hot training path.
    pub fn readback<T: Pod>(&self, src: &wgpu::Buffer, count: usize) -> Vec<T> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll");
        rx.recv().expect("map_async channel").expect("map_async");
        let view = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }
}
