//! Interactive splat viewer — winit window + wgpu surface, reuses the
//! `nano-optimize` rasteriser (project + bin + composite) per frame.
//!
//! v1 takes a CPU `SplatBuffer` (e.g. fresh from the forward fit) and
//! renders it in an orbit-camera window. Mouse-LMB drag rotates around
//! the scene centroid, scroll-wheel zooms. ESC quits.
//!
//! PLY-on-disk loading is a follow-up (`B1.x` — `nano-splat` already
//! has the `Gaussian` Pod, just needs a reader).
//!
//! Surface presentation uses a CPU readback path: composite → readback
//! `vec4<f32>` buffer → CPU tonemap + sRGB-encode to `u8` → `write_texture`
//! to the swapchain → present. Trades ~3 MiB/frame of host-side copy
//! for zero new shaders, and stays well above 60 fps at 1024×768 on a
//! discrete GPU. A future revision can land a `tonemap.wgsl` + blit
//! fragment shader to keep everything on the GPU.

use std::sync::Arc;

use glam::Vec3;
use nano_optimize::raster::CameraUniform;
use nano_optimize::splat_gpu::GpuSplatBuffer;
use nano_optimize::{Rasterizer, SplatBuffer, TileBinner, TilingParams, WgpuCtx};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Open the viewer window on `splats`. Blocks until the user closes
/// the window. The splat buffer is uploaded to the GPU once at startup
/// and is read by the rasteriser every frame.
pub fn run(splats: SplatBuffer) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ViewerApp::new(splats);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Orbit camera around the scene centroid. `target` is the look-at
/// point; `azimuth` rotates in the XZ plane; `elevation` tilts above
/// the plane; `distance` is the world-space radius.
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
}

impl OrbitCamera {
    fn position(&self) -> Vec3 {
        let cos_e = self.elevation.cos();
        Vec3::new(
            self.target.x + self.distance * cos_e * self.azimuth.cos(),
            self.target.y + self.distance * self.elevation.sin(),
            self.target.z + self.distance * cos_e * self.azimuth.sin(),
        )
    }
}

#[derive(Default)]
struct DragState {
    active: bool,
    last_x: f64,
    last_y: f64,
}

struct ViewerApp {
    splats: Option<SplatBuffer>,
    window: Option<Arc<Window>>,
    state: Option<State>,
    orbit: OrbitCamera,
    drag: DragState,
}

impl ViewerApp {
    fn new(splats: SplatBuffer) -> Self {
        // Place the orbit centre at the scene centroid so the first
        // view sees the cloud. Distance 1.5× the bbox extent gives a
        // comfortable framing.
        let (centroid, extent) = scene_centroid_extent(&splats);
        let distance = (extent.max_element() * 1.5).max(2.0);
        Self {
            splats: Some(splats),
            window: None,
            state: None,
            orbit: OrbitCamera {
                target: centroid,
                distance,
                azimuth: 0.0,
                elevation: 0.3,
            },
            drag: DragState::default(),
        }
    }
}

fn scene_centroid_extent(splats: &SplatBuffer) -> (Vec3, Vec3) {
    if splats.is_empty() {
        return (Vec3::ZERO, Vec3::ONE);
    }
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in &splats.positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    ((min + max) * 0.5, max - min)
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("nano-view")
            .with_inner_size(PhysicalSize::new(1024u32, 768u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());
        let splats = self.splats.take().expect("splats only consumed once");
        self.state = Some(State::new(window.clone(), splats).expect("State::new"));
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(window), Some(state)) = (self.window.as_ref(), self.state.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size);
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button: MouseButton::Left,
                ..
            } => {
                self.drag.active = matches!(btn_state, ElementState::Pressed);
                if !self.drag.active {
                    self.drag.last_x = 0.0;
                    self.drag.last_y = 0.0;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.drag.active {
                    if self.drag.last_x != 0.0 || self.drag.last_y != 0.0 {
                        let dx = (position.x - self.drag.last_x) as f32;
                        let dy = (position.y - self.drag.last_y) as f32;
                        self.orbit.azimuth += dx * 0.005;
                        self.orbit.elevation =
                            (self.orbit.elevation - dy * 0.005).clamp(-1.5, 1.5);
                    }
                    self.drag.last_x = position.x;
                    self.drag.last_y = position.y;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let s = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.05,
                };
                self.orbit.distance = (self.orbit.distance * (1.0 - s * 0.1)).clamp(0.5, 500.0);
            }
            WindowEvent::RedrawRequested => {
                state.render(&self.orbit);
                window.request_redraw();
            }
            _ => {}
        }
    }
}

struct State {
    ctx: WgpuCtx,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    rasterizer: Rasterizer,
    binner: TileBinner,
    gpu_splats: GpuSplatBuffer,
    projected: wgpu::Buffer,
    image_buf: wgpu::Buffer,
    size: PhysicalSize<u32>,
}

impl State {
    fn new(
        window: Arc<Window>,
        splats: SplatBuffer,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(window.clone())),
        );
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nano-view device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            },
        ))?;
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        configure_surface(&surface, &device, surface_format, size, caps.alpha_modes[0]);

        let ctx = WgpuCtx {
            instance,
            adapter,
            device,
            queue,
        };
        let rasterizer = Rasterizer::new(&ctx);
        let binner = TileBinner::new(&ctx);
        let gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
        let (projected, image_buf) = alloc_render_buffers(&ctx, &rasterizer, &gpu_splats, size);

        Ok(Self {
            ctx,
            surface,
            surface_format,
            rasterizer,
            binner,
            gpu_splats,
            projected,
            image_buf,
            size,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        let caps = self.surface.get_capabilities(&self.ctx.adapter);
        configure_surface(
            &self.surface,
            &self.ctx.device,
            self.surface_format,
            new_size,
            caps.alpha_modes[0],
        );
        // image_buf grows with image dimensions; projected stays fixed.
        let (_, image_buf) =
            alloc_render_buffers(&self.ctx, &self.rasterizer, &self.gpu_splats, new_size);
        self.image_buf = image_buf;
    }

    fn render(&self, orbit: &OrbitCamera) {
        let surface_tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Outdated / Lost / Timeout / Occluded / Validation — skip
            // this frame, the next resize / redraw will recover.
            _ => return,
        };
        let w = self.size.width;
        let h = self.size.height;
        let cam_pos = orbit.position();
        let camera = CameraUniform::from_pose(
            cam_pos,
            orbit.target,
            Vec3::Y,
            std::f32::consts::FRAC_PI_3, // 60° FoV
            w,
            h,
            self.gpu_splats.n,
        );

        // GPU: project + bin + composite.
        self.rasterizer
            .project(&self.ctx, &self.gpu_splats, &camera, &self.projected);
        let params = TilingParams {
            width: w,
            height: h,
            tile_size: 16,
            depth_max: (orbit.distance * 4.0).max(50.0),
        };
        let bins = self.binner.bin(&self.ctx, &self.projected, self.gpu_splats.n, &params);
        self.rasterizer.composite(
            &self.ctx,
            &self.projected,
            &bins.sorted_payloads,
            &bins.tile_ranges,
            &self.image_buf,
            &params,
        );

        // Readback → tonemap → sRGB → swapchain.
        let pixels: Vec<[f32; 4]> = self.ctx.readback(&self.image_buf, (w * h) as usize);
        let mut rgba: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        let srgb_target = self.surface_format.is_srgb();
        for p in &pixels {
            // Reinhard tonemap then optional linear-to-sRGB (the
            // swapchain handles encoding itself when surface_format is
            // an *_SRGB variant; otherwise we encode here).
            let r = p[0] / (1.0 + p[0]);
            let g = p[1] / (1.0 + p[1]);
            let b = p[2] / (1.0 + p[2]);
            let (r8, g8, b8) = if srgb_target {
                ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            } else {
                (
                    (linear_to_srgb(r) * 255.0) as u8,
                    (linear_to_srgb(g) * 255.0) as u8,
                    (linear_to_srgb(b) * 255.0) as u8,
                )
            };
            // BGRA or RGBA depending on surface_format. is_srgb covers
            // both — read components in the order the format expects.
            if matches!(
                self.surface_format,
                wgpu::TextureFormat::Bgra8Unorm
                    | wgpu::TextureFormat::Bgra8UnormSrgb
            ) {
                rgba.extend_from_slice(&[b8, g8, r8, 255]);
            } else {
                rgba.extend_from_slice(&[r8, g8, b8, 255]);
            }
        }
        self.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &surface_tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        surface_tex.present();
    }
}

fn alloc_render_buffers(
    ctx: &WgpuCtx,
    rasterizer: &Rasterizer,
    gpu_splats: &GpuSplatBuffer,
    size: PhysicalSize<u32>,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let projected = ctx.storage_buffer_zeroed(
        "view-projected",
        (gpu_splats.n as u64) * std::mem::size_of::<nano_optimize::ProjectedSplat>() as u64,
    );
    let image_buf = rasterizer.alloc_image(ctx, size.width.max(1), size.height.max(1));
    (projected, image_buf)
}

fn configure_surface(
    surface: &wgpu::Surface<'_>,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: PhysicalSize<u32>,
    alpha_mode: wgpu::CompositeAlphaMode,
) {
    surface.configure(
        device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        },
    );
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
