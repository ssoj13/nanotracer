//! Interactive splat viewer — winit window + wgpu surface + egui_dock UI.
//!
//! Architecture per frame:
//!
//!   1. winit events → orbit camera state + egui_winit input integration.
//!   2. GPU compute pipeline: project → bin → composite (vec4f buffer)
//!      → tonemap (rgba8unorm storage texture).
//!   3. egui pass: dock layout with a `Viewport` tab (sampling the
//!      tonemapped texture via `egui::Image`) and an `Inspector` tab
//!      (HUD: FPS, splat count, camera pose, FoV slider).
//!   4. `egui_wgpu::Renderer` renders the egui draw lists into the
//!      swapchain texture; surface presents.
//!
//! No CPU readback in the present path — the splat image stays on the
//! GPU all the way to the swapchain. The previous CPU-readback fallback
//! has been retired.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use glam::Vec3;
use nano_core::scene::Scene;
use nano_optimize::raster::CameraUniform;
use nano_optimize::splat_gpu::GpuSplatBuffer;
use nano_optimize::{Rasterizer, SplatBuffer, TileBinner, TilingParams, TrainConfig, WgpuCtx};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Open the viewer window on `splats`. Blocks until the user closes
/// the window.
pub fn run(splats: SplatBuffer) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ViewerApp::new(splats);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Per-iteration snapshot published by the training thread.
#[derive(Clone, Default)]
pub struct TrainStats {
    pub iter: u32,
    pub mse: f32,
    pub splat_count: u32,
}

struct TrainSnapshot {
    splats: SplatBuffer,
    stats: TrainStats,
    /// Monotonic version number; viewer reads it cheaply to decide
    /// whether to re-upload.
    version: u64,
}

type SharedSnapshot = Arc<RwLock<Option<TrainSnapshot>>>;

/// Spawn a background training thread on `scene` + `cfg`, then open
/// the viewer with a live `Training` dock tab showing the loss curve
/// and per-iteration stats. The viewport stays in sync with the
/// training thread by polling a shared `Arc<RwLock<...>>` snapshot
/// each frame and re-uploading whenever a new version is published.
///
/// Blocks until the user closes the window. The training thread
/// runs to completion (or until the window closes, whichever first);
/// when the window closes the worker is detached and finishes on
/// its own.
pub fn run_with_training(
    scene: Scene,
    cfg: TrainConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let shared: SharedSnapshot = Arc::new(RwLock::new(None));
    let shared_w = shared.clone();
    let paused = Arc::new(AtomicBool::new(false));
    let paused_w = paused.clone();

    // Worker: run the full training loop, publishing a snapshot of
    // SplatBuffer + per-iteration stats after every Adam step.
    // Between iterations the worker spin-sleeps if the viewer flips
    // the `paused` flag — keeps the GPU idle while the user
    // inspects the current splat field.
    thread::spawn(move || {
        let mut version: u64 = 0;
        let result = nano_optimize::train(&scene, &cfg, |iter, splats, mse| {
            version = version.wrapping_add(1);
            let snapshot = TrainSnapshot {
                splats: splats.clone(),
                stats: TrainStats {
                    iter,
                    mse,
                    splat_count: splats.len() as u32,
                },
                version,
            };
            *shared_w.write().unwrap() = Some(snapshot);
            while paused_w.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
        });
        if let Err(e) = result {
            eprintln!("[train thread] error: {e}");
        }
    });

    // Block briefly for the first snapshot — the worker has to bake
    // reference views + forward-fit before the first Adam step, which
    // takes a few seconds on a typical scene. Spinning here keeps the
    // ApplicationHandler shape simple (it always sees splats).
    let initial_splats = loop {
        if let Some(snap) = shared.read().unwrap().as_ref() {
            break snap.splats.clone();
        }
        thread::sleep(std::time::Duration::from_millis(100));
    };

    let event_loop = EventLoop::new()?;
    let mut app = ViewerApp::new(initial_splats);
    app.training = Some(TrainingChannel {
        shared,
        history: Vec::new(),
        last_seen_version: 0,
        paused,
    });
    // Augment the dock layout with a Training tab.
    let surface = app.dock.main_surface_mut();
    surface.split_below(NodeIndex::root().right(), 0.5, vec![Tab::Training]);
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Holds the viewer-side state for a live training session.
struct TrainingChannel {
    shared: SharedSnapshot,
    history: Vec<f32>,
    last_seen_version: u64,
    /// Worker checks this between iterations and sleeps while set.
    paused: Arc<AtomicBool>,
}

/// Orbit camera around the scene centroid. LMB drag rotates,
/// scroll-wheel zooms; the dock UI exposes a FoV slider.
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
    fov_y: f32,
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
struct InputState {
    lmb: bool,
    rmb: bool,
    last_x: f64,
    last_y: f64,
    /// WASD + Q/E for fly mode (Q = down, E = up).
    w: bool, a: bool, s: bool, d: bool, q: bool, e: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Viewport,
    Inspector,
    Training,
}

/// Per-frame UI state shared between the dock tabs.
struct UiState {
    splat_texture_id: egui::TextureId,
    image_size: [u32; 2],
    splat_count: u32,
    fps: f32,
    fov_y_deg: f32,
    azimuth: f32,
    elevation: f32,
    distance: f32,
    target: Vec3,
    /// Set by the Viewport tab each frame so the host knows the size
    /// of the area we should render into for the next frame.
    requested_viewport_size: [u32; 2],
    /// Save-to-PLY UI state. `path` is the text-edit buffer; setting
    /// `requested` to true on a button click queues a write that the
    /// host fulfils after the egui pass closes.
    save_path: String,
    save_requested: bool,
    save_last_result: Option<Result<usize, String>>,
    /// Live-training pause toggle. Mirrored from `TrainingChannel`
    /// before the egui pass and pushed back after.
    training_paused: bool,
    training_attached: bool,
}

struct ViewerApp {
    window: Option<Arc<Window>>,
    state: Option<State>,
    splats: Option<SplatBuffer>,
    orbit: OrbitCamera,
    input: InputState,
    dock: DockState<Tab>,
    training: Option<TrainingChannel>,
    save_path: String,
    save_last_result: Option<Result<usize, String>>,
}

impl ViewerApp {
    fn new(splats: SplatBuffer) -> Self {
        let (centroid, extent) = scene_centroid_extent(&splats);
        let distance = (extent.max_element() * 1.5).max(2.0);
        let mut dock = DockState::new(vec![Tab::Viewport]);
        let surface = dock.main_surface_mut();
        surface.split_right(NodeIndex::root(), 0.78, vec![Tab::Inspector]);
        Self {
            splats: Some(splats),
            window: None,
            state: None,
            orbit: OrbitCamera {
                target: centroid,
                distance,
                azimuth: 0.0,
                elevation: 0.3,
                fov_y: std::f32::consts::FRAC_PI_3,
            },
            input: InputState::default(),
            dock,
            training: None,
            save_path: String::from("viewer_export.ply"),
            save_last_result: None,
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
            .with_inner_size(PhysicalSize::new(1280u32, 800u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());
        let splats = self.splats.take().expect("splats consumed once");
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
        // Let egui consume input first; if egui claims it, skip the
        // viewer's own handlers (e.g. typing in a numeric box).
        let response = state.egui_state.on_window_event(window, &event);
        let captured_pointer = state.egui_ctx.is_pointer_over_egui();

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
            } if !response.consumed => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } if !response.consumed => {
                let pressed = matches!(key_state, ElementState::Pressed);
                match code {
                    KeyCode::KeyW => self.input.w = pressed,
                    KeyCode::KeyA => self.input.a = pressed,
                    KeyCode::KeyS => self.input.s = pressed,
                    KeyCode::KeyD => self.input.d = pressed,
                    KeyCode::KeyQ => self.input.q = pressed,
                    KeyCode::KeyE => self.input.e = pressed,
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::MouseInput {
                state: btn_state,
                button: MouseButton::Left,
                ..
            } if !captured_pointer => {
                self.input.lmb = matches!(btn_state, ElementState::Pressed);
                if !self.input.lmb {
                    self.input.last_x = 0.0;
                    self.input.last_y = 0.0;
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button: MouseButton::Right,
                ..
            } if !captured_pointer => {
                self.input.rmb = matches!(btn_state, ElementState::Pressed);
                if !self.input.rmb && !self.input.lmb {
                    self.input.last_x = 0.0;
                    self.input.last_y = 0.0;
                }
            }
            WindowEvent::CursorMoved { position, .. }
                if !captured_pointer && (self.input.lmb || self.input.rmb) =>
            {
                // Skip the delta-apply only on the very first cursor
                // event after a press (when last_{x,y} are still the
                // sentinel 0.0) so we don't lurch toward the window
                // origin. Last_x/last_y are still updated below so
                // subsequent moves work.
                let has_prev = self.input.last_x != 0.0 || self.input.last_y != 0.0;
                let dx = (position.x - self.input.last_x) as f32;
                let dy = (position.y - self.input.last_y) as f32;
                if has_prev && self.input.lmb {
                    // LMB drag — orbit.
                    self.orbit.azimuth += dx * 0.005;
                    self.orbit.elevation =
                        (self.orbit.elevation - dy * 0.005).clamp(-1.5, 1.5);
                } else if has_prev && self.input.rmb {
                    // RMB drag — pan target in the camera's screen
                    // plane. Pan magnitude scales with distance so
                    // the feel is consistent at any zoom level.
                    let cam_pos = self.orbit.position();
                    let forward = (self.orbit.target - cam_pos).normalize_or_zero();
                    let world_up = Vec3::Y;
                    let right = forward.cross(world_up).normalize_or_zero();
                    let up = right.cross(forward).normalize_or_zero();
                    let scale = self.orbit.distance * 0.002;
                    self.orbit.target -= right * (dx * scale) - up * (dy * scale);
                }
                self.input.last_x = position.x;
                self.input.last_y = position.y;
            }
            WindowEvent::MouseWheel { delta, .. } if !captured_pointer => {
                let s = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.y as f32) * 0.05,
                };
                self.orbit.distance = (self.orbit.distance * (1.0 - s * 0.1)).clamp(0.5, 500.0);
            }
            WindowEvent::RedrawRequested => {
                // Apply per-frame WASD / QE translation in the camera
                // frame. Speed is proportional to current distance —
                // moving a far-away cloud at 1 m/frame would feel
                // glacial; moving a 1 m cloud at 100 m/frame would
                // be unusable. Roughly 1% of distance per pressed
                // key per frame ≈ a sensible default at 60 fps.
                let step = self.orbit.distance * 0.01;
                let cam_pos = self.orbit.position();
                let forward = (self.orbit.target - cam_pos).normalize_or_zero();
                let right = forward.cross(Vec3::Y).normalize_or_zero();
                let up = right.cross(forward).normalize_or_zero();
                if self.input.w { self.orbit.target += forward * step; }
                if self.input.s { self.orbit.target -= forward * step; }
                if self.input.d { self.orbit.target += right   * step; }
                if self.input.a { self.orbit.target -= right   * step; }
                if self.input.e { self.orbit.target += up      * step; }
                if self.input.q { self.orbit.target -= up      * step; }
                let window_clone = window.clone();
                self.frame(&window_clone);
                window_clone.request_redraw();
            }
            _ => {}
        }
    }
}

impl ViewerApp {
    fn frame(&mut self, window: &Arc<Window>) {
        // Pull the latest training snapshot, if any.
        let live_stats = if let Some(ch) = self.training.as_mut() {
            let snap_ready = {
                let guard = ch.shared.read().unwrap();
                guard
                    .as_ref()
                    .map(|s| (s.version, s.splats.clone(), s.stats.clone()))
            };
            if let Some((v, splats, stats)) = snap_ready {
                if v > ch.last_seen_version {
                    ch.last_seen_version = v;
                    ch.history.push(stats.mse);
                    if let Some(state) = self.state.as_mut() {
                        state.update_splats(splats);
                    }
                }
                Some((stats, ch.history.clone()))
            } else {
                None
            }
        } else {
            None
        };
        // Snapshot the pause flag for the UI; the frame writes any
        // toggle back to the AtomicBool below.
        let training_paused_before = self
            .training
            .as_ref()
            .map(|ch| ch.paused.load(Ordering::Relaxed))
            .unwrap_or(false);
        let training_attached = self.training.is_some();
        let Some(state) = self.state.as_mut() else { return; };
        let frame_in = FrameInputs {
            live_stats,
            training_paused: training_paused_before,
            training_attached,
            save_path: self.save_path.clone(),
            save_last_result: self.save_last_result.take(),
        };
        let frame_out = state.frame(window, &mut self.orbit, &mut self.dock, frame_in);
        // Push UI mutations back into the app state.
        self.save_path = frame_out.save_path;
        if let Some(ch) = self.training.as_mut()
            && frame_out.training_paused != training_paused_before
        {
            ch.paused.store(frame_out.training_paused, Ordering::Relaxed);
        }
        // Save was requested — write the latest CPU SplatBuffer to PLY.
        // Synchronous; ~ms per million splats, fine on the UI thread.
        if frame_out.save_requested {
            let path = std::path::PathBuf::from(&self.save_path);
            let result = if let Some(state) = self.state.as_ref() {
                let gaussians = state.last_cpu_splats.to_gaussians();
                let n = gaussians.len();
                nano_splat::write_ply(&path, &gaussians)
                    .map(|()| n)
                    .map_err(|e| format!("{e}"))
            } else {
                Err("viewer state not ready".to_string())
            };
            self.save_last_result = Some(result);
        }
    }
}

struct FrameInputs {
    live_stats: Option<(TrainStats, Vec<f32>)>,
    training_paused: bool,
    training_attached: bool,
    save_path: String,
    save_last_result: Option<Result<usize, String>>,
}

struct FrameOutputs {
    save_path: String,
    save_requested: bool,
    training_paused: bool,
}

struct State {
    ctx: WgpuCtx,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    size: PhysicalSize<u32>,

    rasterizer: Rasterizer,
    binner: TileBinner,
    gpu_splats: GpuSplatBuffer,
    /// Latest CPU SplatBuffer — kept so the "Save PLY" button doesn't
    /// have to read back from GPU. Synced from the initial load and
    /// from every live-training snapshot.
    last_cpu_splats: SplatBuffer,

    // Per-frame GPU resources sized to the viewport tab.
    image_w: u32,
    image_h: u32,
    projected: wgpu::Buffer,
    composite_buf: wgpu::Buffer,
    splat_tex: wgpu::Texture,
    splat_view: wgpu::TextureView,

    // egui glue.
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    splat_texture_id: egui::TextureId,

    // Frame stats.
    frame_times: VecDeque<Instant>,
}

const VIEWPORT_TEX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEFAULT_VIEWPORT: (u32, u32) = (960, 720);

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
        // egui-wgpu prefers an sRGB swapchain so its UI colours are
        // gamma-correct out of the box.
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        configure_surface(&surface, &device, surface_format, size, caps.alpha_modes[0]);

        let ctx = WgpuCtx { instance, adapter, device, queue };
        let rasterizer = Rasterizer::new(&ctx);
        let binner = TileBinner::new(&ctx);
        let gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
        let last_cpu_splats = splats;

        let (image_w, image_h) = DEFAULT_VIEWPORT;
        let projected = ctx.storage_buffer_zeroed(
            "view-projected",
            (gpu_splats.n as u64)
                * std::mem::size_of::<nano_optimize::ProjectedSplat>() as u64,
        );
        let composite_buf = rasterizer.alloc_image(&ctx, image_w, image_h);
        let (splat_tex, splat_view) = make_splat_texture(&ctx.device, image_w, image_h);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(
            &ctx.device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );
        let splat_texture_id = egui_renderer.register_native_texture(
            &ctx.device,
            &splat_view,
            wgpu::FilterMode::Linear,
        );

        Ok(Self {
            ctx,
            surface,
            surface_format,
            size,
            rasterizer,
            binner,
            gpu_splats,
            last_cpu_splats,
            image_w,
            image_h,
            projected,
            composite_buf,
            splat_tex,
            splat_view,
            egui_ctx,
            egui_state,
            egui_renderer,
            splat_texture_id,
            frame_times: VecDeque::with_capacity(120),
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
    }

    /// Reallocate the render targets to match the viewport tab's size
    /// (called when the tab is resized by dragging the dock split).
    /// Swap the GPU splat buffer for a freshly trained CPU buffer.
    /// If the count changed (densify / prune), we have to reallocate
    /// instead of `sync_from`. Always stash a copy on the CPU side so
    /// the "Save PLY" button has zero-readback access.
    fn update_splats(&mut self, splats: SplatBuffer) {
        if splats.len() as u32 == self.gpu_splats.n {
            self.gpu_splats.sync_from(&self.ctx, &splats);
        } else {
            self.gpu_splats = GpuSplatBuffer::upload(&self.ctx, &splats);
            self.projected = self.ctx.storage_buffer_zeroed(
                "view-projected",
                (self.gpu_splats.n as u64)
                    * std::mem::size_of::<nano_optimize::ProjectedSplat>() as u64,
            );
        }
        self.last_cpu_splats = splats;
    }

    fn resize_viewport(&mut self, new_w: u32, new_h: u32) {
        if (new_w, new_h) == (self.image_w, self.image_h) {
            return;
        }
        self.image_w = new_w.max(1);
        self.image_h = new_h.max(1);
        self.composite_buf = self.rasterizer.alloc_image(&self.ctx, self.image_w, self.image_h);
        let (tex, view) = make_splat_texture(&self.ctx.device, self.image_w, self.image_h);
        self.splat_tex = tex;
        self.splat_view = view;
        // egui caches texture bindings — update the registered slot.
        self.egui_renderer
            .update_egui_texture_from_wgpu_texture(
                &self.ctx.device,
                &self.splat_view,
                wgpu::FilterMode::Linear,
                self.splat_texture_id,
            );
    }

    fn frame(
        &mut self,
        window: &Arc<Window>,
        orbit: &mut OrbitCamera,
        dock: &mut DockState<Tab>,
        frame_in: FrameInputs,
    ) -> FrameOutputs {
        // ── Compute pass: project + bin + composite + tonemap ─────────
        let cam = CameraUniform::from_pose(
            orbit.position(),
            orbit.target,
            Vec3::Y,
            orbit.fov_y,
            self.image_w,
            self.image_h,
            self.gpu_splats.n,
        );
        self.rasterizer
            .project(&self.ctx, &self.gpu_splats, &cam, &self.projected);
        let params = TilingParams {
            width: self.image_w,
            height: self.image_h,
            tile_size: 16,
            depth_max: (orbit.distance * 4.0).max(50.0),
        };
        let bins = self
            .binner
            .bin(&self.ctx, &self.projected, self.gpu_splats.n, &params);
        self.rasterizer.composite(
            &self.ctx,
            &self.projected,
            &bins.sorted_payloads,
            &bins.tile_ranges,
            &self.composite_buf,
            &params,
        );
        self.rasterizer.tonemap(
            &self.ctx,
            &self.composite_buf,
            &self.splat_view,
            self.image_w,
            self.image_h,
        );

        // ── Frame stats ───────────────────────────────────────────────
        let now = Instant::now();
        self.frame_times.push_back(now);
        while let Some(t) = self.frame_times.front() {
            if now.duration_since(*t).as_secs_f32() > 1.0 {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }
        let fps = self.frame_times.len() as f32;

        // ── egui pass ─────────────────────────────────────────────────
        let raw_input = self.egui_state.take_egui_input(window);
        let mut ui_state = UiState {
            splat_texture_id: self.splat_texture_id,
            image_size: [self.image_w, self.image_h],
            splat_count: self.gpu_splats.n,
            fps,
            fov_y_deg: orbit.fov_y.to_degrees(),
            azimuth: orbit.azimuth,
            elevation: orbit.elevation,
            distance: orbit.distance,
            target: orbit.target,
            requested_viewport_size: [self.image_w, self.image_h],
            save_path: frame_in.save_path,
            save_requested: false,
            save_last_result: frame_in.save_last_result,
            training_paused: frame_in.training_paused,
            training_attached: frame_in.training_attached,
        };
        let live_stats = frame_in.live_stats;
        // egui 0.34 prefers `run_ui` (yields a `&mut Ui` directly) over
        // `run` + a CentralPanel wrapper. We grab the dock style once
        // outside the closure since `global_style()` returns an Arc
        // snapshot — cheap to clone, no need for the per-tick Context
        // borrow.
        let dock_style = Style::from_egui(self.egui_ctx.global_style().as_ref());
        let output = self.egui_ctx.clone().run_ui(raw_input, |ui| {
            let mut viewer = DockedTabs {
                state: &mut ui_state,
                train: live_stats.as_ref(),
            };
            DockArea::new(dock)
                .style(dock_style.clone())
                .show_inside(ui, &mut viewer);
        });
        // Push any UI-side mutations back out (FoV slider).
        orbit.fov_y = ui_state.fov_y_deg.to_radians().clamp(
            std::f32::consts::FRAC_PI_8,
            std::f32::consts::PI - 0.1,
        );
        let req = ui_state.requested_viewport_size;
        self.resize_viewport(req[0], req[1]);

        self.egui_state
            .handle_platform_output(window, output.platform_output);
        let paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: output.pixels_per_point,
        };

        // Build the outputs that the host caller pushes back into
        // ViewerApp regardless of whether the surface present succeeds.
        let frame_out = FrameOutputs {
            save_path: ui_state.save_path.clone(),
            save_requested: ui_state.save_requested,
            training_paused: ui_state.training_paused,
        };

        // ── Surface render ────────────────────────────────────────────
        let surface_tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return frame_out,
        };
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui-encoder"),
                });
        for (id, delta) in &output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut rp = render_pass.forget_lifetime();
            self.egui_renderer
                .render(&mut rp, &paint_jobs, &screen_desc);
        }
        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.ctx
            .queue
            .submit(std::iter::once(encoder.finish()));
        surface_tex.present();
        frame_out
    }
}

/// egui_dock `TabViewer` impl — drawing logic for each tab kind.
struct DockedTabs<'a> {
    state: &'a mut UiState,
    train: Option<&'a (TrainStats, Vec<f32>)>,
}

impl TabViewer for DockedTabs<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "Viewport".into(),
            Tab::Inspector => "Inspector".into(),
            Tab::Training => "Training".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewport => {
                let available = ui.available_size_before_wrap();
                // Round to multiples of the tile size — keeps the
                // rasteriser's workgroups happy and avoids reallocating
                // every single mouse pixel.
                let w = (available.x.max(64.0) as u32).max(64).next_multiple_of(16);
                let h = (available.y.max(64.0) as u32).max(64).next_multiple_of(16);
                self.state.requested_viewport_size = [w, h];
                let aspect = w as f32 / h as f32;
                let img_h = available.y;
                let img_w = img_h * aspect;
                ui.add(
                    egui::Image::from_texture((
                        self.state.splat_texture_id,
                        egui::vec2(img_w, img_h),
                    ))
                    .fit_to_exact_size(egui::vec2(img_w, img_h)),
                );
            }
            Tab::Inspector => {
                ui.heading("Stats");
                ui.label(format!("Splats: {}", self.state.splat_count));
                ui.label(format!(
                    "Viewport: {}×{}",
                    self.state.image_size[0], self.state.image_size[1]
                ));
                ui.label(format!("FPS: {:.1}", self.state.fps));
                ui.separator();
                ui.heading("Camera");
                ui.add(
                    egui::Slider::new(&mut self.state.fov_y_deg, 20.0..=120.0)
                        .text("FoV (vertical, °)"),
                );
                ui.label(format!("Azimuth: {:.2}", self.state.azimuth));
                ui.label(format!("Elevation: {:.2}", self.state.elevation));
                ui.label(format!("Distance: {:.2}", self.state.distance));
                ui.label(format!(
                    "Target: ({:.2}, {:.2}, {:.2})",
                    self.state.target.x, self.state.target.y, self.state.target.z
                ));
                ui.separator();
                ui.heading("Export");
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    ui.text_edit_singleline(&mut self.state.save_path);
                });
                if ui.button("💾 Save PLY").clicked() {
                    self.state.save_requested = true;
                }
                if let Some(result) = &self.state.save_last_result {
                    match result {
                        Ok(n) => {
                            ui.colored_label(
                                egui::Color32::LIGHT_GREEN,
                                format!("✓ wrote {n} splats"),
                            );
                        }
                        Err(e) => {
                            ui.colored_label(
                                egui::Color32::LIGHT_RED,
                                format!("✗ {e}"),
                            );
                        }
                    }
                }
                ui.separator();
                ui.small("LMB orbit · RMB pan · WASD / QE fly · scroll zoom · ESC quit");
            }
            Tab::Training => {
                ui.heading("Training");
                if self.state.training_attached {
                    let mut paused = self.state.training_paused;
                    if ui.checkbox(&mut paused, "⏸ Pause training").changed() {
                        self.state.training_paused = paused;
                    }
                    ui.separator();
                }
                match self.train {
                    None => {
                        ui.label("No live training session attached.");
                        ui.small("Run with `--view-training` to populate this tab.");
                    }
                    Some((stats, history)) => {
                        ui.label(format!("Iter: {}", stats.iter));
                        ui.label(format!("MSE:  {:.6}", stats.mse));
                        ui.label(format!("Splats: {}", stats.splat_count));
                        ui.separator();
                        egui_plot::Plot::new("loss_curve")
                            .view_aspect(2.0)
                            .show_axes([true, true])
                            .allow_drag(true)
                            .allow_zoom(true)
                            .show(ui, |plot_ui| {
                                let pts: egui_plot::PlotPoints = history
                                    .iter()
                                    .enumerate()
                                    .map(|(i, m)| [i as f64, *m as f64])
                                    .collect();
                                plot_ui.line(egui_plot::Line::new("MSE", pts));
                            });
                    }
                }
            }
        }
    }
}

fn make_splat_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("splat-tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: VIEWPORT_TEX_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
