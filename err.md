render complete [========================================] 8/8
Timing (GPU render): buffers 0.13s, accel 0.01s, env 0.00s, pipeline 0.14s, dispatch 0.01s, readback 0.34s, total 0.63s
ref 49/50 baked
upload buffers [----------------------------------------] 0/8
Using GPU: NVIDIA GeForce RTX 3080 Ti
render complete [========================================] 8/8
Timing (GPU render): buffers 0.13s, accel 0.01s, env 0.00s, pipeline 0.18s, dispatch 0.01s, readback 0.30s, total 0.63s
ref 50/50 baked
[train] 50 references ready
[train] seeding splats via forward fit...
upload buffers [----------------------------------------] 0/8
Using GPU: NVIDIA GeForce RTX 3080 Ti
splats complete [========================================] 8/8
Timing (GPU splats): buffers 0.14s, accel 0.01s, env 0.00s, pipeline 0.29s, dispatch 30.72s, readback 50.59s, total 81.75s
[train] 6685364 seed splats
[train] initialising wgpu rasteriser...
[nano-optimize] wgpu adapter: NVIDIA GeForce RTX 3080 Ti (Vulkan, DiscreteGpu)
[train] forward rasterising 30000 iterations...

thread 'main' (32752) panicked at C:\Programs\Ntutil\apps\prog\lang\Rust\cargo\registry\src\index.crates.io-1949cf8c6b5b557f\wgpu-29.0.3\src\backend\wgpu_core.r
s:2653:18:
wgpu error: Validation Error

Caused by:
  In a CommandEncoder, label = 'project-encoder'
    In a dispatch command, indirect:false
      Each current dispatch group size dimension ([104459, 1, 1]) must be less or equal to 65535


stack backtrace:
   0: std::panicking::panic_handler
             at /rustc/59807616e1fa2540724bfbac14d7976d7e4a3860/library\std\src\panicking.rs:689
   1: core::panicking::panic_fmt
             at /rustc/59807616e1fa2540724bfbac14d7976d7e4a3860/library\core\src\panicking.rs:80
   2: wgpu::backend::wgpu_core::ContextWgpuCore::format_error
   3: wgpu::backend::wgpu_core::ContextWgpuCore::handle_error_inner
   4: <wgpu::backend::wgpu_core::CoreCommandEncoder as wgpu::dispatch::CommandEncoderInterface>::finish
   5: wgpu::api::command_encoder::CommandEncoder::finish
   6: nano_optimize::raster::Rasterizer::project
   7: nano_optimize::train::train
   8: nanotracer_rs::main
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.