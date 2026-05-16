//! Accumulate per-splat L1(d_position) into a running scalar — feeds
//! the densify gate (Inria-style "high-gradient splats either CLONE or
//! SPLIT"). Reads the vec4-padded position-gradient buffer and ignores
//! the W lane (padding).

struct AccParams {
    n:     u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       d_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> grad_acc:    array<f32>;
@group(0) @binding(2) var<uniform>             cfg:         AccParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let dp = d_positions[i];
    grad_acc[i] = grad_acc[i] + abs(dp.x) + abs(dp.y) + abs(dp.z);
}
