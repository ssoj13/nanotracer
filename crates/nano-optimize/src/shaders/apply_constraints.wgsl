//! Per-splat physical-state clamps applied after the Adam update:
//!   * quaternion re-normalisation (rotation must stay on the unit
//!     hyper-sphere; Adam happily drifts off otherwise),
//!   * log-σ clamp `[log_scale_min, log_scale_max]` (prevents runaway
//!     exponentials in the scale -> σ map),
//!   * opacity-logit clamp `[-opacity_logit_max, opacity_logit_max]`.

struct ConstraintParams {
    n:                 u32,
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,
    log_scale_min:     f32,
    log_scale_max:     f32,
    opacity_logit_max: f32,
    _pad3:             f32,
};

@group(0) @binding(0) var<storage, read_write> rotations: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> scales:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> opacities: array<f32>;
@group(0) @binding(3) var<uniform>             cfg:       ConstraintParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }

    let q = rotations[i];
    let n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    let nrm = max(sqrt(n2), 1e-8);
    rotations[i] = q / nrm;

    var s = scales[i];
    s.x = clamp(s.x, cfg.log_scale_min, cfg.log_scale_max);
    s.y = clamp(s.y, cfg.log_scale_min, cfg.log_scale_max);
    s.z = clamp(s.z, cfg.log_scale_min, cfg.log_scale_max);
    scales[i] = s;

    opacities[i] = clamp(opacities[i], -cfg.opacity_logit_max, cfg.opacity_logit_max);
}
