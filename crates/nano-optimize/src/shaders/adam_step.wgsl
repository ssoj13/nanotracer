//! Generic Adam optimiser kernel — operates on a flat f32 lane view of
//! whatever attribute buffer (positions, scales, …) the caller binds.
//! Padded vec4 lanes (sh_rest's 3 trailing pads, vec4 W slot) carry
//! zero gradient so their moments stay at zero and the param update is
//! `0`, which is correct for "this lane doesn't exist".

struct AdamParams {
    lr:     f32,
    beta1:  f32,
    beta2:  f32,
    eps:    f32,
    bc1:    f32,
    bc2:    f32,
    n:      u32,
    _pad:   u32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read>       grads:  array<f32>;
@group(0) @binding(2) var<storage, read_write> m:      array<f32>;
@group(0) @binding(3) var<storage, read_write> v:      array<f32>;
@group(0) @binding(4) var<uniform>             cfg:    AdamParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    let g = grads[i];
    let m_new = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g;
    let v_new = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;
    let m_hat = m_new / cfg.bc1;
    let v_hat = v_new / cfg.bc2;
    params[i] = params[i] - cfg.lr * m_hat / (sqrt(v_hat) + cfg.eps);
}
