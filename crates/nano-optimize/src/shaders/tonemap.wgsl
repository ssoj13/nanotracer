//! Convert the `vec4<f32>` composite framebuffer into an
//! `rgba8unorm` storage texture suitable for egui to sample (or any
//! standard surface presentation path).
//!
//! Per pixel: Reinhard `c / (1 + c)` tonemap, linear → sRGB, clamp,
//! quantise to `u8`. The `w` channel (forward "coverage" = `1 − T`)
//! becomes alpha so transparent splat regions read as transparent in
//! egui's image widget.

struct Params {
    width:  u32,
    height: u32,
};

@group(0) @binding(0) var<storage, read>            source:  array<vec4<f32>>;
@group(0) @binding(1) var                            output:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform>                  params:  Params;

fn linear_to_srgb(v: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
    let cutoff = vec3<f32>(0.0031308);
    let low = clamped * 12.92;
    let high = 1.055 * pow(clamped, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(low, high, clamped > cutoff);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let raw = source[idx];
    let tonemapped = raw.xyz / (vec3<f32>(1.0) + max(raw.xyz, vec3<f32>(0.0)));
    let srgb = linear_to_srgb(tonemapped);
    textureStore(
        output,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(srgb, raw.w),
    );
}
