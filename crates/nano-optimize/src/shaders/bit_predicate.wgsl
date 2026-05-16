//! Predicate kernel for the 1-bit-at-a-time stable radix sort.
//!
//! Writes `predicate[i] = 1` if bit `params.bit` of `keys[i]` is zero,
//! else `0`. A subsequent exclusive prefix-scan over `predicate` tells
//! each element where it lands in the "0-bits first" partition.

struct Params {
    n:     u32,
    bit:   u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>          keys:      array<u32>;
@group(0) @binding(1) var<storage, read_write>    predicate: array<u32>;
@group(0) @binding(2) var<uniform>                params:    Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let bit = (keys[i] >> params.bit) & 1u;
    predicate[i] = 1u - bit;
}
