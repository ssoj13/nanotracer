//! Stable scatter for one bit of the radix sort.
//!
//! `scan[i]` is the exclusive prefix-count of "bit-zero" predicates
//! (i.e. how many elements with bit==0 appear at positions < i).
//! `total[0]` is the grand total of "bit-zero" elements. Then:
//!
//!   • bit==0 elements pack into `[0, total)` keeping input order
//!   • bit==1 elements pack into `[total, n)` keeping input order
//!
//! Per-element destination is computed in O(1). Each `dst` is unique
//! across threads — no atomics needed.

struct Params {
    n:     u32,
    bit:   u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>          keys_src: array<u32>;
@group(0) @binding(1) var<storage, read>          vals_src: array<u32>;
@group(0) @binding(2) var<storage, read_write>    keys_dst: array<u32>;
@group(0) @binding(3) var<storage, read_write>    vals_dst: array<u32>;
@group(0) @binding(4) var<storage, read>          scan:     array<u32>;
@group(0) @binding(5) var<storage, read>          total:    array<u32>;
@group(0) @binding(6) var<uniform>                params:   Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let k = keys_src[i];
    let v = vals_src[i];
    let bit = (k >> params.bit) & 1u;
    let zeros_before = scan[i];
    let dst = select(
        total[0] + (i - zeros_before),  // bit == 1: pack after all zeros
        zeros_before,                    // bit == 0: pack at zeros prefix
        bit == 0u,
    );
    keys_dst[dst] = k;
    vals_dst[dst] = v;
}
