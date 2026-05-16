//! Add per-block offsets to the per-element scan results.
//!
//! `offsets[wid]` carries the exclusive prefix sum of block totals (i.e.
//! the sum of all blocks strictly before `wid`). Adding it to every
//! element in block `wid` completes the global scan.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<storage, read_write> data:    array<u32>;
@group(0) @binding(1) var<storage, read>       offsets: array<u32>;
@group(0) @binding(2) var<uniform>             params:  Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    data[i] = data[i] + offsets[wid.x];
}
