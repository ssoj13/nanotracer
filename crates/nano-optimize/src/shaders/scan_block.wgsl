//! Single-block exclusive prefix-scan (Hillis-Steele).
//!
//! Each workgroup of 256 threads handles 256 consecutive u32s in
//! `data[]`. After this kernel runs:
//!   • `data[i]` holds the exclusive prefix sum within its block
//!   • `block_sums[wid]` holds the inclusive total of that block
//!
//! Multi-block orchestration: the host runs `scan_block` recursively
//! on `block_sums`, then adds the resulting per-block offsets back
//! into `data` with `scan_add_offsets.wgsl`.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<storage, read_write> data:        array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums:  array<u32>;
@group(0) @binding(2) var<uniform>              params:     Params;

var<workgroup> shared_data: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let tid = lid.x;
    let i = gid.x;
    let local_val = select(0u, data[i], i < params.n);

    shared_data[tid] = local_val;
    workgroupBarrier();

    // Hillis-Steele inclusive scan within the block (log₂(256) = 8 passes).
    var offset: u32 = 1u;
    for (var step: u32 = 0u; step < 8u; step = step + 1u) {
        var v: u32 = 0u;
        if (tid >= offset) {
            v = shared_data[tid - offset];
        }
        workgroupBarrier();
        shared_data[tid] = shared_data[tid] + v;
        workgroupBarrier();
        offset = offset * 2u;
    }

    let inclusive = shared_data[tid];
    let exclusive = inclusive - local_val;
    if (i < params.n) {
        data[i] = exclusive;
    }

    // Last active thread of the block writes the block's inclusive sum.
    let block_size: u32 = 256u;
    let block_end = (wid.x + 1u) * block_size;
    let last_active_tid = select(block_size - 1u, params.n - wid.x * block_size - 1u, block_end > params.n);
    if (tid == last_active_tid) {
        block_sums[wid.x] = inclusive;
    }
}
