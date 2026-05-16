//! Compute the total number of "bit is 0" elements after the predicate
//! scan. The scan produces `scan[n-1] = count of 0-bits in [0, n-2]`;
//! adding the predicate value at index `n-1` gives the true total.
//!
//! Single-thread kernel — writes one `u32` to `total[0]`.

struct Params {
    n:     u32,
    bit:   u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>          keys:   array<u32>;
@group(0) @binding(1) var<storage, read>          scan:   array<u32>;
@group(0) @binding(2) var<storage, read_write>    total:  array<u32>;
@group(0) @binding(3) var<uniform>                params: Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
    if (params.n == 0u) {
        total[0] = 0u;
        return;
    }
    let last = params.n - 1u;
    let last_bit = (keys[last] >> params.bit) & 1u;
    total[0] = scan[last] + (1u - last_bit);
}
