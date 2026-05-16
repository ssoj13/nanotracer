//! Build per-tile `[begin, end)` ranges over the sorted (tile-id, depth) key stream.
//!
//! `tile_ranges` is laid out as two `u32` per tile: `[begin, end]`. The
//! caller zero-initialises it before this dispatch — tiles with no
//! splats stay `(0, 0)`, which the rasteriser interprets as empty.
//!
//! Each thread inspects neighbour keys to detect a tile-id transition
//! (and the first/last entries of the whole stream). Tile-id occupies
//! the high 16 bits of each key.

struct Params {
    total_pairs: u32,
    num_tiles:   u32,
    _pad0:       u32,
    _pad1:       u32,
};

@group(0) @binding(0) var<storage, read>          sorted_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write>    tile_ranges: array<u32>;
@group(0) @binding(2) var<uniform>                params:      Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.total_pairs) { return; }

    let tile = sorted_keys[i] >> 16u;
    let starts_here = (i == 0u) || ((sorted_keys[i - 1u] >> 16u) != tile);
    let ends_here   = (i + 1u == params.total_pairs)
                   || ((sorted_keys[i + 1u] >> 16u) != tile);

    if (tile >= params.num_tiles) {
        return;
    }
    if (starts_here) {
        tile_ranges[tile * 2u + 0u] = i;
    }
    if (ends_here) {
        tile_ranges[tile * 2u + 1u] = i + 1u;
    }
}
