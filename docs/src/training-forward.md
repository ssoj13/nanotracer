# Forward rasteriser

The forward pass turns a CPU-resident `SplatBuffer` into a
`vec4<f32>` framebuffer. Three logical stages: project, bin, composite.

## Stage 1 — `project_gaussians.wgsl`

One thread per splat. Output `ProjectedSplat` (48 B, three `vec4`):

```text
mean_depth_radius (vec4)  // .xy mean pixel, .z depth (positive forward), .w 3σ radius px
conic_visible    (vec4)   // .xyz inverse 2D Σ, .w 1.0/0.0
color_opacity    (vec4)   // .xyz SH-evaluated RGB, .w sigmoid(opacity_logit)
```

### Projection

```text
t_cam   = view ⋅ (p_world, 1)
depth   = −t_cam.z                       (right-handed view → positive forward)
mean.x  = cx + focal ⋅ t_cam.x / depth
mean.y  = cy − focal ⋅ t_cam.y / depth   (flip Y to image-down coords)
```

Cull splats with `depth < near` (sub-near-plane).

### 3D → 2D covariance

\\[
\Sigma_{3D} = R \cdot \mathrm{diag}(\sigma^2) \cdot R^T, \quad
\sigma = \exp(\mathrm{log\_scale})
\\]

In camera space:

\\[
\Sigma_{\mathrm{cam}} = W \cdot \Sigma_{3D} \cdot W^T, \quad
W = \mathrm{view}_{3\times3}
\\]

Pinhole projection's Jacobian at `t`:

\\[
J = \begin{pmatrix}
f / t_z & 0 & -f t_x / t_z^2 \\
0 & -f / t_z & f t_y / t_z^2
\end{pmatrix}
\\]

The Y row carries the image-flip sign. Screen-space covariance:

\\[
\Sigma_{2D} = J \cdot \Sigma_{\mathrm{cam}} \cdot J^T + 0.3 \cdot I
\\]

The `0.3 · I` term is **Inria's 3-pixel dilation low-pass**: without
it, splats that collapse to sub-pixel size alias hard. After dilation
we always have `det(Σ_2D) > 0`.

### Conic + radius

`conic = inv(Σ_2D)` — three floats `(a, b, c)` for the bilinear form
`½ (a·dx² + 2b·dx·dy + c·dy²)`. The 3σ pixel radius comes from the
larger eigenvalue of `Σ_2D`:

```text
mid     = ½ (Σ_00 + Σ_11)
λ_max   = mid + √(mid² − det)
radius  = ⌈3 · √λ_max⌉
```

Off-screen splats (mean outside `[−radius, width+radius] ×
[−radius, height+radius]`) write `visible = 0` and are skipped by
all later passes.

### SH evaluation

Order-3 real SH (16 coefficients per channel) evaluated at the view
direction `to_cam = normalize(cam_pos − p_world)`:

```text
rgb = SH_C0 · sh_dc
    + Σ_{l=1..3, m=-l..l} sh_rest[l*l + m + l] · Y_lm(to_cam)
```

3DGS viewers expect the +0.5 offset baked in (CPU LSQ fitter
produces `srgb - 0.5`), so we add `+0.5` before clamping to ≥ 0.

### Opacity

`opacity_sigmoid = sigmoid(opacity_logit)`. The kernel emits the
post-sigmoid value (raster uses it directly).

## Stage 2 — Tile binning

The binning pipeline produces three buffers consumed by the
rasteriser:

- `sorted_keys: u32` — packed `(tile_id << 16) | depth_u16`.
- `sorted_payloads: u32` — splat index.
- `tile_ranges: u32 × 2 · num_tiles` — per-tile `[begin, end)` over
  the sorted stream.

The packing limits the system to ≤ 65k tiles (image ≤ 4096×4096 with
16-pixel tiles) and 16-bit depth quantisation. Depth is mapped
linearly into `[near, depth_max]`; depth_max is a `TilingParams`
field, set conservatively to `4 · orbit_distance` in the viewer and
`2 · orbit_radius` in `train()`.

### Sub-passes

1. **`tile_count.wgsl`** — per splat, compute tile bbox from
   `(mean, radius)`, write `tiles_touched` and atomically add to the
   grand-total counter.
2. **CPU readback** of the total — exact `(splat × tile)` allocation.
3. **`PrefixScan::scan`** over `tile_counts` — see
   [Shaders](./shaders.md) for the three-level Hillis-Steele.
4. **`tile_emit.wgsl`** — per splat, walk its tile bbox and emit
   `(key, splat_idx)` pairs starting at `scan_offsets[i]`.
5. **`RadixSort::sort`** — 1-bit-at-a-time stable sort over the
   keys. See [Backward](./training-backward.md) end-note for why
   byte-radix with atomic scatter was abandoned.
6. **`tile_ranges.wgsl`** — per entry, compare neighbour keys'
   tile-ids and write `[begin, end)` on transitions.

Empty scene (zero pairs) is fast-pathed before the scan: skip the
remaining sub-passes and return empty results.

## Stage 3 — `rasterize.wgsl`

One workgroup per tile, 16×16 = 256 threads (one per pixel). Each
tile owns `tile_ranges[tile_id] = [begin, end)` over the sorted
stream.

Per-tile state machine:

```text
T = 1
C = vec3(0)
for chunk in chunks_of_256(splat_range):
    # cooperative load — thread `lidx` loads the lidx-th splat
    shared_mean[lidx]    = projected[sorted_payloads[chunk[lidx]]].mean
    shared_conic[lidx]   = projected[sorted_payloads[chunk[lidx]]].conic
    shared_color[lidx]   = projected[sorted_payloads[chunk[lidx]]].color
    shared_opacity[lidx] = projected[sorted_payloads[chunk[lidx]]].opacity
    workgroupBarrier()
    # process all 256 splats against my pixel
    for k in 0..chunk_size:
        dx = pix - shared_mean[k]
        power = -½ (a·dx² + 2b·dx·dy + c·dy²)
        if power > 0: continue                # numerical safety net
        α = min(σ · exp(power), 0.99)
        if α < 1/255: continue                # below sensor noise
        C += T · α · color
        T *= (1 − α)
        if T < 1e-4: done = true; break       # forward early-out
    workgroupBarrier()
```

The chunked cooperative load amortises one global SSBO read over
256 pixels — that's the speedup over a naive "each pixel reads each
splat from global memory" loop.

### Output

`output[pixel_y · width + pixel_x] = vec4(C, 1 − T)`. The W
channel doubles as the per-pixel "coverage" needed by the backward
pass to reconstruct T without storing per-splat intermediates.

## Cull and stability knobs

- `α < 1/255` threshold (`ALPHA_MIN`) skips invisible contributions
  early — equivalent to Inria's `α_min`.
- `α ≤ 0.99` clamp (`ALPHA_MAX`) keeps `1 − α ≥ 0.01`. Without this
  cap, the backward T-reconstruction divides by an arbitrarily small
  number.
- `T < 1e-4` early-out limits the per-pixel splat loop. The backward
  walk has its own corresponding `T > 1.0001` early-out — see
  [Backward](./training-backward.md).
- `power > 0` cull is a paranoia check. With `det(Σ_2D) > 0.3`
  guaranteed by the dilation, `power` is non-positive analytically;
  the check protects against any future bug that breaks the
  invariant.

## Loss — combined MSE + SSIM

After readback of the predicted frame we evaluate the Inria-style
combined objective in `nano_optimize::loss`:

\\[
L = (1 - \lambda)\,\mathrm{MSE}(\hat{C}, C^*) + \lambda\,\bigl(1 - \mathrm{SSIM}(\hat{C}, C^*)\bigr)
\\]

- `MSE` — mean squared per-pixel RGB. Same definition as the old
  pure-MSE path; gradient is `dL/dC = 2(pred − target) / (W·H·3)`.
- `SSIM` — Wang 2004 structural similarity over an **11×11 separable
  Gaussian window** with `σ = 1.5`. Stabiliser constants
  `C1 = (0.01)² = 1e-4` and `C2 = (0.03)² = 9e-4`. Reflection padding
  on the window edges, average over R/G/B channels.
- `λ ∈ [0, 1]` — `--ssim-lambda` flag. Inria default is `0.2`. At
  `λ = 0` the pass is bit-equivalent to the pre-SSIM path (the SSIM
  convolutions are skipped entirely).

MSE alone plateaus once the splat cloud is locally correct: every
pixel's mean is roughly right but the local structure (edges,
gradients, fine texture) is smeared. SSIM scores those local
windows, so adding a small `λ` carries the optimiser past that
plateau in the last ~20% of training without breaking the bulk of
the L2-driven convergence earlier on.

Gradient flow: the analytic `dSSIM/dx` simplifies to a cascade of
three Gaussian smoothings over per-pixel quotient-rule coefficients
(see `loss::ssim_loss_grad_channel`). The combined gradient is the
λ-weighted sum of the MSE and SSIM gradients (linearity of
derivatives), so the GPU backward pass receives a single `dL/dC`
buffer just as before.
