# Backward pass

Two kernels chain `dL/dC` (per-pixel loss gradient on the predicted
framebuffer) back to per-splat parameter gradients:

1. `rasterize_backward.wgsl` — per-pixel reverse α-blend writes
   `ProjectedGrad` (2D-state gradient) via atomic-add-f32.
2. `project_backward.wgsl` — per-splat chain from 2D state to 3D
   parameters (`pos`, `rot`, `log_σ`, `opacity_logit`, `sh_dc`,
   `sh_rest`).

Finite-difference verification (`tests/backward_finite_diff.rs`) pins
every chain-rule step to ≤ 5–10 % relative error at ε = 1e-3.

## `rasterize_backward.wgsl`

Walks the **same** sorted tile slice as forward, but back-to-front.
Per pixel maintains:

- `T` — transmittance after the splat we're about to process (init
  from `1 - forward_out[pix].w`).
- `S` — `Σ_{j>i} T_j α_j c_j` (accumulator, init 0).

For each splat `i` (from `end-1` down to `begin`):

\\[
T_i = \frac{T}{1 - \alpha_i}, \quad
\mathrm{contrib}_i = T_i \cdot \alpha_i \cdot c_i
\\]

\\[
\frac{\partial C}{\partial c_i} = T_i \alpha_i, \qquad
\frac{\partial C}{\partial \alpha_i} = T_i c_i - \frac{S}{1 - \alpha_i}
\\]

Per-pixel gradient contributions:

```text
grad_color_pix  = dL/dC ⋅ T_i ⋅ α_i           (vec3)
grad_alpha_pix  = dot(dL/dC, T_i·c_i − S/(1−α_i))  (scalar)
```

Chain `α` to `(σ_opacity, conic, mean)` via:

```text
α = min(σ_opacity · exp(power), α_max),  power = −½ xᵀΣ⁻¹x
saturation_gate = (α_raw < α_max) ? 1 : 0
grad_α_raw      = grad_α_pix · saturation_gate

grad_σ_opacity  = grad_α_raw · g                 (g = exp(power))
grad_power      = grad_α_raw · σ_opacity · g

# dpower/d(conic.{a,b,c}) = −½ (dx², 2·dx·dy, dy²)
grad_conic.a    = grad_power · −½ · dx²
grad_conic.b    = grad_power · −½ · 2·dx·dy
grad_conic.c    = grad_power · −½ · dy²

# dpower/d(mean.x) = a·dx + b·dy   (after the −½ and −1 chain cancel)
grad_mean.x     = grad_power · (a·dx + b·dy)
grad_mean.y     = grad_power · (b·dx + c·dy)
```

All four contributions accumulate into the splat's
`projected_grad[idx]` slot via `atomic_add_f32` (CAS loop on `u32`
bitcasts — WGSL has no native float atomics).

State update for the next iteration (earlier splat):

```text
S = S + contrib_i        # = Σ_{j ≥ i} T_j α_j c_j now
T  stays unchanged        # T was T_{i+1}, becomes T_i = same value semantically
```

Wait — that last line needs care. After `T = T / (1−α)`, the local
variable holds `T_i`, which is also the `T_{i+1}` value the *next*
splat (index `i−1`) will see. So `T` is correct as-is for the next
iteration. No re-assignment needed.

### The numerical-stability T-clamp

Forward composites bail at `T < 1e-4` (early-out). After that point,
earlier splats had zero contribution to the final pixel — but their
α-values are still nonzero, so dividing through them on the backward
walk just amplifies floating-point error.

A pixel that early-outed at iter K of the forward pass starts
backward with `T_final = T_K ≈ 10⁻⁵`. Walking through splats with
α ≈ 0.99 multiplies `T` by `1/0.01 = 100` each step. After 10
such splats, `T ≈ 10⁻⁵ · 100¹⁰ = 10¹⁵`. Beyond `T = 1` we're past
the legitimate transmittance range — those splats didn't contribute
to the forward result, gradients should be zero.

```glsl
T = T / one_minus_alpha;
if (T > 1.0001) { done = true; break; }   // crossed early-out boundary
```

Without this clamp: NaN floods through atomic adds on a 5800-splat
scene (gradient norms reported as NaN, splats stop moving).

With it: clean gradients. Finite-difference verification still
passes — the clamp only kicks in for pixels where forward already
gave up.

## `project_backward.wgsl`

One thread per splat. Reads `projected_grad[idx]` and chains back
to 3D parameters. Writes to `GradSplatBuffers` slots; no atomics
needed (single thread per splat = no contention).

### Opacity-logit gradient

```text
grad_opacity_logit = grad_α_sigmoid · α · (1 − α)
```

where `α = sigmoid(opacity_logit)` is the splat's stored opacity
sigmoid (read from `color_opacity.w`).

### SH gradients

\\[
\frac{\partial \mathrm{color}_k}{\partial \mathrm{sh\_dc}_k} = \mathrm{SH\_C}_0
\\]

For each band-1..3 coefficient at index `r` (planar layout —
`r ∈ [0, 15)` indexes into the 15-coefficient-per-channel slab):

\\[
\frac{\partial \mathrm{color}_k}{\partial \mathrm{sh\_rest}[k][r]} = Y_{l(r), m(r)}(\hat{\omega})
\\]

`ŵ = normalize(cam_pos − p_world)` — the view direction at the splat.

Known limitation: `ŵ` itself depends on `p_world`, so a strictly
correct backward would also chain SH gradients through position.
We drop that term (matches brush's reference implementation); the
contribution is small compared to the projection's direct effect
on `mean_xy`.

### Conic → Σ_2D

`conic = inv(Σ_2D + filter)`. For a 2×2 symmetric input:

\\[
\frac{\partial L}{\partial \Sigma_{2D}} = -\Sigma_{2D}^{-1} \cdot \frac{\partial L}{\partial \text{conic}} \cdot \Sigma_{2D}^{-1} = -C \cdot \frac{\partial L}{\partial \text{conic}} \cdot C
\\]

The off-diagonal element `b` appears twice in the symmetric matrix,
so it's halved before the sandwich and the result's off-diagonals
are summed when materialising `dL/dΣ_2D`.

### Σ_2D → Σ_3D (Jacobian sandwich)

Both sides:

\\[
\frac{\partial L}{\partial \Sigma_{3D}} = (J \cdot W)^T \cdot \frac{\partial L}{\partial \Sigma_{2D}} \cdot (J \cdot W)
\\]

Implemented in WGSL by manually scattering the `(2 × 2 → 3 × 3)`
sandwich into six `dcov_cam_ij` accumulators. Then
`dcov_world = Wᵀ · dcov_cam · W` (one mat3x3 expression).

### Σ_3D → R, σ

\\[
\Sigma_{3D} = R \cdot S^2 \cdot R^T, \qquad S = \mathrm{diag}(\sigma_x, \sigma_y, \sigma_z)
\\]

For symmetric `dL/dΣ_3D`:

\\[
\frac{\partial L}{\partial R} = 2 \cdot \frac{\partial L}{\partial \Sigma_{3D}} \cdot R \cdot S^2
\\]

Per-axis scale:

\\[
\frac{\partial L}{\partial \sigma_i^2} = R_{:,i}^T \cdot \frac{\partial L}{\partial \Sigma_{3D}} \cdot R_{:,i}
\\]

\\[
\frac{\partial L}{\partial \log \sigma_i} = \frac{\partial L}{\partial \sigma_i^2} \cdot 2 \sigma_i^2
\\]

(The `2σ²` factor compounds the `σ → σ²` and `log σ → σ` chains.)

### R → quaternion

Splat rotation is stored `(w, x, y, z)`. The closed-form derivatives
of each `R[i,j]` w.r.t. each quat component yield:

```text
dL/dw = 2 · ( z·(G[1,0] − G[0,1]) + y·(G[0,2] − G[2,0]) + x·(G[2,1] − G[1,2]) )

dL/dx = 2 · ( y·(G[0,1] + G[1,0]) + z·(G[0,2] + G[2,0]) + w·(G[2,1] − G[1,2]) )
     − 4 · x · (G[1,1] + G[2,2])

dL/dy = −4 · y · (G[0,0] + G[2,2])
     + 2 · ( x·(G[0,1] + G[1,0]) + w·(G[0,2] − G[2,0]) + z·(G[1,2] + G[2,1]) )

dL/dz = −4 · z · (G[0,0] + G[1,1])
     + 2 · ( w·(G[1,0] − G[0,1]) + x·(G[0,2] + G[2,0]) + y·(G[1,2] + G[2,1]) )
```

where `G[i,j] = (dL/dR)[i,j]` in (row, col) order.

After the Adam step the result drifts off the unit hyper-sphere, so
`train()` re-normalises every iteration. See [Adam +
densify + prune](./training-adam.md).

### mean_xy → position

\\[
\frac{\partial \mathrm{mean.x}}{\partial t_{cam}} = (f / t_z, \; 0, \; -f t_x / t_z^2), \qquad
\frac{\partial \mathrm{mean.y}}{\partial t_{cam}} = (0, \; -f / t_z, \; f t_y / t_z^2)
\\]

Account for the camera-space-Z flip:

```text
dL/dt_cam = dL/dmean.x · ∂mean.x/∂t + dL/dmean.y · ∂mean.y/∂t
dL/d(cam_xyz) = (dL/dt.x, dL/dt.y, −dL/dt.z)    # depth = −cam_z
dL/d(p_world) = Wᵀ · dL/d(cam_xyz)              # cam = W · p_world
```

## Why 1-bit-at-a-time radix sort

First implementation used 4-pass byte radix with atomic-fetch-add
scatter. That gave correct **bucket counts** but the **order within
a bucket** was determined by atomicAdd race-winning order — not
stable.

For radix sort, byte-(k+1)'s pass relies on byte-k order being
preserved for pairs with equal byte-(k+1). Non-stable scatter breaks
this invariant on pass 2 onward; the final output had ~50 % of
keys out of order.

The fix is stable scatter. We do 32 passes of 1-bit radix:
1. `bit_predicate.wgsl` writes `predicate[i] = (bit i is 0) ? 1 : 0`.
2. `PrefixScan` over `predicate`.
3. `bit_total_zeros.wgsl` adds `predicate[n-1]` to `scan[n-1]` for
   the total count of zero-bit elements.
4. `bit_scatter.wgsl` computes `dst = (bit==0) ? scan[i] :
   total_zeros + (i − scan[i])`. Every thread gets a unique `dst`
   by construction — no atomics, perfectly stable.

32 passes × 4 kernels per pass = 128 kernel dispatches for a sort.
For 10M-key inputs that's still a few ms on a discrete GPU.
