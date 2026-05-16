# Adam + densify + prune

Adam updates and the maintenance passes (`prune`, `densify`) live on
the CPU. Splats and Adam moments are uploaded / read back through
`GpuSplatBuffer::sync_from` and `GpuSplatBuffer::upload`.

## Adam per attribute

Six independent `AdamState` instances, one per splat attribute, sized
to that attribute's per-splat stride:

| Attribute        | Stride per splat (f32) | Notes |
|------------------|------------------------|-------|
| `positions`      | 3                      | Vec3 |
| `rotations`      | 4                      | (w, x, y, z) quaternion |
| `scales`         | 3                      | log-σ |
| `opacities`      | 1                      | logit |
| `sh_dc`          | 3                      | RGB DC band |
| `sh_rest`        | 45                     | Planar R[0..15] G[0..15] B[0..15] |

Per iteration:

1. Read back the matching `d_*` buffer from the GPU.
2. Flatten gradients to a single `Vec<f32>` (strip vec4 padding,
   `sh_rest` 48 → 45).
3. Flatten the *current* SplatBuffer attribute to a matching
   `Vec<f32>`.
4. `AdamState::step(&mut params_flat, &grads_flat)` does the standard
   update:

   \\[
   m \leftarrow \beta_1 m + (1-\beta_1) g
   \\]
   \\[
   v \leftarrow \beta_2 v + (1-\beta_2) g^2
   \\]
   \\[
   \hat m = m / (1 - \beta_1^t), \quad \hat v = v / (1 - \beta_2^t)
   \\]
   \\[
   \theta \leftarrow \theta - \mathrm{lr} \cdot \hat m / (\sqrt{\hat v} + \epsilon)
   \\]

5. Unflatten back into `SplatBuffer`, applying physical constraints:
   - **Quaternion**: re-normalise to unit length. Adam drifts off
     the unit hyper-sphere within ~100 iters otherwise.
   - **`log_σ`**: clamp to `[-10, 5]` (σ ∈ [4.5e-5, 148]). Prevents
     runaway `exp` that trips `det(Σ_2D) ≤ 0`.
   - **`opacity_logit`**: clamp to `[-10, 10]`. Keeps sigmoid in
     [4.5e-5, 0.9999] so the prune threshold (`-5.3` ≈ σ 0.005)
     remains reachable from both sides.

## Hyperparameters

Inria-style per-attribute learning rates. `position` needs a
noticeably higher rate to escape its forward-fit seed; the others
stay conservative:

```rust
// From src/main.rs train_cfg
adam_pos: AdamConfig {
    lr: 1.6e-4,
    beta1: 0.9, beta2: 0.999, eps: 1e-15,
},
adam_attr: AdamConfig::default(),   // lr = 1.6e-4 same as pos
                                    // β₁ = 0.9, β₂ = 0.999, ε = 1e-15
```

## Prune

Triggered every `PRUNE_EVERY = 100` iters past `PRUNE_WARMUP = 50`.
Sweeps splats with `opacity_logit < PRUNE_OPACITY_LOGIT (-5.3)`
(σ ≈ 0.005 — invisible to the α-blend).

Implementation:

```rust
let mut to_remove: Vec<usize> = (0..splats.len())
    .filter(|&i| splats.opacities[i] < threshold)
    .collect();
to_remove.sort_unstable_by(|a, b| b.cmp(a));   // descending
for &i in &to_remove {
    splats.swap_remove(i);
    adam_pos.swap_remove_range(i * 3, 3);
    adam_rot.swap_remove_range(i * 4, 4);
    adam_scale.swap_remove_range(i * 3, 3);
    adam_op.swap_remove_range(i, 1);
    adam_dc.swap_remove_range(i * 3, 3);
    adam_rest.swap_remove_range(i * 45, 45);
    grad_acc.swap_remove(i);
}
```

`swap_remove` is O(1) per attribute; AdamState slabs use the same
swap pattern (`swap_remove_range(start, n)`).

## Densify

Triggered every `DENSIFY_EVERY = 200` iters past `DENSIFY_WARMUP = 100`.
Per-iteration accumulator `grad_acc[i] += |d_position[i]| L1` is
divided by `grad_count` at densify time.

Two policies based on the splat's largest log-σ:

| Condition                             | Action      |
|---------------------------------------|-------------|
| `avg(grad_acc[i]) ≤ DENSIFY_GRAD_THRESHOLD` (`2e-4`) | skip |
| `max(log_σ) < SPLIT_SCALE_THRESHOLD` (`-2.3` → σ < 0.1) | **CLONE** |
| `max(log_σ) ≥ -2.3` (σ ≥ 0.1)         | **SPLIT**   |

Both checks respect `max_splats` — candidates exceeding the budget
are silently skipped.

### Clone

Append a duplicate with **halved opacity** (`logit − ln 2 ≈ −0.693`);
also halve the **parent's** opacity. Two splats with half the
original opacity blend to the same total α — Adam then has room to
move them apart and pick up different details.

### Split

Append **2 children** sampled from the parent's Gaussian
`N(0, parent.Σ)`. Each child's `log_σ` is reduced by `ln 1.6 ≈
0.47` (factor 1.6 shrink). Mark the parent for removal.

Sampling uses Box-Muller on a deterministic LCG seeded from `iter`,
so a training run is reproducible given the seed. Offset is rotated
by the parent's quaternion before adding to `parent.pos`.

### Buffer reallocation

When `splats.len()` changes, GPU-side scratch buffers must
reallocate:

```rust
if splats.len() as u32 != gpu_splats.n {
    gpu_splats = GpuSplatBuffer::upload(&ctx, &splats);
    projected = ctx.storage_buffer_zeroed(
        "projected",
        (gpu_splats.n as u64) * size_of::<ProjectedSplat>() as u64);
    projected_grad = rasterizer.alloc_projected_grad(&ctx, gpu_splats.n);
    grads = GradSplatBuffers::new(&ctx, gpu_splats.n);
} else {
    gpu_splats.sync_from(&ctx, &splats);
}
```

`sync_from` writes into the existing buffers via
`queue.write_buffer` — no allocator churn on the no-change path.

## Why CPU Adam (for now)

A pure-GPU Adam (kernel reading m, v, gradients, parameters, applying
the formula in-place) would skip the per-iteration readback +
upload bounce. It's a clear performance optimisation but:

- Six independent slabs would need six dispatches (or one combined
  kernel with branchy logic).
- Quat re-normalisation and the log-σ / opacity-logit clamps would
  also need to live in WGSL.
- The constraint logic is easier to reason about on the CPU,
  especially while densify is still evolving.

Tracked under [Roadmap](./roadmap.md). For now the per-iter
~3-5 ms readback / upload is acceptable: total iter ≈ 20 ms on
discrete GPU, training 30k iters in 5–10 min.
