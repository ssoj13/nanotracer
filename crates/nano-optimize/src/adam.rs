//! Adam optimiser state for a flat parameter vector.
//!
//! Inria 3DGS uses Adam with per-parameter learning rates and an exponential
//! schedule on `position_lr`. This module is the bare numerical kernel —
//! the wrapping logic that maps `SplatBuffer` attributes to flat parameter
//! slabs (and back) lives in [`train`](crate::train).
//!
//! Each parameter has two moment estimates `m` (first) and `v` (second).
//! The update is the standard
//!   `m ← β₁·m + (1 − β₁)·g`
//!   `v ← β₂·v + (1 − β₂)·g²`
//!   `m̂ ← m / (1 − β₁ᵗ)`
//!   `v̂ ← v / (1 − β₂ᵗ)`
//!   `θ ← θ − lr · m̂ / (√v̂ + ε)`
//!
//! We don't store `t` per parameter — bias correction is applied with the
//! optimiser's shared step counter.

/// Adam hyperparameters. Defaults match the Inria 3DGS reference
/// (`adam.py`) for the SH-coefficient channel; per-attribute learning rates
/// override `lr` in the training loop.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1.6e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-15,
        }
    }
}

/// Adam moment-estimate state for a single flat parameter slab.
///
/// `m` and `v` are kept on the CPU until the rasteriser becomes GPU-resident
/// (Phase A3 — backward pass). In Phase A2 the rasteriser computes
/// gradients on the GPU and reads them back to apply Adam here; by A3 the
/// optimiser will also live on the GPU.
#[derive(Debug, Clone)]
pub struct AdamState {
    pub config: AdamConfig,
    /// Step counter (incremented once per `step` call), shared across all
    /// parameter slabs. Used for the bias-correction exponent.
    pub t: u64,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl AdamState {
    pub fn new(config: AdamConfig, n_params: usize) -> Self {
        Self {
            config,
            t: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }

    /// Apply one Adam step in-place to `params` given matching `grads`.
    /// Returns silently if the lengths mismatch — the training loop is
    /// expected to guarantee them.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());
        debug_assert_eq!(params.len(), self.v.len());

        self.t += 1;
        let bc1 = 1.0 - self.config.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.config.beta2.powi(self.t as i32);

        for ((p, &g), (m, v)) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *m = self.config.beta1 * *m + (1.0 - self.config.beta1) * g;
            *v = self.config.beta2 * *v + (1.0 - self.config.beta2) * g * g;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            *p -= self.config.lr * m_hat / (v_hat.sqrt() + self.config.eps);
        }
    }

    /// Densify support — append zero-initialised moment entries when
    /// splats are added. Must be called in lock-step with the SplatBuffer.
    pub fn grow(&mut self, additional_params: usize) {
        self.m.resize(self.m.len() + additional_params, 0.0);
        self.v.resize(self.v.len() + additional_params, 0.0);
    }

    /// Prune support — swap-remove a contiguous range of `n_params`
    /// entries starting at `start`. Matches `SplatBuffer::swap_remove`'s
    /// O(1) pattern.
    pub fn swap_remove_range(&mut self, start: usize, n_params: usize) {
        let len = self.m.len();
        debug_assert!(start + n_params <= len);
        let last = len - n_params;
        for k in 0..n_params {
            self.m.swap(start + k, last + k);
            self.v.swap(start + k, last + k);
        }
        self.m.truncate(last);
        self.v.truncate(last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_moves_param_against_gradient() {
        let mut s = AdamState::new(AdamConfig::default(), 3);
        let original = [1.0_f32, 2.0, 3.0];
        let mut params = original.to_vec();
        let grads = vec![1.0_f32, 1.0, 1.0]; // positive gradient — Adam should subtract.
        s.step(&mut params, &grads);
        for (p, o) in params.iter().zip(original.iter()) {
            assert!(*p < *o, "param {p} should have decreased from {o}");
        }
        assert_eq!(s.t, 1);
    }

    #[test]
    fn moments_persist_across_steps() {
        let mut s = AdamState::new(AdamConfig::default(), 1);
        for _ in 0..10 {
            s.step(&mut [0.0], &[1.0]);
        }
        assert!(s.m[0] > 0.0);
        assert!(s.v[0] > 0.0);
    }

    #[test]
    fn grow_and_shrink_match_buffer_pattern() {
        let mut s = AdamState::new(AdamConfig::default(), 4);
        s.grow(2);
        assert_eq!(s.m.len(), 6);
        s.swap_remove_range(1, 2);
        assert_eq!(s.m.len(), 4);
    }
}
