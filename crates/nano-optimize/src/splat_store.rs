//! CPU-side splat parameter buffer in a layout friendly to densify / prune.
//!
//! Adam optimises each parameter independently, so we lay out parallel
//! `Vec`s per attribute. Splat `i` is `(positions[i], rotations[i],
//! scales[i], opacities[i], sh_dc[i], sh_rest[i*45 .. (i+1)*45])`.
//! Adding/removing splats is `Vec::push` / `Vec::swap_remove` per attribute.
//!
//! GPU upload (Phase A2 — differentiable rasteriser) re-packs this into the
//! per-splat struct the WGSL rasteriser expects. CPU storage stays in this
//! AoS-friendly layout the whole time.

use glam::Vec3;
use nano_splat::ply::Gaussian;

/// Bundle of parallel `Vec`s holding every per-splat parameter.
///
/// Length invariants: `positions.len() == rotations.len() == scales.len()
///   == opacities.len() == sh_dc.len()`, and `sh_rest.len() == 45 *
///   positions.len()`.
#[derive(Debug, Clone, Default)]
pub struct SplatBuffer {
    pub positions: Vec<Vec3>,
    /// Quaternion `(w, x, y, z)` per splat.
    pub rotations: Vec<[f32; 4]>,
    /// Anisotropic scale `(sx, sy, sz)` in **log space** (`ln(σ)`).
    /// Adam optimises in log-space because raw σ must stay positive.
    pub scales: Vec<[f32; 3]>,
    /// Opacity in **logit space** (`ln(p / (1 − p))`); the rasteriser
    /// applies `sigmoid` before α-blending. Logit space lets Adam push
    /// the value unbounded.
    pub opacities: Vec<f32>,
    pub sh_dc: Vec<[f32; 3]>,
    /// 45 SH rest coefficients per splat, planar layout
    /// (`R[1..15], G[1..15], B[1..15]`) matching the Inria PLY field order.
    pub sh_rest: Vec<f32>,
}

impl SplatBuffer {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            positions: Vec::with_capacity(n),
            rotations: Vec::with_capacity(n),
            scales: Vec::with_capacity(n),
            opacities: Vec::with_capacity(n),
            sh_dc: Vec::with_capacity(n),
            sh_rest: Vec::with_capacity(n * 45),
        }
    }

    /// Seed from the forward-fit splats produced by `nano-splat`.
    /// Used by `train()` to initialise optimisation before iterating.
    pub fn from_gaussians(gaussians: &[Gaussian]) -> Self {
        let mut out = Self::with_capacity(gaussians.len());
        for g in gaussians {
            out.positions.push(g.pos);
            out.rotations.push(g.rotation);
            out.scales.push(g.scale);
            out.opacities.push(g.opacity);
            out.sh_dc.push(g.sh_dc);
            let mut rest = [0.0f32; 45];
            for (i, &v) in g.sh_rest.iter().enumerate().take(45) {
                rest[i] = v;
            }
            out.sh_rest.extend_from_slice(&rest);
        }
        out
    }

    /// Dump back to the 3DGS PLY-ready `Gaussian` type for serialisation.
    pub fn to_gaussians(&self) -> Vec<Gaussian> {
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let base = i * 45;
            let sh_rest = self.sh_rest[base..base + 45].to_vec();
            out.push(Gaussian {
                pos: self.positions[i],
                normal: Vec3::Y, // not used by viewers; written for PLY field completeness
                sh_dc: self.sh_dc[i],
                sh_rest,
                opacity: self.opacities[i],
                scale: self.scales[i],
                rotation: self.rotations[i],
            });
        }
        out
    }

    /// Remove splat at index `i` by swapping with the last element — O(1)
    /// per remove. Indexing into Adam state must use the same swap-remove
    /// pattern to stay consistent.
    pub fn swap_remove(&mut self, i: usize) {
        self.positions.swap_remove(i);
        self.rotations.swap_remove(i);
        self.scales.swap_remove(i);
        self.opacities.swap_remove(i);
        self.sh_dc.swap_remove(i);
        // 45 floats per splat: swap the i-th and last 45-float windows.
        let last = self.len(); // already decremented by the prior swap_removes
        let last_base = last * 45;
        let i_base = i * 45;
        for k in 0..45 {
            self.sh_rest.swap(i_base + k, last_base + k);
        }
        self.sh_rest.truncate(last_base);
    }

    /// Append a new splat in one go. `sh_rest_45` must have length 45.
    pub fn push_splat(
        &mut self,
        pos: Vec3,
        rotation: [f32; 4],
        scale: [f32; 3],
        opacity: f32,
        sh_dc: [f32; 3],
        sh_rest_45: &[f32],
    ) {
        debug_assert_eq!(sh_rest_45.len(), 45);
        self.positions.push(pos);
        self.rotations.push(rotation);
        self.scales.push(scale);
        self.opacities.push(opacity);
        self.sh_dc.push(sh_dc);
        self.sh_rest.extend_from_slice(sh_rest_45);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_from_to_gaussians() {
        let g = Gaussian {
            pos: Vec3::new(1.0, 2.0, 3.0),
            normal: Vec3::Y,
            sh_dc: [0.1, 0.2, 0.3],
            sh_rest: vec![0.0; 45],
            opacity: 1.5,
            scale: [-2.0, -2.0, -3.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let buf = SplatBuffer::from_gaussians(std::slice::from_ref(&g));
        let back = buf.to_gaussians();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pos, g.pos);
        assert_eq!(back[0].sh_dc, g.sh_dc);
        assert_eq!(back[0].opacity, g.opacity);
    }

    #[test]
    fn swap_remove_keeps_invariants() {
        let mut buf = SplatBuffer::default();
        for i in 0..5 {
            buf.push_splat(
                Vec3::splat(i as f32),
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0],
                1.0,
                [0.1, 0.2, 0.3],
                &[0.0; 45],
            );
        }
        buf.swap_remove(1);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.sh_rest.len(), 4 * 45);
        // index 1 should now hold the previously-last splat (index 4 → pos 4)
        assert_eq!(buf.positions[1], Vec3::splat(4.0));
    }
}
