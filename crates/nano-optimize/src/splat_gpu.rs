//! GPU-resident counterpart of [`SplatBuffer`].
//!
//! Lays out each per-splat attribute as a dedicated storage buffer so
//! the rasteriser kernels (Phase A2) can bind them independently. Every
//! buffer is padded to `vec4` lanes (16-byte alignment) — std430 needs
//! 16-byte alignment for `vec3` anyway, and lane-aligned reads stay
//! coalesced on the SM/CU.
//!
//! Layout per splat:
//!
//! | Buffer       | Stride                | Notes                       |
//! |--------------|----------------------:|-----------------------------|
//! | `positions`  | 16 B (`vec4`)         | `xyz, 1.0`                  |
//! | `rotations`  | 16 B (`vec4`)         | quaternion `(w, x, y, z)`   |
//! | `scales`     | 16 B (`vec4`)         | log-σ `xyz, 0`              |
//! | `opacities`  |  4 B (`f32`)          | logit space                 |
//! | `sh_dc`      | 16 B (`vec4`)         | `rgb, 0`                    |
//! | `sh_rest`    | 192 B (`vec4[12]`)    | 45 floats + 3 pad           |
//!
//! Read-back strips padding so the result round-trips bit-equal back
//! into [`SplatBuffer`].

use glam::Vec3;

use crate::gpu::WgpuCtx;
use crate::splat_store::SplatBuffer;

const SH_REST_PADDED: usize = 48; // 45 active + 3 padding floats per splat

/// Per-attribute gradient buffers. Same layout as [`GpuSplatBuffer`]
/// (vec4-padded, planar SH rest) so the backward kernels can index
/// in lockstep with the forward state. All buffers are f32; the
/// backward kernel that writes them has exactly one thread per splat,
/// so no atomics are required here.
pub struct GradSplatBuffers {
    pub n: u32,
    pub d_positions: wgpu::Buffer,
    pub d_rotations: wgpu::Buffer,
    pub d_scales: wgpu::Buffer,
    pub d_opacities: wgpu::Buffer,
    pub d_sh_dc: wgpu::Buffer,
    pub d_sh_rest: wgpu::Buffer,
}

impl GradSplatBuffers {
    /// Allocate zeroed gradient buffers for `n` splats.
    pub fn new(ctx: &crate::gpu::WgpuCtx, n: u32) -> Self {
        let n_u = n.max(1) as u64;
        Self {
            n,
            d_positions: ctx.storage_buffer_zeroed("grad-pos", n_u * 16),
            d_rotations: ctx.storage_buffer_zeroed("grad-rot", n_u * 16),
            d_scales: ctx.storage_buffer_zeroed("grad-scale", n_u * 16),
            d_opacities: ctx.storage_buffer_zeroed("grad-op", n_u * 4),
            d_sh_dc: ctx.storage_buffer_zeroed("grad-sh-dc", n_u * 16),
            d_sh_rest: ctx.storage_buffer_zeroed("grad-sh-rest", n_u * (SH_REST_PADDED as u64) * 4),
        }
    }

    /// Reset every gradient buffer to zero. Cheaper than reallocating
    /// when the training loop runs the same `n` across iterations —
    /// `queue.write_buffer` writes one zero block per buffer.
    pub fn zero(&self, ctx: &crate::gpu::WgpuCtx) {
        let n = self.n.max(1) as usize;
        let zeros_vec4 = vec![0u8; n * 16];
        let zeros_f32 = vec![0u8; n * 4];
        let zeros_sh = vec![0u8; n * SH_REST_PADDED * 4];
        ctx.queue.write_buffer(&self.d_positions, 0, &zeros_vec4);
        ctx.queue.write_buffer(&self.d_rotations, 0, &zeros_vec4);
        ctx.queue.write_buffer(&self.d_scales, 0, &zeros_vec4);
        ctx.queue.write_buffer(&self.d_opacities, 0, &zeros_f32);
        ctx.queue.write_buffer(&self.d_sh_dc, 0, &zeros_vec4);
        ctx.queue.write_buffer(&self.d_sh_rest, 0, &zeros_sh);
    }
}

/// GPU-side mirror of [`SplatBuffer`]. Buffers are sized for `n` splats
/// at creation; densify (Phase A5) will reallocate by recreating the
/// `GpuSplatBuffer` from the CPU side.
pub struct GpuSplatBuffer {
    pub n: u32,
    pub positions: wgpu::Buffer,
    pub rotations: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub opacities: wgpu::Buffer,
    pub sh_dc: wgpu::Buffer,
    pub sh_rest: wgpu::Buffer,
}

impl GpuSplatBuffer {
    /// Upload a [`SplatBuffer`] to the GPU. Pads `vec3`-shaped attributes
    /// to `vec4` for std430 alignment.
    pub fn upload(ctx: &WgpuCtx, src: &SplatBuffer) -> Self {
        let n = src.len();

        let positions: Vec<[f32; 4]> =
            src.positions.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        let rotations: Vec<[f32; 4]> = src.rotations.clone();
        let scales: Vec<[f32; 4]> = src
            .scales
            .iter()
            .map(|s| [s[0], s[1], s[2], 0.0])
            .collect();
        let opacities: Vec<f32> = src.opacities.clone();
        let sh_dc: Vec<[f32; 4]> = src
            .sh_dc
            .iter()
            .map(|c| [c[0], c[1], c[2], 0.0])
            .collect();

        // 45 → 48 floats per splat, three trailing zero pads.
        let mut sh_rest: Vec<f32> = Vec::with_capacity(n * SH_REST_PADDED);
        for i in 0..n {
            let base = i * 45;
            sh_rest.extend_from_slice(&src.sh_rest[base..base + 45]);
            sh_rest.extend_from_slice(&[0.0; 3]);
        }

        Self {
            n: n as u32,
            positions: ctx.storage_buffer("splat-positions", &positions),
            rotations: ctx.storage_buffer("splat-rotations", &rotations),
            scales: ctx.storage_buffer("splat-scales", &scales),
            opacities: ctx.storage_buffer("splat-opacities", &opacities),
            sh_dc: ctx.storage_buffer("splat-sh-dc", &sh_dc),
            sh_rest: ctx.storage_buffer("splat-sh-rest", &sh_rest),
        }
    }

    /// Read back to a CPU [`SplatBuffer`]. Strips the `vec4` padding so
    /// the result is bit-identical to the source of a prior [`Self::upload`].
    pub fn readback(&self, ctx: &WgpuCtx) -> SplatBuffer {
        let n = self.n as usize;
        let positions_padded: Vec<[f32; 4]> = ctx.readback(&self.positions, n);
        let rotations: Vec<[f32; 4]> = ctx.readback(&self.rotations, n);
        let scales_padded: Vec<[f32; 4]> = ctx.readback(&self.scales, n);
        let opacities: Vec<f32> = ctx.readback(&self.opacities, n);
        let sh_dc_padded: Vec<[f32; 4]> = ctx.readback(&self.sh_dc, n);
        let sh_rest_padded: Vec<f32> = ctx.readback(&self.sh_rest, n * SH_REST_PADDED);

        let positions: Vec<Vec3> = positions_padded
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        let scales: Vec<[f32; 3]> = scales_padded.iter().map(|s| [s[0], s[1], s[2]]).collect();
        let sh_dc: Vec<[f32; 3]> = sh_dc_padded.iter().map(|c| [c[0], c[1], c[2]]).collect();

        let mut sh_rest: Vec<f32> = Vec::with_capacity(n * 45);
        for i in 0..n {
            let base = i * SH_REST_PADDED;
            sh_rest.extend_from_slice(&sh_rest_padded[base..base + 45]);
        }

        SplatBuffer {
            positions,
            rotations,
            scales,
            opacities,
            sh_dc,
            sh_rest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_buffer() -> SplatBuffer {
        let mut buf = SplatBuffer::default();
        for i in 0..3 {
            let f = i as f32;
            let mut rest = [0.0f32; 45];
            for (k, slot) in rest.iter_mut().enumerate() {
                *slot = f * 100.0 + k as f32;
            }
            buf.push_splat(
                Vec3::new(f, f * 2.0, f * 3.0),
                [1.0 + f, 0.1 * f, 0.2 * f, 0.3 * f],
                [-0.5 * f, -0.6 * f, -0.7 * f],
                0.9 * (f + 1.0),
                [0.1 * f, 0.2 * f, 0.3 * f],
                &rest,
            );
        }
        buf
    }

    /// Skips cleanly when no Vulkan/DX12/Metal adapter is available
    /// (headless CI), like the renderer's `camera_views` suite.
    #[test]
    fn roundtrip_upload_readback() {
        let ctx = match WgpuCtx::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        let src = sample_buffer();
        let gpu = GpuSplatBuffer::upload(&ctx, &src);
        let back = gpu.readback(&ctx);
        assert_eq!(back.len(), src.len());
        for i in 0..src.len() {
            assert_eq!(back.positions[i], src.positions[i]);
            assert_eq!(back.rotations[i], src.rotations[i]);
            assert_eq!(back.scales[i], src.scales[i]);
            assert_eq!(back.opacities[i], src.opacities[i]);
            assert_eq!(back.sh_dc[i], src.sh_dc[i]);
            let s = i * 45;
            assert_eq!(&back.sh_rest[s..s + 45], &src.sh_rest[s..s + 45]);
        }
    }
}
