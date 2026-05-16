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

use crate::adam::AdamState;
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

        let positions: Vec<[f32; 4]> = src.positions.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        let rotations: Vec<[f32; 4]> = src.rotations.clone();
        let scales: Vec<[f32; 4]> = src.scales.iter().map(|s| [s[0], s[1], s[2], 0.0]).collect();
        let opacities: Vec<f32> = src.opacities.clone();
        let sh_dc: Vec<[f32; 4]> = src.sh_dc.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();

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

    /// Re-upload an updated [`SplatBuffer`] into the *existing* GPU
    /// buffers (without reallocating). Caller must guarantee
    /// `src.len() == self.n` — densify / prune will replace the
    /// whole `GpuSplatBuffer` rather than calling this.
    pub fn sync_from(&self, ctx: &crate::gpu::WgpuCtx, src: &SplatBuffer) {
        let n = self.n as usize;
        debug_assert_eq!(src.len(), n);

        let positions: Vec<[f32; 4]> = src.positions.iter().map(|p| [p.x, p.y, p.z, 1.0]).collect();
        let scales: Vec<[f32; 4]> = src.scales.iter().map(|s| [s[0], s[1], s[2], 0.0]).collect();
        let sh_dc: Vec<[f32; 4]> = src.sh_dc.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let mut sh_rest: Vec<f32> = Vec::with_capacity(n * SH_REST_PADDED);
        for i in 0..n {
            let base = i * 45;
            sh_rest.extend_from_slice(&src.sh_rest[base..base + 45]);
            sh_rest.extend_from_slice(&[0.0; 3]);
        }

        ctx.queue
            .write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions));
        ctx.queue
            .write_buffer(&self.rotations, 0, bytemuck::cast_slice(&src.rotations));
        ctx.queue
            .write_buffer(&self.scales, 0, bytemuck::cast_slice(&scales));
        ctx.queue
            .write_buffer(&self.opacities, 0, bytemuck::cast_slice(&src.opacities));
        ctx.queue
            .write_buffer(&self.sh_dc, 0, bytemuck::cast_slice(&sh_dc));
        ctx.queue
            .write_buffer(&self.sh_rest, 0, bytemuck::cast_slice(&sh_rest));
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

/// GPU-resident Adam first/second moment buffers — same vec4-padded
/// layout as [`GradSplatBuffers`] so a single generic Adam kernel can
/// step every attribute. Padding lanes carry zero gradients and stay
/// at zero throughout training; the kernel writes `0 - lr * 0 = 0`
/// into them which is harmless.
///
/// Sizes per splat (matches `GpuSplatBuffer` exactly):
///   * `m_pos`,  `v_pos`   : `n × 16 B` (vec4 padded)
///   * `m_rot`,  `v_rot`   : `n × 16 B` (vec4)
///   * `m_scale`,`v_scale` : `n × 16 B` (vec4 padded)
///   * `m_op`,   `v_op`    : `n × 4  B` (f32 scalar)
///   * `m_dc`,   `v_dc`    : `n × 16 B` (vec4 padded)
///   * `m_rest`, `v_rest`  : `n × 192 B` (12 vec4, 45 active + 3 pad)
pub struct AdamMomentBuffers {
    pub n: u32,
    pub m_pos: wgpu::Buffer,
    pub v_pos: wgpu::Buffer,
    pub m_rot: wgpu::Buffer,
    pub v_rot: wgpu::Buffer,
    pub m_scale: wgpu::Buffer,
    pub v_scale: wgpu::Buffer,
    pub m_op: wgpu::Buffer,
    pub v_op: wgpu::Buffer,
    pub m_dc: wgpu::Buffer,
    pub v_dc: wgpu::Buffer,
    pub m_rest: wgpu::Buffer,
    pub v_rest: wgpu::Buffer,
}

impl AdamMomentBuffers {
    /// Allocate zeroed moment buffers for `n` splats.
    pub fn new(ctx: &WgpuCtx, n: u32) -> Self {
        let n_u = n.max(1) as u64;
        let vec4 = n_u * 16;
        let scalar = n_u * 4;
        let rest = n_u * (SH_REST_PADDED as u64) * 4;
        Self {
            n,
            m_pos: ctx.storage_buffer_zeroed("adam-m-pos", vec4),
            v_pos: ctx.storage_buffer_zeroed("adam-v-pos", vec4),
            m_rot: ctx.storage_buffer_zeroed("adam-m-rot", vec4),
            v_rot: ctx.storage_buffer_zeroed("adam-v-rot", vec4),
            m_scale: ctx.storage_buffer_zeroed("adam-m-scale", vec4),
            v_scale: ctx.storage_buffer_zeroed("adam-v-scale", vec4),
            m_op: ctx.storage_buffer_zeroed("adam-m-op", scalar),
            v_op: ctx.storage_buffer_zeroed("adam-v-op", scalar),
            m_dc: ctx.storage_buffer_zeroed("adam-m-dc", vec4),
            v_dc: ctx.storage_buffer_zeroed("adam-v-dc", vec4),
            m_rest: ctx.storage_buffer_zeroed("adam-m-rest", rest),
            v_rest: ctx.storage_buffer_zeroed("adam-v-rest", rest),
        }
    }

    /// Upload moments from CPU `AdamState` mirrors (flat unpadded
    /// layout: pos `3·n`, rot `4·n`, scale `3·n`, op `n`, dc `3·n`,
    /// rest `45·n`). Used after densify/prune mutates the CPU state.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_from_cpu(
        ctx: &WgpuCtx,
        adam_pos: &AdamState,
        adam_rot: &AdamState,
        adam_scale: &AdamState,
        adam_op: &AdamState,
        adam_dc: &AdamState,
        adam_rest: &AdamState,
    ) -> Self {
        let n = adam_op.m.len() as u32;
        let me = Self::new(ctx, n);
        me.write_pos(ctx, &adam_pos.m, &adam_pos.v);
        me.write_rot(ctx, &adam_rot.m, &adam_rot.v);
        me.write_scale(ctx, &adam_scale.m, &adam_scale.v);
        me.write_op(ctx, &adam_op.m, &adam_op.v);
        me.write_dc(ctx, &adam_dc.m, &adam_dc.v);
        me.write_rest(ctx, &adam_rest.m, &adam_rest.v);
        me
    }

    /// Re-pad a `3·n` flat slab into the vec4-padded GPU shape and
    /// upload it to `m_pos`/`v_pos`.
    pub fn write_pos(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n * 3);
        debug_assert_eq!(v.len(), n * 3);
        let m_p = pad3_to_vec4(m, n);
        let v_p = pad3_to_vec4(v, n);
        ctx.queue
            .write_buffer(&self.m_pos, 0, bytemuck::cast_slice(&m_p));
        ctx.queue
            .write_buffer(&self.v_pos, 0, bytemuck::cast_slice(&v_p));
    }

    pub fn write_rot(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n * 4);
        debug_assert_eq!(v.len(), n * 4);
        ctx.queue
            .write_buffer(&self.m_rot, 0, bytemuck::cast_slice(m));
        ctx.queue
            .write_buffer(&self.v_rot, 0, bytemuck::cast_slice(v));
    }

    pub fn write_scale(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n * 3);
        debug_assert_eq!(v.len(), n * 3);
        let m_p = pad3_to_vec4(m, n);
        let v_p = pad3_to_vec4(v, n);
        ctx.queue
            .write_buffer(&self.m_scale, 0, bytemuck::cast_slice(&m_p));
        ctx.queue
            .write_buffer(&self.v_scale, 0, bytemuck::cast_slice(&v_p));
    }

    pub fn write_op(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n);
        debug_assert_eq!(v.len(), n);
        ctx.queue
            .write_buffer(&self.m_op, 0, bytemuck::cast_slice(m));
        ctx.queue
            .write_buffer(&self.v_op, 0, bytemuck::cast_slice(v));
    }

    pub fn write_dc(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n * 3);
        debug_assert_eq!(v.len(), n * 3);
        let m_p = pad3_to_vec4(m, n);
        let v_p = pad3_to_vec4(v, n);
        ctx.queue
            .write_buffer(&self.m_dc, 0, bytemuck::cast_slice(&m_p));
        ctx.queue
            .write_buffer(&self.v_dc, 0, bytemuck::cast_slice(&v_p));
    }

    /// sh_rest mirror is `45·n` floats; pad to `48·n` (3 trailing
    /// zeros per splat) before upload.
    pub fn write_rest(&self, ctx: &WgpuCtx, m: &[f32], v: &[f32]) {
        let n = self.n as usize;
        debug_assert_eq!(m.len(), n * 45);
        debug_assert_eq!(v.len(), n * 45);
        let m_p = pad45_to_48(m, n);
        let v_p = pad45_to_48(v, n);
        ctx.queue
            .write_buffer(&self.m_rest, 0, bytemuck::cast_slice(&m_p));
        ctx.queue
            .write_buffer(&self.v_rest, 0, bytemuck::cast_slice(&v_p));
    }

    /// Read all 12 moment buffers and unpad into the CPU `AdamState`
    /// mirrors in-place. Used right before densify/prune mutates the
    /// CPU state — preserves per-splat history for survivors.
    #[allow(clippy::too_many_arguments)]
    pub fn download_to_cpu(
        &self,
        ctx: &WgpuCtx,
        adam_pos: &mut AdamState,
        adam_rot: &mut AdamState,
        adam_scale: &mut AdamState,
        adam_op: &mut AdamState,
        adam_dc: &mut AdamState,
        adam_rest: &mut AdamState,
    ) {
        let n = self.n as usize;

        let m_pos: Vec<[f32; 4]> = ctx.readback(&self.m_pos, n);
        let v_pos: Vec<[f32; 4]> = ctx.readback(&self.v_pos, n);
        adam_pos.m = unpad_vec4_to_3(&m_pos);
        adam_pos.v = unpad_vec4_to_3(&v_pos);

        let m_rot: Vec<f32> = ctx.readback(&self.m_rot, n * 4);
        let v_rot: Vec<f32> = ctx.readback(&self.v_rot, n * 4);
        adam_rot.m = m_rot;
        adam_rot.v = v_rot;

        let m_scale: Vec<[f32; 4]> = ctx.readback(&self.m_scale, n);
        let v_scale: Vec<[f32; 4]> = ctx.readback(&self.v_scale, n);
        adam_scale.m = unpad_vec4_to_3(&m_scale);
        adam_scale.v = unpad_vec4_to_3(&v_scale);

        let m_op: Vec<f32> = ctx.readback(&self.m_op, n);
        let v_op: Vec<f32> = ctx.readback(&self.v_op, n);
        adam_op.m = m_op;
        adam_op.v = v_op;

        let m_dc: Vec<[f32; 4]> = ctx.readback(&self.m_dc, n);
        let v_dc: Vec<[f32; 4]> = ctx.readback(&self.v_dc, n);
        adam_dc.m = unpad_vec4_to_3(&m_dc);
        adam_dc.v = unpad_vec4_to_3(&v_dc);

        let m_rest: Vec<f32> = ctx.readback(&self.m_rest, n * SH_REST_PADDED);
        let v_rest: Vec<f32> = ctx.readback(&self.v_rest, n * SH_REST_PADDED);
        adam_rest.m = unpad_48_to_45(&m_rest, n);
        adam_rest.v = unpad_48_to_45(&v_rest, n);
    }
}

fn pad3_to_vec4(src: &[f32], n: usize) -> Vec<[f32; 4]> {
    debug_assert_eq!(src.len(), n * 3);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([src[i * 3], src[i * 3 + 1], src[i * 3 + 2], 0.0]);
    }
    out
}

fn pad45_to_48(src: &[f32], n: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), n * 45);
    let mut out = Vec::with_capacity(n * SH_REST_PADDED);
    for i in 0..n {
        let base = i * 45;
        out.extend_from_slice(&src[base..base + 45]);
        out.extend_from_slice(&[0.0; 3]);
    }
    out
}

fn unpad_vec4_to_3(padded: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(padded.len() * 3);
    for p in padded {
        out.push(p[0]);
        out.push(p[1]);
        out.push(p[2]);
    }
    out
}

fn unpad_48_to_45(padded: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 45);
    for i in 0..n {
        let base = i * SH_REST_PADDED;
        out.extend_from_slice(&padded[base..base + 45]);
    }
    out
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
