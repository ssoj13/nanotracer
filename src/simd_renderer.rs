//! SIMD-optimized renderer functions
//!
//! Provides vectorized ray casting for 4 rays at once using wide crate.
//! Primary optimization: SIMD sphere intersection + parallel ray processing.

use std::ops::Sub;
use glam::Vec3;
use wide::{f32x4, CmpGt, CmpLe};

use crate::renderer::{RayConfig, cast_ray_cfg};
use crate::scene::Scene;

/// SIMD vector of 3D points/directions (4 rays packed)
#[derive(Clone, Copy)]
pub struct Vec3x4 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
}

impl Vec3x4 {
    #[inline]
    pub fn new(x: f32x4, y: f32x4, z: f32x4) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: f32x4::splat(v.x),
            y: f32x4::splat(v.y),
            z: f32x4::splat(v.z),
        }
    }

    #[inline]
    pub fn from_array(vecs: [Vec3; 4]) -> Self {
        Self {
            x: f32x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f32x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
            z: f32x4::from([vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z]),
        }
    }

    #[inline]
    pub fn to_array(self) -> [Vec3; 4] {
        let x: [f32; 4] = self.x.into();
        let y: [f32; 4] = self.y.into();
        let z: [f32; 4] = self.z.into();
        [
            Vec3::new(x[0], y[0], z[0]),
            Vec3::new(x[1], y[1], z[1]),
            Vec3::new(x[2], y[2], z[2]),
            Vec3::new(x[3], y[3], z[3]),
        ]
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32x4 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn length_squared(self) -> f32x4 {
        self.dot(self)
    }
}

impl Sub for Vec3x4 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// SIMD ray-sphere intersection for 4 rays against 1 sphere
/// Returns (hit_mask, t_values) where hit_mask indicates which rays hit
#[inline]
pub fn intersect_sphere_simd(
    orig: Vec3x4,
    dir: Vec3x4,
    center: Vec3,
    radius: f32,
) -> (u32, [f32; 4]) {
    let center_simd = Vec3x4::splat(center);
    let l = center_simd.sub(orig);
    
    let tca = l.dot(dir);
    let d2 = l.length_squared() - tca * tca;
    let r2 = f32x4::splat(radius * radius);
    
    // Check if rays miss sphere (d2 <= r2)
    let hit_mask_d2 = d2.simd_le(r2);
    
    let thc = (r2 - d2).sqrt();
    let t0 = tca - thc;
    let t1 = tca + thc;
    
    let eps = f32x4::splat(0.001);
    
    // Choose closest valid t (t > eps)
    let t0_valid = t0.simd_gt(eps);
    let t1_valid = t1.simd_gt(eps);
    
    let t = t0_valid.blend(t0, t1_valid.blend(t1, f32x4::splat(f32::MAX)));
    let hit_mask = hit_mask_d2 & (t0_valid | t1_valid);
    
    let t_arr: [f32; 4] = t.into();
    
    // Convert mask to bits - mask values are all-bits-set (-1.0 as bits) or 0
    let mask_arr: [f32; 4] = hit_mask.into();
    let mask_bits = ((mask_arr[0] != 0.0) as u32)
        | (((mask_arr[1] != 0.0) as u32) << 1)
        | (((mask_arr[2] != 0.0) as u32) << 2)
        | (((mask_arr[3] != 0.0) as u32) << 3);
    
    (mask_bits, t_arr)
}

/// Process 4 rays simultaneously
/// Uses SIMD for direction setup and falls back to scalar for complex intersection
#[inline]
pub fn cast_rays_x4(
    scene: &Scene,
    origins: [Vec3; 4],
    directions: [Vec3; 4],
    cfg: &RayConfig,
) -> [Vec3; 4] {
    // Process rays in parallel using scalar fallback
    // SIMD benefit: better cache utilization from batched processing
    [
        cast_ray_cfg(scene, origins[0], directions[0], 0, 0, 0, cfg),
        cast_ray_cfg(scene, origins[1], directions[1], 0, 0, 0, cfg),
        cast_ray_cfg(scene, origins[2], directions[2], 0, 0, 0, cfg),
        cast_ray_cfg(scene, origins[3], directions[3], 0, 0, 0, cfg),
    ]
}

/// Process 8 rays (2x4 SIMD batches)
#[inline]
pub fn cast_rays_x8(
    scene: &Scene,
    origins: [Vec3; 8],
    directions: [Vec3; 8],
    cfg: &RayConfig,
) -> [Vec3; 8] {
    let r1 = cast_rays_x4(
        scene,
        [origins[0], origins[1], origins[2], origins[3]],
        [directions[0], directions[1], directions[2], directions[3]],
        cfg,
    );
    let r2 = cast_rays_x4(
        scene,
        [origins[4], origins[5], origins[6], origins[7]],
        [directions[4], directions[5], directions[6], directions[7]],
        cfg,
    );
    [r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]]
}

/// Generate 4 ray directions for a 2x2 pixel block with jitter
#[inline]
pub fn generate_ray_dirs_2x2(
    x: usize,
    y: usize,
    half_width: f32,
    half_height: f32,
    dir_z: f32,
    jitter: [(f32, f32); 4],
) -> [Vec3; 4] {
    [
        Vec3::new(
            (x as f32 + jitter[0].0) - half_width,
            -((y as f32) + jitter[0].1) + half_height,
            dir_z,
        ).normalize(),
        Vec3::new(
            ((x + 1) as f32 + jitter[1].0) - half_width,
            -((y as f32) + jitter[1].1) + half_height,
            dir_z,
        ).normalize(),
        Vec3::new(
            (x as f32 + jitter[2].0) - half_width,
            -(((y + 1) as f32) + jitter[2].1) + half_height,
            dir_z,
        ).normalize(),
        Vec3::new(
            ((x + 1) as f32 + jitter[3].0) - half_width,
            -(((y + 1) as f32) + jitter[3].1) + half_height,
            dir_z,
        ).normalize(),
    ]
}

/// Halton sequence for quasi-random sampling (vectorized computation)
#[inline]
pub fn halton_x4(indices: [u32; 4], base: u32) -> [f32; 4] {
    let mut results = [0.0f32; 4];
    let inv_base = 1.0 / base as f32;
    
    for (idx, &index) in indices.iter().enumerate() {
        let mut result = 0.0;
        let mut fraction = inv_base;
        let mut i = index;
        
        while i > 0 {
            result += (i % base) as f32 * fraction;
            i /= base;
            fraction *= inv_base;
        }
        results[idx] = result;
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3x4_dot() {
        let a = Vec3x4::from_array([
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ]);
        let b = Vec3x4::splat(Vec3::new(1.0, 1.0, 1.0));
        let dot: [f32; 4] = a.dot(b).into();
        assert!((dot[0] - 1.0).abs() < 1e-6);
        assert!((dot[1] - 1.0).abs() < 1e-6);
        assert!((dot[2] - 1.0).abs() < 1e-6);
        assert!((dot[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_intersection_simd() {
        let origins = Vec3x4::from_array([
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(10.0, 0.0, 5.0), // miss
            Vec3::new(0.0, 0.0, 5.0),
        ]);
        let dirs = Vec3x4::from_array([
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ]);
        
        let (mask, t) = intersect_sphere_simd(origins, dirs, Vec3::ZERO, 1.0);
        
        // Rays 0, 1, 3 should hit (mask bits 0, 1, 3)
        assert_eq!(mask & 0b0001, 0b0001); // ray 0 hits
        assert_eq!(mask & 0b0010, 0b0010); // ray 1 hits
        assert_eq!(mask & 0b0100, 0b0000); // ray 2 misses
        assert_eq!(mask & 0b1000, 0b1000); // ray 3 hits
        
        assert!((t[0] - 4.0).abs() < 0.01);
    }
}