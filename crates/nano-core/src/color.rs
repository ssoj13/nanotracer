//! Shared color pipeline utilities (tonemap + sRGB conversion).

use glam::Vec3;

pub fn tonemap_reinhard(color: Vec3) -> Vec3 {
    let c = color.max(Vec3::ZERO);
    c / (Vec3::ONE + c)
}

pub fn linear_to_srgb(linear: Vec3) -> Vec3 {
    fn channel(v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    Vec3::new(channel(linear.x), channel(linear.y), channel(linear.z))
}

/// Apply optional tonemapping (Reinhard) and convert to sRGB.
pub fn apply_tonemap_srgb(color_linear: Vec3, tonemap: bool) -> Vec3 {
    let mapped = if tonemap {
        tonemap_reinhard(color_linear)
    } else {
        color_linear.max(Vec3::ZERO).min(Vec3::ONE)
    };

    linear_to_srgb(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tonemap_black_is_black() {
        let c = apply_tonemap_srgb(Vec3::ZERO, true);
        assert_eq!(c, Vec3::ZERO);
    }

    #[test]
    fn test_tonemap_off_clamps_white() {
        let c = apply_tonemap_srgb(Vec3::splat(1.0), false);
        let expected = Vec3::splat(1.0);
        let diff = (c - expected).abs().max_element();
        assert!(diff < 1e-6);
    }
}
