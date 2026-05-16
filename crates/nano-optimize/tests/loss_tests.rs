//! Integration tests for the combined MSE + SSIM loss in
//! `nano_optimize::loss`. The unit-test module inside `loss.rs` already
//! covers the bulk of the contract; this file pins the public-API
//! surface from outside the crate so a regression renaming a `pub`
//! item or changing a return-tuple shape would show up here too.

use glam::Vec3;
use nano_optimize::loss::{combined_loss_grad, mse_loss_grad, ssim_loss_grad};

fn ramp_image(width: u32, height: u32, scale: f32) -> Vec<Vec3> {
    let n = (width * height) as usize;
    (0..n)
        .map(|i| {
            let f = (i as f32) / (n as f32);
            Vec3::new(scale * f, scale * (1.0 - f), scale * (0.3 + 0.4 * f))
        })
        .collect()
}

#[test]
fn ssim_identical_images_zero_loss() {
    let img = ramp_image(8, 8, 0.7);
    let (loss, _) = ssim_loss_grad(&img, &img, 8, 8);
    assert!(loss.abs() < 1e-5, "expected ~0, got {loss}");
}

#[test]
fn ssim_loss_increases_with_noise() {
    let img = ramp_image(8, 8, 0.7);
    let mut noisy = img.clone();
    let mut s: u32 = 12345;
    for px in &mut noisy {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let n = ((s >> 16) as f32 / 65535.0 - 0.5) * 0.4;
        *px += Vec3::splat(n);
    }
    let (clean, _) = ssim_loss_grad(&img, &img, 8, 8);
    let (dirty, _) = ssim_loss_grad(&noisy, &img, 8, 8);
    assert!(dirty > clean);
}

#[test]
fn ssim_grad_matches_finite_diff() {
    let width = 8;
    let height = 8;
    let n = (width * height) as usize;
    let mut pred = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for i in 0..n {
        let f = i as f32 / n as f32;
        pred.push(Vec3::new(0.3 + 0.4 * f, 0.5 - 0.3 * f, 0.2 + 0.6 * f));
        target.push(Vec3::new(0.4 + 0.3 * f, 0.6 - 0.4 * f, 0.3 + 0.5 * f));
    }
    let (_, analytic) = ssim_loss_grad(&pred, &target, width, height);
    let eps = 1e-3_f32;
    for i in 0..n {
        for ch in 0..3 {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            let bump = |v: &mut Vec3, c: usize, e: f32| match c {
                0 => v.x += e,
                1 => v.y += e,
                _ => v.z += e,
            };
            bump(&mut pp[i], ch, eps);
            bump(&mut pm[i], ch, -eps);
            let (lp, _) = ssim_loss_grad(&pp, &target, width, height);
            let (lm, _) = ssim_loss_grad(&pm, &target, width, height);
            let fd = (lp - lm) / (2.0 * eps);
            let an = match ch {
                0 => analytic[i].x,
                1 => analytic[i].y,
                _ => analytic[i].z,
            };
            assert!(
                (fd - an).abs() < 5e-3,
                "ch{ch} px{i}: fd={fd} an={an} diff={}",
                (fd - an).abs()
            );
        }
    }
}

#[test]
fn combined_lambda_zero_is_pure_mse() {
    let img = ramp_image(8, 8, 0.8);
    let tgt = ramp_image(8, 8, 0.7);
    let (mse, mse_grad) = mse_loss_grad(&img, &tgt, 8, 8);
    let (total, mse_c, ssim_c, grad) = combined_loss_grad(&img, &tgt, 8, 8, 0.0);
    assert_eq!(total, mse);
    assert_eq!(mse_c, mse);
    assert_eq!(ssim_c, 0.0);
    for (a, b) in grad.iter().zip(mse_grad.iter()) {
        assert!((a - b).length() < 1e-9);
    }
}

#[test]
fn combined_lambda_one_is_pure_ssim() {
    let img = ramp_image(8, 8, 0.8);
    let tgt = ramp_image(8, 8, 0.7);
    let (ssim, ssim_grad) = ssim_loss_grad(&img, &tgt, 8, 8);
    let (total, mse_c, ssim_c, grad) = combined_loss_grad(&img, &tgt, 8, 8, 1.0);
    assert_eq!(total, ssim);
    assert_eq!(mse_c, 0.0);
    assert_eq!(ssim_c, ssim);
    for (a, b) in grad.iter().zip(ssim_grad.iter()) {
        assert!((a - b).length() < 1e-9);
    }
}

#[test]
fn combined_gradient_is_lambda_weighted_sum() {
    let pred = ramp_image(8, 8, 0.8);
    let target = ramp_image(8, 8, 0.6);
    let lambda = 0.2_f32;
    let (_, mse_grad) = mse_loss_grad(&pred, &target, 8, 8);
    let (_, ssim_grad) = ssim_loss_grad(&pred, &target, 8, 8);
    let (_, _, _, grad) = combined_loss_grad(&pred, &target, 8, 8, lambda);
    for i in 0..pred.len() {
        let expect = mse_grad[i] * (1.0 - lambda) + ssim_grad[i] * lambda;
        assert!((grad[i] - expect).length() < 1e-6);
    }
}
