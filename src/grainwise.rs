use std::sync::atomic::{AtomicU64, Ordering};

use rand::distributions::{Distribution, Uniform};
use rand_distr::Poisson;
use rayon::prelude::*;

use crate::model::{Derived, Plane};
use crate::params::Params;
use crate::rng;
use crate::{RenderError, RenderResult};

pub fn render_grainwise(
    lambda: &Plane,
    params: &Params,
    derived: &Derived,
    cancel: Option<&(dyn Fn() -> bool + Send + Sync)>,
) -> RenderResult<Plane> {
    let out_w = derived.output_width;
    let out_h = derived.output_height;
    let total = out_w * out_h;
    let lanes = (params.n_samples as usize).div_ceil(64);
    let bitsets: Vec<AtomicU64> = (0..total * lanes).map(|_| AtomicU64::new(0)).collect();

    let offsets = &derived.offsets;
    let zoom = params.zoom;

    let inv_samples = 1.0 / params.n_samples.max(1) as f32;
    let cancel = cancel;
    let render_result: Result<(), RenderError> =
        (0..derived.input_height).into_par_iter().try_for_each(|y| {
            if cancel.map_or(false, |check| check()) {
                return Err(RenderError::Cancelled);
            }
            for x in 0..derived.input_width {
                if cancel.map_or(false, |check| check()) {
                    return Err(RenderError::Cancelled);
                }
                let lambda_val = lambda.get(x, y);
                if lambda_val <= 0.0 {
                    continue;
                }
                let mut rng = rng::pixel_rng(params.seed, x as i32, y as i32);
                let poisson = Poisson::new(lambda_val as f64).unwrap();
                let q = poisson.sample(&mut rng) as u32;
                if q == 0 {
                    continue;
                }
                let uniform = Uniform::new(0.0f32, 1.0);
                for _ in 0..q {
                    if cancel.map_or(false, |check| check()) {
                        return Err(RenderError::Cancelled);
                    }
                    let cx = x as f32 + uniform.sample(&mut rng);
                    let cy = y as f32 + uniform.sample(&mut rng);
                    let mut radius = derived.radius.sample(&mut rng);
                    if radius > derived.rm {
                        radius = derived.rm;
                    }
                    if radius <= 0.0 {
                        continue;
                    }
                    let radius_out = radius * zoom;
                    if radius_out <= 0.0 {
                        continue;
                    }
                    let radius_sq = radius_out * radius_out;
                    for (k, offset) in offsets.iter().enumerate() {
                        if cancel.map_or(false, |check| check()) {
                            return Err(RenderError::Cancelled);
                        }
                        let tx = (cx * zoom) + offset[0];
                        let ty = (cy * zoom) + offset[1];
                        if let Some((x_min, x_max)) =
                            bounds(tx, radius_out, derived.output_width as i32)
                            && let Some((y_min, y_max)) =
                                bounds(ty, radius_out, derived.output_height as i32)
                        {
                            let lane_idx = k / 64;
                            let bit_mask = 1u64 << (k % 64);

                            for oy in y_min..=y_max {
                                if cancel.map_or(false, |check| check()) {
                                    return Err(RenderError::Cancelled);
                                }
                                let center_y = oy as f32 + 0.5;
                                let dy = center_y - ty;
                                let dy_sq = dy * dy;
                                if dy_sq > radius_sq {
                                    continue;
                                }
                                for ox in x_min..=x_max {
                                    if cancel.map_or(false, |check| check()) {
                                        return Err(RenderError::Cancelled);
                                    }
                                    let center_x = ox as f32 + 0.5;
                                    let dx = center_x - tx;
                                    if dx * dx + dy_sq <= radius_sq {
                                        let idx = oy as usize * derived.output_width + ox as usize;
                                        let slot = idx * lanes + lane_idx;
                                        bitsets[slot].fetch_or(bit_mask, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        });
    render_result?;

    if cancel.map_or(false, |check| check()) {
        return Err(RenderError::Cancelled);
    }
    let mut data = vec![0.0f32; total];
    for (pixel_idx, value) in data.iter_mut().enumerate() {
        let mut count = 0u32;
        for lane in 0..lanes {
            let bits = bitsets[pixel_idx * lanes + lane].load(Ordering::Relaxed);
            count += bits.count_ones();
        }
        *value = count as f32 * inv_samples;
    }
    Ok(Plane::from_vec(out_w, out_h, data))
}

fn bounds(center: f32, radius: f32, limit: i32) -> Option<(i32, i32)> {
    let mut min = ((center - radius) - 0.5).ceil() as i32;
    let mut max = ((center + radius) - 0.5).floor() as i32;
    if max < min {
        return None;
    }
    if limit <= 0 {
        return None;
    }
    let last = limit - 1;
    if min > last || max < 0 {
        return None;
    }
    min = min.clamp(0, last);
    max = max.clamp(0, last);
    if min > max { None } else { Some((min, max)) }
}
