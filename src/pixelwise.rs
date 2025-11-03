use rand::distributions::{Distribution, Uniform};
use rand_distr::Poisson;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

use crate::model::{Derived, Plane};
use crate::params::Params;
use crate::rng;

pub fn render_pixelwise(lambda: &Plane, params: &Params, derived: &Derived) -> Plane {
    let out_w = derived.output_width;
    let out_h = derived.output_height;
    let mut pixels = vec![0.0f32; out_w * out_h];
    let inv_samples = 1.0 / params.n_samples.max(1) as f32;
    let inv_zoom = 1.0 / params.zoom;
    let offsets_input = &derived.offsets_input;

    pixels
        .par_chunks_mut(out_w)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as usize;
            for (x, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for offset in offsets_input {
                    let xg = ((x as f32 + 0.5) * inv_zoom) - offset[0];
                    let yg = ((y as f32 + 0.5) * inv_zoom) - offset[1];
                    sum += evaluate_indicator(xg, yg, lambda, params, derived);
                }
                *value = sum * inv_samples;
            }
        });

    Plane::from_vec(out_w, out_h, pixels)
}

fn evaluate_indicator(xg: f32, yg: f32, lambda: &Plane, params: &Params, derived: &Derived) -> f32 {
    let rm = derived.rm;
    let delta = derived.delta;

    if rm <= 0.0 {
        return 0.0;
    }

    let i0 = ((xg - rm) / delta).floor() as i32;
    let i1 = ((xg + rm) / delta).floor() as i32;
    let j0 = ((yg - rm) / delta).floor() as i32;
    let j1 = ((yg + rm) / delta).floor() as i32;

    if i0 > i1 || j0 > j1 {
        return 0.0;
    }

    let uniform = Uniform::new(0.0f32, delta);

    for i_delta in i0..=i1 {
        for j_delta in j0..=j1 {
            let mut rng = rng::cell_rng(params.seed, i_delta, j_delta);
            let sample_x = (i_delta as f32) * delta;
            let sample_y = (j_delta as f32) * delta;
            let ix = sample_x.floor() as isize;
            let iy = sample_y.floor() as isize;
            let lambda_cell = lambda.get_clamped(ix, iy);
            if lambda_cell <= 0.0 {
                continue;
            }
            let expected = lambda_cell * delta * delta;
            if expected <= 0.0 {
                continue;
            }
            let poisson = Poisson::new(expected as f64).unwrap();
            let q = poisson.sample(&mut rng) as u32;
            if q == 0 {
                continue;
            }
            for _ in 0..q {
                let cx = sample_x + uniform.sample(&mut rng);
                let cy = sample_y + uniform.sample(&mut rng);
                let mut radius = derived.radius.sample(&mut rng);
                if radius > derived.rm {
                    radius = derived.rm;
                }
                if radius <= 0.0 {
                    continue;
                }
                let dx = xg - cx;
                let dy = yg - cy;
                if dx * dx + dy * dy <= radius * radius {
                    return 1.0;
                }
            }
        }
    }

    0.0
}
