use crate::{
    config::*,
    error::*,
    rng::{SplitMix64, gaussian_pair, poisson_inverse},
};
use rayon::prelude::*;
use tracing::instrument;
use crate::util::hash3_u64;

/// Evaluate v(y) = (1/N) Σ 1_Z((y - ξ_k)/s) for each output pixel y.
/// Locally generate Poisson grains for overlapping cells, using hashed cell seeds.
#[instrument(skip(gray01, params, lambda_lut, exp_lut))]
pub fn render_pixelwise_cpu(
    gray01: &[f32],
    input_width: u32,
    input_height: u32,
    params: &FilmGrainParams,
    lambda_lut: &[f32],
    exp_lut: &[f32],
    _threads: Option<usize>,
) -> Result<Vec<f32>> {
    let FilmGrainParams {
        grain_size,
        grain_size_std_dev,
        blur_sigma,
        n_monte_carlo,
        seed,
        ..
    } = *params;

    let output_width = params.output_width;
    let output_height = params.output_height;

    let roi_span_x: f32 = (params.x_end as f32) - (params.x_start as f32);
    let roi_span_y: f32 = (params.y_end as f32) - (params.y_start as f32);
    let scale_x = (output_width - 1) as f32 / roi_span_x;
    let scale_y = (output_height - 1) as f32 / roi_span_y;

    // Precompute Gaussian offsets (same for every output pixel)
    let mut rng = SplitMix64::new(seed);
    let mut mc_offset_x = vec![0.0f32; n_monte_carlo as usize];
    let mut mc_offset_y = vec![0.0f32; n_monte_carlo as usize];
    for mc_index in (0..n_monte_carlo as usize).step_by(2) {
        let (g0, g1) = gaussian_pair(&mut rng);
        mc_offset_x[mc_index] = g0 * blur_sigma;
        mc_offset_y[mc_index] = g1 * blur_sigma;
        if mc_index + 1 < mc_offset_x.len() {
            let (g2, g3) = gaussian_pair(&mut rng);
            mc_offset_x[mc_index + 1] = g2 * blur_sigma;
            mc_offset_y[mc_index + 1] = g3 * blur_sigma;
        }
    }

    let cell_size = 1.0 / (1.0 / grain_size).ceil().max(1.0); // cell size
    let max_grain_radius = if grain_size_std_dev > 0.0 {
        // 99.9% quantile of log-normal: exp(mu + sigma*3.0902)
        let sigma2 = ((grain_size_std_dev / grain_size).powi(2) + 1.0).ln();
        let sigma = sigma2.sqrt();
        let mu = grain_size.ln() - 0.5 * sigma2;
        (mu + sigma * 3.0902).exp()
    } else {
        grain_size
    };

    // Output buffer
    let mut out = vec![0.0f32; (output_height * output_width) as usize];

    // Parallel over output pixels
    out.par_chunks_mut(output_width as usize).enumerate().for_each(|(out_row_index, out_row_pixels)| {
        let out_y = out_row_index as i32;
        for out_x in 0..output_width {
            let out_x_i32 = out_x as i32;
            let input_sample_x: f32 = params.x_start as f32
                + (out_x_i32 as f32 + 0.5) * (roi_span_x / output_width as f32);
            let input_sample_y: f32 = params.y_start as f32
                + (out_y as f32 + 0.5) * (roi_span_y / output_height as f32);
            let mut covered_count = 0u32;

            // For each Monte Carlo sample
            for mc_idx in 0..(n_monte_carlo as usize) {
                let shifted_x = input_sample_x + mc_offset_x[mc_idx] / scale_x;
                let shifted_y = input_sample_y + mc_offset_y[mc_idx] / scale_y;

                // Bounding cells
                let min_x = ((shifted_x - max_grain_radius) / cell_size).floor() as i32;
                let max_x = ((shifted_x + max_grain_radius) / cell_size).floor() as i32;
                let min_y = ((shifted_y - max_grain_radius) / cell_size).floor() as i32;
                let max_y = ((shifted_y + max_grain_radius) / cell_size).floor() as i32;

                'scan_cells: for cell_y in min_y..=max_y {
                    for cell_x in min_x..=max_x {
                        // Cell corner in input pixel coords
                        let cell_origin_x = (cell_x as f32) * cell_size;
                        let cell_origin_y = (cell_y as f32) * cell_size;

                        // Sample u from *nearest* input pixel (piecewise constant λ)
                        let nearest_px = cell_origin_x.floor().clamp(0.0, (input_width - 1) as f32) as usize;
                        let nearest_py = cell_origin_y.floor().clamp(0.0, (input_height - 1) as f32) as usize;
                        let local_intensity = gray01[nearest_py * input_width as usize + nearest_px];
                        let grey_lut_idx = (local_intensity * (MAX_GREY_LEVEL as f32 + EPSILON_GREY_LEVEL))
                            .floor()
                            .clamp(0.0, MAX_GREY_LEVEL as f32)
                            as usize;
                        let lambda = lambda_lut[grey_lut_idx];
                        if lambda <= 0.0 {
                            continue;
                        }
                        let exp_lambda = exp_lut[grey_lut_idx];

                        // Cell-seeded RNG (reproducible)
                        let mut cell_rng = SplitMix64::new(hash3_u64(
                            cell_x as u32 as u64,
                            cell_y as u32 as u64,
                            params.seed,
                        ));

                        let grains_in_cell = poisson_inverse(
                            &mut cell_rng,
                            lambda,
                            Some(exp_lambda),
                        );
                        for _ in 0..grains_in_cell {
                            // centre
                            let grain_center_x = cell_origin_x + cell_rng.next_f32() * cell_size;
                            let grain_center_y = cell_origin_y + cell_rng.next_f32() * cell_size;

                            // radius
                            let radius2 = if params.grain_size_std_dev > 0.0 {
                                let sigma2 = ((params.grain_size_std_dev / params.grain_size).powi(2) + 1.0).ln();
                                let sigma = sigma2.sqrt();
                                let mu = params.grain_size.ln() - 0.5 * sigma2;
                                let z = {
                                    let (z0, _) = gaussian_pair(&mut cell_rng);
                                    z0
                                };
                                let radius = (mu + sigma * z).exp().min(max_grain_radius);
                                radius * radius
                            } else {
                                params.grain_size * params.grain_size
                            };

                            // test coverage of (xg_i, yg_i)
                            let dx = grain_center_x - shifted_x;
                            let dy = grain_center_y - shifted_y;
                            if dx * dx + dy * dy < radius2 {
                                covered_count += 1;
                                break 'scan_cells;
                            }
                        }
                    }
                }
            }

            out_row_pixels[out_x as usize] = (covered_count as f32) / (n_monte_carlo as f32);
        }
    });

    Ok(out)
}
