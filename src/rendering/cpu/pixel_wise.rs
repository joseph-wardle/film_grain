use crate::prng::{cell_seed, Prng};
use crate::rendering::{sq_distance, FilmGrainOptions};
use ndarray::Array2;
use rayon::prelude::*;
use std::f32::consts::PI;
use tracing::{debug, instrument, trace};

/// Renders a single output pixel using the pixel-wise algorithm.
/// This function uses precomputed Monte Carlo offsets and a lambda lookup table.
#[instrument(
    level = "trace",
    skip(img_in, opts, x_offset_list, y_offset_list, lambda_lookup),
    fields(output_row = y_out, output_col = x_out)
)]
fn render_pixel(
    img_in: &Array2<f32>,
    y_out: usize,
    x_out: usize,
    opts: &FilmGrainOptions,
    x_offset_list: &[f32],
    y_offset_list: &[f32],
    lambda_lookup: &[f32],
) -> f32 {
    // Get input image dimensions.
    let (m_in, n_in) = (img_in.shape()[0] as f32, img_in.shape()[1] as f32);
    let n_out = opts.n_out as f32;
    let m_out = opts.m_out as f32;

    // Convert output pixel center to input coordinates.
    let x_in = opts.x_a + (x_out as f32 + 0.5) * ((opts.x_b - opts.x_a) / n_out);
    let y_in = opts.y_a + (y_out as f32 + 0.5) * ((opts.y_b - opts.y_a) / m_out);

    // Scaling factors.
    let s_x = (opts.n_out as f32 - 1.0) / (opts.x_b - opts.x_a);
    let s_y = (opts.m_out as f32 - 1.0) / (opts.y_b - opts.y_a);

    // Determine cell size.
    let cell_size = 1.0 / ((1.0 / opts.grain_radius).ceil());

    // Compute maximum grain radius (for variable radii).
    let max_radius = if opts.sigma_r > 0.0 {
        let sigma = (((opts.sigma_r / opts.grain_radius).powi(2) + 1.0).ln()).sqrt();
        let sigma_sq = sigma * sigma;
        let mu = opts.grain_radius.ln() - sigma_sq / 2.0;
        let normal_quantile = 3.0902;
        (mu + sigma * normal_quantile).exp()
    } else {
        opts.grain_radius
    };

    let mut success_count = 0;
    // Loop over Monte Carlo iterations.
    for i in 0..opts.n_monte_carlo {
        let x_shifted = x_in + opts.sigma_filter * (x_offset_list[i] / s_x);
        let y_shifted = y_in + opts.sigma_filter * (y_offset_list[i] / s_y);

        let min_cell_x = ((x_shifted - max_radius) / cell_size).floor() as i32;
        let max_cell_x = ((x_shifted + max_radius) / cell_size).floor() as i32;
        let min_cell_y = ((y_shifted - max_radius) / cell_size).floor() as i32;
        let max_cell_y = ((y_shifted + max_radius) / cell_size).floor() as i32;

        'cell_loop: for cell_x in min_cell_x..=max_cell_x {
            for cell_y in min_cell_y..=max_cell_y {
                let cell_corner_x = cell_size * cell_x as f32;
                let cell_corner_y = cell_size * cell_y as f32;
                let seed = cell_seed(cell_x as u32, cell_y as u32, opts.grain_seed);
                let mut cell_rng = Prng::new(seed);

                let sample_x = cell_corner_x.floor().clamp(0.0, n_in - 1.0) as usize;
                let sample_y = cell_corner_y.floor().clamp(0.0, m_in - 1.0) as usize;
                let u = img_in[[sample_y, sample_x]];

                const MAX_GREY_LEVEL: usize = 255;
                const EPSILON: f32 = 0.1;
                let u_index =
                    ((u * (MAX_GREY_LEVEL as f32 + EPSILON)).floor() as usize).min(MAX_GREY_LEVEL);
                let curr_lambda = lambda_lookup[u_index];

                let n_cell = if curr_lambda > 0.0 {
                    cell_rng.next_poisson(curr_lambda, None)
                } else {
                    0
                };
                trace!(
                    cell = ?(cell_x, cell_y),
                    grains = n_cell,
                    lambda = curr_lambda,
                    "Sampled grain count for cell"
                );

                for _ in 0..(n_cell as usize) {
                    let x_center = cell_corner_x + cell_size * cell_rng.next_f32();
                    let y_center = cell_corner_y + cell_size * cell_rng.next_f32();

                    let curr_radius = if opts.sigma_r > 0.0 {
                        let sample = cell_rng.next_standard_normal();
                        let sigma =
                            (((opts.sigma_r / opts.grain_radius).powi(2) + 1.0).ln()).sqrt();
                        let mu = opts.grain_radius.ln() - (sigma * sigma) / 2.0;
                        (mu + sigma * sample).exp().min(max_radius)
                    } else {
                        opts.grain_radius
                    };

                    if sq_distance(x_center, y_center, x_shifted, y_shifted)
                        < curr_radius * curr_radius
                    {
                        success_count += 1;
                        trace!("Monte Carlo sample hit grain");
                        break 'cell_loop;
                    }
                }
            }
        }
    }
    success_count as f32 / opts.n_monte_carlo as f32
}

/// Render the entire image using the pixel-wise film grain rendering algorithm.
#[instrument(level = "info", skip(img_in, opts))]
pub fn render_pixel_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    debug!(?opts, "Starting pixel-wise CPU rendering");
    debug!("Precomputing Monte Carlo offsets");
    // Generate Monte Carlo translation offsets using the PRNG.
    let mut prng = Prng::new(opts.grain_seed);
    let n_mc = opts.n_monte_carlo;
    let mut x_offsets = Vec::with_capacity(n_mc);
    let mut y_offsets = Vec::with_capacity(n_mc);
    for _ in 0..n_mc {
        // Each offset is sampled from a Gaussian N(0, 1) scaled by sigma_filter.
        x_offsets.push(prng.next_standard_normal());
        y_offsets.push(prng.next_standard_normal());
    }

    // Precompute a lambda lookup table for each grey-level value.
    const MAX_GREY_LEVEL: usize = 255;
    const EPSILON: f32 = 0.1;
    let cell_size = 1.0 / ((1.0 / opts.grain_radius).ceil());
    let mut lambda_lookup = vec![0.0f32; MAX_GREY_LEVEL + 1];
    debug!("Computing lambda lookup table");
    lambda_lookup
        .iter_mut()
        .enumerate()
        .for_each(|(i, lambda)| {
            let u = i as f32 / (MAX_GREY_LEVEL as f32 + EPSILON);
            let denom = PI
                * (opts.grain_radius * opts.grain_radius
                    + if opts.sigma_r > 0.0 {
                        opts.sigma_r * opts.sigma_r
                    } else {
                        0.0
                    });
            *lambda = -((cell_size * cell_size) / denom) * (1.0 - u).ln();
        });

    // Create the output image and compute each pixel in parallel.
    let mut img_out = Array2::<f32>::zeros((opts.m_out, opts.n_out));
    debug!(
        rows = opts.m_out,
        cols = opts.n_out,
        "Rendering output image in parallel"
    );
    img_out
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((row, col), pixel)| {
            *pixel = render_pixel(
                img_in,
                row,
                col,
                opts,
                &x_offsets,
                &y_offsets,
                &lambda_lookup,
            );
        });
    debug!("Completed pixel-wise rendering");
    img_out
}
