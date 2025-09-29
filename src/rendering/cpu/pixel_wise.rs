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
    skip(input_image, options, x_offset_samples, y_offset_samples, lambda_lookup_table),
    fields(output_row = output_row_index, output_col = output_column_index)
)]
fn render_pixel(
    input_image: &Array2<f32>,
    output_row_index: usize,
    output_column_index: usize,
    options: &FilmGrainOptions,
    x_offset_samples: &[f32],
    y_offset_samples: &[f32],
    lambda_lookup_table: &[f32],
) -> f32 {
    // Get input image dimensions.
    let (input_row_count, input_column_count) =
        (input_image.shape()[0] as f32, input_image.shape()[1] as f32);
    let output_width = options.output_width as f32;
    let output_height = options.output_height as f32;

    // Convert output pixel center to input coordinates.
    let input_x_position = options.input_region_min_x
        + (output_column_index as f32 + 0.5)
            * ((options.input_region_max_x - options.input_region_min_x) / output_width);
    let input_y_position = options.input_region_min_y
        + (output_row_index as f32 + 0.5)
            * ((options.input_region_max_y - options.input_region_min_y) / output_height);

    // Scaling factors.
    let scale_x = (options.output_width as f32 - 1.0)
        / (options.input_region_max_x - options.input_region_min_x);
    let scale_y = (options.output_height as f32 - 1.0)
        / (options.input_region_max_y - options.input_region_min_y);

    // Determine cell size.
    let cell_size = 1.0 / ((1.0 / options.grain_radius).ceil());

    // Compute maximum grain radius (for variable radii).
    let max_radius = if options.grain_radius_stddev_factor > 0.0 {
        let sigma = (((options.grain_radius_stddev_factor / options.grain_radius).powi(2) + 1.0)
            .ln())
        .sqrt();
        let sigma_sq = sigma * sigma;
        let mu = options.grain_radius.ln() - sigma_sq / 2.0;
        let normal_quantile = 3.0902;
        (mu + sigma * normal_quantile).exp()
    } else {
        options.grain_radius
    };

    let mut successful_samples = 0;
    // Loop over Monte Carlo iterations.
    for monte_carlo_index in 0..options.monte_carlo_sample_count {
        let x_shifted = input_x_position
            + options.gaussian_filter_stddev * (x_offset_samples[monte_carlo_index] / scale_x);
        let y_shifted = input_y_position
            + options.gaussian_filter_stddev * (y_offset_samples[monte_carlo_index] / scale_y);

        let min_cell_x = ((x_shifted - max_radius) / cell_size).floor() as i32;
        let max_cell_x = ((x_shifted + max_radius) / cell_size).floor() as i32;
        let min_cell_y = ((y_shifted - max_radius) / cell_size).floor() as i32;
        let max_cell_y = ((y_shifted + max_radius) / cell_size).floor() as i32;

        'cell_loop: for cell_x in min_cell_x..=max_cell_x {
            for cell_y in min_cell_y..=max_cell_y {
                let cell_corner_x = cell_size * cell_x as f32;
                let cell_corner_y = cell_size * cell_y as f32;
                let seed = cell_seed(cell_x as u32, cell_y as u32, options.grain_seed_offset);
                let mut cell_rng = Prng::new(seed);

                let sample_x = cell_corner_x.floor().clamp(0.0, input_column_count - 1.0) as usize;
                let sample_y = cell_corner_y.floor().clamp(0.0, input_row_count - 1.0) as usize;
                let normalized_intensity = input_image[[sample_y, sample_x]];

                const MAX_GREY_LEVEL: usize = 255;
                const EPSILON: f32 = 0.1;
                let lookup_index = ((normalized_intensity * (MAX_GREY_LEVEL as f32 + EPSILON))
                    .floor() as usize)
                    .min(MAX_GREY_LEVEL);
                let poisson_rate = lambda_lookup_table[lookup_index];

                let grain_count = if poisson_rate > 0.0 {
                    cell_rng.next_poisson(poisson_rate, None)
                } else {
                    0
                };
                trace!(
                    cell = ?(cell_x, cell_y),
                    grains = grain_count,
                    lambda = poisson_rate,
                    "Sampled grain count for cell"
                );

                for _ in 0..(grain_count as usize) {
                    let x_center = cell_corner_x + cell_size * cell_rng.next_f32();
                    let y_center = cell_corner_y + cell_size * cell_rng.next_f32();

                    let curr_radius = if options.grain_radius_stddev_factor > 0.0 {
                        let sample = cell_rng.next_standard_normal();
                        let sigma = (((options.grain_radius_stddev_factor / options.grain_radius)
                            .powi(2)
                            + 1.0)
                            .ln())
                        .sqrt();
                        let mu = options.grain_radius.ln() - (sigma * sigma) / 2.0;
                        (mu + sigma * sample).exp().min(max_radius)
                    } else {
                        options.grain_radius
                    };

                    if sq_distance(x_center, y_center, x_shifted, y_shifted)
                        < curr_radius * curr_radius
                    {
                        successful_samples += 1;
                        trace!("Monte Carlo sample hit grain");
                        break 'cell_loop;
                    }
                }
            }
        }
    }
    successful_samples as f32 / options.monte_carlo_sample_count as f32
}

/// Render the entire image using the pixel-wise film grain rendering algorithm.
#[instrument(level = "info", skip(input_image, options))]
pub fn render_pixel_wise(input_image: &Array2<f32>, options: &FilmGrainOptions) -> Array2<f32> {
    debug!(?options, "Starting pixel-wise CPU rendering");
    debug!("Precomputing Monte Carlo offsets");
    // Generate Monte Carlo translation offsets using the PRNG.
    let mut prng = Prng::new(options.grain_seed_offset);
    let sample_count = options.monte_carlo_sample_count;
    let mut x_offsets = Vec::with_capacity(sample_count);
    let mut y_offsets = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        // Each offset is sampled from a Gaussian N(0, 1) scaled by sigma_filter.
        x_offsets.push(prng.next_standard_normal());
        y_offsets.push(prng.next_standard_normal());
    }

    // Precompute a lambda lookup table for each grey-level value.
    const MAX_GREY_LEVEL: usize = 255;
    const EPSILON: f32 = 0.1;
    let cell_size = 1.0 / ((1.0 / options.grain_radius).ceil());
    let mut lambda_lookup_table = vec![0.0f32; MAX_GREY_LEVEL + 1];
    debug!("Computing lambda lookup table");
    lambda_lookup_table
        .iter_mut()
        .enumerate()
        .for_each(|(i, lambda)| {
            let u = i as f32 / (MAX_GREY_LEVEL as f32 + EPSILON);
            let denom = PI
                * (options.grain_radius * options.grain_radius
                    + if options.grain_radius_stddev_factor > 0.0 {
                        options.grain_radius_stddev_factor * options.grain_radius_stddev_factor
                    } else {
                        0.0
                    });
            *lambda = -((cell_size * cell_size) / denom) * (1.0 - u).ln();
        });

    // Create the output image and compute each pixel in parallel.
    let mut output_image = Array2::<f32>::zeros((options.output_height, options.output_width));
    debug!(
        rows = options.output_height,
        cols = options.output_width,
        "Rendering output image in parallel"
    );
    output_image
        .indexed_iter_mut()
        .par_bridge()
        .for_each(|((row, col), pixel)| {
            *pixel = render_pixel(
                input_image,
                row,
                col,
                options,
                &x_offsets,
                &y_offsets,
                &lambda_lookup_table,
            );
        });
    debug!("Completed pixel-wise rendering");
    output_image
}
