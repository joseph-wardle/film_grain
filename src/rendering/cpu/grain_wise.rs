use crate::prng::{cell_seed, Prng};
use crate::rendering::FilmGrainOptions;
use ndarray::Array2;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument, trace};

/// Optimized grain–wise film grain rendering.
///
/// This version parallelizes over input pixels (rows) rather than over Monte Carlo iterations,
/// and uses shared, atomically updated bit–maps (as flat Vec<AtomicBool>) for each Monte Carlo sample.
/// This minimizes allocation and scheduling overhead while maintaining the original logic.
#[instrument(level = "info", skip(input_image, options))]
pub fn render_grain_wise(input_image: &Array2<f32>, options: &FilmGrainOptions) -> Array2<f32> {
    // Input image dimensions.
    let input_height = input_image.shape()[0] as i32;
    let input_width = input_image.shape()[1] as i32;
    debug!(
        input_height,
        input_width,
        ?options,
        "Starting grain-wise CPU rendering"
    );

    // Output image dimensions.
    let output_height = options.output_height;
    let output_width = options.output_width;
    let output_pixel_count = output_height * output_width;

    // Scaling factors from input to output.
    let scale_x =
        (output_width as f32 - 1.0) / (options.input_region_max_x - options.input_region_min_x);
    let scale_y =
        (output_height as f32 - 1.0) / (options.input_region_max_y - options.input_region_min_y);

    // Precompute Monte Carlo offset vectors using our PRNG.
    let mut prng = Prng::new(options.grain_seed_offset);
    let monte_carlo_sample_count = options.monte_carlo_sample_count;
    debug!(monte_carlo_sample_count, "Generating Monte Carlo offsets");
    let monte_carlo_offsets: Vec<(f32, f32)> = (0..monte_carlo_sample_count)
        .map(|_| (prng.next_standard_normal(), prng.next_standard_normal()))
        .collect();

    // Compute parameters for the grain radius distribution.
    let (sigma, mu, max_radius) = if options.grain_radius_stddev_factor > 0.0 {
        let sigma = ((options.grain_radius_stddev_factor / options.grain_radius).powi(2) + 1.0)
            .ln()
            .sqrt();
        let sigma_sq = sigma * sigma;
        let mu = options.grain_radius.ln() - sigma_sq / 2.0;
        let normal_quantile = 3.0902; // for α = 0.999
        (sigma, mu, (mu + sigma * normal_quantile).exp())
    } else {
        (0.0, 0.0, options.grain_radius)
    };

    // Precompute the input region boundaries.
    let y_start = options.input_region_min_y.floor() as i32;
    let y_end = options.input_region_max_y.ceil() as i32;
    let x_start = options.input_region_min_x.floor() as i32;
    let x_end = options.input_region_max_x.ceil() as i32;

    // Allocate one shared atomic boolean "image" per Monte Carlo iteration.
    // Each image is represented as a flat vector of AtomicBool.
    let monte_carlo_atomic_images: Vec<Arc<Vec<AtomicBool>>> = (0..monte_carlo_sample_count)
        .map(|iteration| {
            trace!(
                iteration,
                "Allocating atomic image for Monte Carlo iteration"
            );
            Arc::new(
                (0..output_pixel_count)
                    .map(|_| AtomicBool::new(false))
                    .collect(),
            )
        })
        .collect();

    // Parallelize over the input rows.
    (y_start..y_end)
        .into_par_iter()
        .for_each(|input_row_index| {
            trace!(row = input_row_index, "Processing input row");
            for input_column_index in x_start..x_end {
                // Retrieve the normalized pixel value; if out-of-bounds, assume 0.
                let normalized_intensity = if input_row_index >= 0
                    && input_row_index < input_height
                    && input_column_index >= 0
                    && input_column_index < input_width
                {
                    input_image[[input_row_index as usize, input_column_index as usize]]
                } else {
                    0.0
                };

                // Compute local grain density λ using: λ = 1/(π*(r² + σ_r²)) * ln(1/(1 - u))
                let grain_variance = if options.grain_radius_stddev_factor > 0.0 {
                    options.grain_radius_stddev_factor * options.grain_radius_stddev_factor
                } else {
                    0.0
                };
                let denom = PI * (options.grain_radius * options.grain_radius + grain_variance);
                let poisson_rate = if normalized_intensity < 1.0 {
                    1.0 / denom * (1.0 / (1.0 - normalized_intensity)).ln()
                } else {
                    0.0
                };

                // Create a local PRNG seeded by the cell coordinates.
                let seed = cell_seed(
                    input_column_index as u32,
                    input_row_index as u32,
                    options.grain_seed_offset,
                );
                let mut cell_rng = Prng::new(seed);
                // Sample the number of grains in this input pixel.
                let grain_count = if poisson_rate > 0.0 {
                    cell_rng.next_poisson(poisson_rate, None)
                } else {
                    0
                };
                trace!(
                    row = input_row_index,
                    col = input_column_index,
                    grain_count,
                    poisson_rate,
                    "Sampled grains for input pixel"
                );

                for _ in 0..(grain_count as usize) {
                    // Sample grain center uniformly within the pixel.
                    let x_center = input_column_index as f32 + cell_rng.next_f32();
                    let y_center = input_row_index as f32 + cell_rng.next_f32();
                    // Sample grain radius (either constant or via lognormal distribution).
                    let radius = if options.grain_radius_stddev_factor > 0.0 {
                        let sample = cell_rng.next_standard_normal();
                        (mu + sigma * sample).exp().min(max_radius)
                    } else {
                        options.grain_radius
                    };

                    // For each Monte Carlo iteration, update the corresponding atomic image.
                    for (monte_carlo_index, atomic_image) in
                        monte_carlo_atomic_images.iter().enumerate()
                    {
                        let (offset_x, offset_y) = monte_carlo_offsets[monte_carlo_index];
                        // Apply the Gaussian offset.
                        let x_center_shifted = x_center - (offset_x / scale_x);
                        let y_center_shifted = y_center - (offset_y / scale_y);
                        // Project to output coordinates.
                        let x_proj = (x_center_shifted - options.input_region_min_x) * scale_x;
                        let y_proj = (y_center_shifted - options.input_region_min_y) * scale_y;
                        let radius_proj = radius * scale_x; // assuming s_x ≃ s_y

                        // Compute bounding box in output grid.
                        let min_x = (x_proj - radius_proj).ceil() as i32;
                        let max_x = (x_proj + radius_proj).floor() as i32;
                        let min_y = (y_proj - radius_proj).ceil() as i32;
                        let max_y = (y_proj + radius_proj).floor() as i32;

                        // For each output pixel in the bounding box, check if it lies within the grain circle.
                        for output_row_index in min_y..=max_y {
                            if output_row_index < 0 || output_row_index >= output_height as i32 {
                                continue;
                            }
                            for output_column_index in min_x..=max_x {
                                if output_column_index < 0
                                    || output_column_index >= output_width as i32
                                {
                                    continue;
                                }
                                let pixel_x = output_column_index as f32 + 0.5;
                                let pixel_y = output_row_index as f32 + 0.5;
                                let dx = pixel_x - x_proj;
                                let dy = pixel_y - y_proj;
                                if dx * dx + dy * dy <= radius_proj * radius_proj {
                                    // Compute flat index.
                                    let index = (output_row_index as usize) * output_width
                                        + (output_column_index as usize);
                                    // Set the atomic flag to true.
                                    atomic_image[index].store(true, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
        });

    // Aggregate the results from each Monte Carlo iteration into the final output image.
    let mut output = Array2::<f32>::zeros((output_height, output_width));
    debug!("Aggregating Monte Carlo results");
    for atomic_image in monte_carlo_atomic_images.iter() {
        for (idx, flag) in atomic_image.iter().enumerate() {
            if flag.load(Ordering::Relaxed) {
                let row = idx / output_width;
                let col = idx % output_width;
                output[[row, col]] += 1.0;
            }
        }
    }
    // Normalize by the number of Monte Carlo iterations.
    output.mapv_inplace(|v| v / monte_carlo_sample_count as f32);
    debug!("Completed grain-wise rendering");
    output
}
