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
#[instrument(level = "info", skip(img_in, opts))]
pub fn render_grain_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    // Input image dimensions.
    let m_in = img_in.shape()[0] as i32;
    let n_in = img_in.shape()[1] as i32;
    debug!(m_in, n_in, ?opts, "Starting grain-wise CPU rendering");

    // Output image dimensions.
    let m_out = opts.m_out;
    let n_out = opts.n_out;
    let num_pixels = m_out * n_out;

    // Scaling factors from input to output.
    let s_x = (n_out as f32 - 1.0) / (opts.x_b - opts.x_a);
    let s_y = (m_out as f32 - 1.0) / (opts.y_b - opts.y_a);

    // Precompute Monte Carlo offset vectors using our PRNG.
    let mut prng = Prng::new(opts.grain_seed);
    let n_mc = opts.n_monte_carlo;
    debug!(n_mc, "Generating Monte Carlo offsets");
    let monte_carlo_offsets: Vec<(f32, f32)> = (0..n_mc)
        .map(|iteration| (prng.next_standard_normal(), prng.next_standard_normal()))
        .collect();

    // Compute parameters for the grain radius distribution.
    let (sigma, mu, max_radius) = if opts.sigma_r > 0.0 {
        let sigma = (((opts.sigma_r / opts.grain_radius).powi(2) + 1.0).ln()).sqrt();
        let sigma_sq = sigma * sigma;
        let mu = opts.grain_radius.ln() - sigma_sq / 2.0;
        let normal_quantile = 3.0902; // for α = 0.999
        (sigma, mu, (mu + sigma * normal_quantile).exp())
    } else {
        (0.0, 0.0, opts.grain_radius)
    };

    // Precompute the input region boundaries.
    let y_start = opts.y_a.floor() as i32;
    let y_end = opts.y_b.ceil() as i32;
    let x_start = opts.x_a.floor() as i32;
    let x_end = opts.x_b.ceil() as i32;

    // Allocate one shared atomic boolean "image" per Monte Carlo iteration.
    // Each image is represented as a flat vector of AtomicBool (length = m_out * n_out).
    let mc_atomic_images: Vec<Arc<Vec<AtomicBool>>> = (0..n_mc)
        .map(|iteration| {
            trace!(
                iteration,
                "Allocating atomic image for Monte Carlo iteration"
            );
            Arc::new((0..num_pixels).map(|_| AtomicBool::new(false)).collect())
        })
        .collect();

    // Parallelize over the input rows.
    (y_start..y_end).into_par_iter().for_each(|i| {
        trace!(row = i, "Processing input row");
        for j in x_start..x_end {
            // Retrieve the normalized pixel value; if out-of-bounds, assume 0.
            let u = if i >= 0 && i < m_in && j >= 0 && j < n_in {
                img_in[[i as usize, j as usize]]
            } else {
                0.0
            };

            // Compute local grain density λ using: λ = 1/(π*(r² + σ_r²)) * ln(1/(1 - u))
            let grain_var = if opts.sigma_r > 0.0 {
                opts.sigma_r * opts.sigma_r
            } else {
                0.0
            };
            let denom = PI * (opts.grain_radius * opts.grain_radius + grain_var);
            let lambda = if u < 1.0 {
                1.0 / denom * (1.0 / (1.0 - u)).ln()
            } else {
                0.0
            };

            // Create a local PRNG seeded by the cell coordinates.
            let seed = cell_seed(j as u32, i as u32, opts.grain_seed);
            let mut local_rng = Prng::new(seed);
            // Sample the number of grains in this input pixel.
            let num_grains = if lambda > 0.0 {
                local_rng.next_poisson(lambda, None)
            } else {
                0
            };
            trace!(
                row = i,
                col = j,
                num_grains,
                lambda,
                "Sampled grains for input pixel"
            );

            for _ in 0..(num_grains as usize) {
                // Sample grain center uniformly within the pixel.
                let x_center = j as f32 + local_rng.next_f32();
                let y_center = i as f32 + local_rng.next_f32();
                // Sample grain radius (either constant or via lognormal distribution).
                let radius = if opts.sigma_r > 0.0 {
                    let sample = local_rng.next_standard_normal();
                    (mu + sigma * sample).exp().min(max_radius)
                } else {
                    opts.grain_radius
                };

                // For each Monte Carlo iteration, update the corresponding atomic image.
                for (k, atomic_image) in mc_atomic_images.iter().enumerate() {
                    let (offset_x, offset_y) = monte_carlo_offsets[k];
                    // Apply the Gaussian offset.
                    let x_center_shifted = x_center - (offset_x / s_x);
                    let y_center_shifted = y_center - (offset_y / s_y);
                    // Project to output coordinates.
                    let x_proj = (x_center_shifted - opts.x_a) * s_x;
                    let y_proj = (y_center_shifted - opts.y_a) * s_y;
                    let r_proj = radius * s_x; // assuming s_x ≃ s_y

                    // Compute bounding box in output grid.
                    let min_x = (x_proj - r_proj).ceil() as i32;
                    let max_x = (x_proj + r_proj).floor() as i32;
                    let min_y = (y_proj - r_proj).ceil() as i32;
                    let max_y = (y_proj + r_proj).floor() as i32;

                    // For each output pixel in the bounding box, check if it lies within the grain circle.
                    for out_y in min_y..=max_y {
                        if out_y < 0 || out_y >= m_out as i32 {
                            continue;
                        }
                        for out_x in min_x..=max_x {
                            if out_x < 0 || out_x >= n_out as i32 {
                                continue;
                            }
                            let pixel_x = out_x as f32 + 0.5;
                            let pixel_y = out_y as f32 + 0.5;
                            let dx = pixel_x - x_proj;
                            let dy = pixel_y - y_proj;
                            if dx * dx + dy * dy <= r_proj * r_proj {
                                // Compute flat index.
                                let index = (out_y as usize) * n_out + (out_x as usize);
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
    let mut output = Array2::<f32>::zeros((m_out, n_out));
    debug!("Aggregating Monte Carlo results");
    for atomic_image in mc_atomic_images.iter() {
        for (idx, flag) in atomic_image.iter().enumerate() {
            if flag.load(Ordering::Relaxed) {
                let row = idx / n_out;
                let col = idx % n_out;
                output[[row, col]] += 1.0;
            }
        }
    }
    // Normalize by the number of Monte Carlo iterations.
    output.mapv_inplace(|v| v / n_mc as f32);
    debug!("Completed grain-wise rendering");
    output
}
