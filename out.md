```
src/
├─ rendering
│   ├─ grain_wise.rs
│   ├─ mod.rs
│   └─ pixel_wise.rs
├─ cli.rs
├─ io.rs
├─ lib.rs
├─ main.rs
└─ prng.rs
```

---

# io.rs

```rust
use image::{GrayImage, Luma};
use ndarray::Array2;
use std::path::Path;

/// Reads a grayscale image from the given file path and normalizes pixel values to [0, 1].
pub fn read_image(path: &str) -> Array2<f32> {
    let img = image::open(path).expect("Failed to open input image");
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let data: Vec<f32> = gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    Array2::from_shape_vec((height as usize, width as usize), data)
        .expect("Error creating image array")
}

/// Writes an image (as an Array2<f32> with values in [0, 1]) to the given file path.
pub fn write_image<P: AsRef<Path>>(path: P, img: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = (img.shape()[0], img.shape()[1]);
    let mut gray_img = GrayImage::new(cols as u32, rows as u32);
    for ((row, col), &val) in img.indexed_iter() {
        let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        gray_img.put_pixel(col as u32, row as u32, Luma([pixel_val]));
    }
    gray_img.save(path).map_err(|e| e.to_string())
}
```

---

# prng.rs

```rust
use std::f32::consts::PI;

/// A simple pseudo–random number generator based on a xorshift algorithm.
/// All random sampling (uniform, Gaussian, Poisson) is done through this PRNG.
#[derive(Debug, Clone)]
pub struct Prng {
    state: u32,
}

impl Prng {
    /// Creates a new PRNG with a given seed (which is first scrambled via the wang–hash).
    pub fn new(seed: u32) -> Self {
        Self {
            state: Self::wang_hash(seed),
        }
    }

    /// Wang hash: a simple hash function to scramble the seed.
    fn wang_hash(seed: u32) -> u32 {
        let mut seed = seed;
        seed = (seed ^ 61) ^ (seed >> 16);
        seed = seed.wrapping_mul(9);
        seed ^= seed >> 4;
        seed = seed.wrapping_mul(668_265_261);
        seed ^= seed >> 15;
        seed
    }

    /// Generates the next random u32 and updates the internal state.
    pub fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    /// Returns a uniform random number in [0, 1].
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4294967295.0
    }

    /// Returns a standard normally distributed random number (mean 0, std dev 1) using Box–Muller.
    pub fn next_standard_normal(&mut self) -> f32 {
        let u = self.next_f32();
        let v = self.next_f32();
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }

    /// Samples a Poisson–distributed random variable with parameter `lambda`.
    /// An optional precomputed exp(–lambda) may be supplied.
    pub fn next_poisson(&mut self, lambda: f32, prod_in: Option<f32>) -> u32 {
        let u = self.next_f32();
        let mut x: u32 = 0;
        let mut prod = match prod_in {
            Some(val) if val > 0.0 => val,
            _ => (-lambda).exp(),
        };
        let mut sum = prod;
        let limit = (10000.0 * lambda).floor() as u32;
        while u > sum && x < limit {
            x += 1;
            prod = prod * lambda / (x as f32);
            sum += prod;
        }
        x
    }
}

/// Computes a unique seed for a given cell based on its (x, y) coordinates and a constant offset.
/// The period is 2^16; if the resulting seed is zero, returns 1.
pub fn cell_seed(x: u32, y: u32, offset: u32) -> u32 {
    const PERIOD: u32 = 65_536;
    let s = ((y % PERIOD) * PERIOD + (x % PERIOD)).wrapping_add(offset);
    if s == 0 {
        1
    } else {
        s
    }
}
```

---

# lib.rs

```rust
//! Realistic Film Grain Rendering
//!
//! This library implements a film grain rendering algorithm based on the physically–motivated
//! model described in Newson et al. (2017). Two implementations are provided – a pixel–wise
//! approach and a grain–wise approach – with an automatic selection mechanism.

pub mod cli;
pub mod io;
mod prng;
pub mod rendering;
```

---

# main.rs

```rust
use clap::Parser;
use film_grain::cli::Cli;
use film_grain::io::{read_image, write_image};
use film_grain::rendering::grain_wise::render_grain_wise;
use film_grain::rendering::pixel_wise::render_pixel_wise;
use film_grain::rendering::{FilmGrainOptions, RenderingAlgorithm};

fn main() {
    // Parse command line arguments.
    let args = Cli::parse();

    // Read input image.
    let image_in = read_image(&args.input);

    // Determine default region and output dimensions if not provided.
    let (img_rows, img_cols) = (image_in.shape()[0] as f32, image_in.shape()[1] as f32);
    let x_b = args.x_b.unwrap_or(img_cols);
    let y_b = args.y_b.unwrap_or(img_rows);
    let m_out = args.height.unwrap_or((img_rows * args.zoom) as u32) as usize;
    let n_out = args.width.unwrap_or((img_cols * args.zoom) as u32) as usize;

    // Set up rendering options.
    let opts = FilmGrainOptions {
        grain_radius: args.mu_r,
        sigma_r: args.sigma_r,
        sigma_filter: args.sigma_filter,
        n_monte_carlo: args.n_monte_carlo as usize,
        grain_seed: args.seed,
        x_a: args.x_a,
        y_a: args.y_a,
        x_b,
        y_b,
        m_out,
        n_out,
    };

    let start_time = std::time::Instant::now();
    // Choose and run the rendering algorithm.
    let image_out = match args.algorithm {
        RenderingAlgorithm::GrainWise => render_grain_wise(&image_in, &opts),
        RenderingAlgorithm::PixelWise => render_pixel_wise(&image_in, &opts),
        RenderingAlgorithm::Automatic => {
            // Use grain-wise if grain radius is large; otherwise, pixel-wise.
            if opts.grain_radius > 1.0 {
                render_grain_wise(&image_in, &opts)
            } else {
                render_pixel_wise(&image_in, &opts)
            }
        }
    };
    let elapsed_time = start_time.elapsed();
    println!("Rendering took {} ms", elapsed_time.as_millis());

    // Write the rendered output image.
    write_image(&args.output, &image_out).expect("Failed to write output image");
}
```

---

# rendering/pixel_wise.rs

```rust
use crate::prng::{cell_seed, Prng};
use crate::rendering::{sq_distance, FilmGrainOptions};
use ndarray::Array2;
use rayon::prelude::*;
use std::f32::consts::PI;

/// Renders a single output pixel using the pixel-wise algorithm.
/// This function uses precomputed Monte Carlo offsets and a lambda lookup table.
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
                        break 'cell_loop;
                    }
                }
            }
        }
    }
    success_count as f32 / opts.n_monte_carlo as f32
}

/// Render the entire image using the pixel-wise film grain rendering algorithm.
pub fn render_pixel_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
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
    img_out
}
```

---

# rendering/mod.rs

```rust
pub mod grain_wise;
pub mod pixel_wise;

use clap::ValueEnum;

/// Specifies which rendering algorithm to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum RenderingAlgorithm {
    GrainWise,
    PixelWise,
    Automatic,
}

/// Options for film grain rendering. These parameters correspond to the model in Newson et al. (2017).
#[derive(Debug, Clone)]
pub struct FilmGrainOptions {
    /// Average grain radius (µ_r) in input image units.
    pub grain_radius: f32,
    /// Standard deviation of the grain radius (σ_r) as a fraction of µ_r.
    pub sigma_r: f32,
    /// Standard deviation of the Gaussian filter (σ_filter) in output pixels.
    pub sigma_filter: f32,
    /// Number of Monte Carlo iterations.
    pub n_monte_carlo: usize,
    /// Seed offset used for seeding cell–specific PRNGs.
    pub grain_seed: u32,
    /// Rendering region in the input image (top-left: (x_a, y_a), bottom–right: (x_b, y_b)).
    pub x_a: f32,
    pub y_a: f32,
    pub x_b: f32,
    pub y_b: f32,
    /// Output image dimensions (rows: m_out, columns: n_out).
    pub m_out: usize,
    pub n_out: usize,
}

/// Returns the squared Euclidean distance between (x1, y1) and (x2, y2).
#[inline]
pub fn sq_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (x1 - x2).powi(2) + (y1 - y2).powi(2)
}
```

---

# rendering/grain_wise.rs

```rust
use crate::prng::{cell_seed, Prng};
use crate::rendering::FilmGrainOptions;
use ndarray::Array2;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Optimized grain–wise film grain rendering.
///
/// This version parallelizes over input pixels (rows) rather than over Monte Carlo iterations,
/// and uses shared, atomically updated bit–maps (as flat Vec<AtomicBool>) for each Monte Carlo sample.
/// This minimizes allocation and scheduling overhead while maintaining the original logic.
pub fn render_grain_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    // Input image dimensions.
    let m_in = img_in.shape()[0] as i32;
    let n_in = img_in.shape()[1] as i32;

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
    let monte_carlo_offsets: Vec<(f32, f32)> = (0..n_mc)
        .map(|_| (prng.next_standard_normal(), prng.next_standard_normal()))
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
        .map(|_| Arc::new((0..num_pixels).map(|_| AtomicBool::new(false)).collect()))
        .collect();

    // Parallelize over the input rows.
    (y_start..y_end).into_par_iter().for_each(|i| {
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
    output
}
```

---

# cli.rs

```rust
use crate::rendering::RenderingAlgorithm;
use clap::Parser;

/// Film Grain Rendering based on Newson et al. (2017)
#[derive(Debug, Parser)]
#[clap(author, version, about = "Apply a realistic film grain rendering to an input image", long_about = None)]
pub struct Cli {
    /// Input image file path
    pub input: String,

    /// Output image file path
    pub output: String,

    /// Average grain radius (µ_r) in input image units
    #[clap(short = 'r', long = "radius", default_value_t = 0.1)]
    pub mu_r: f32,

    /// Grain standard deviation factor (σ_r, as a fraction of µ_r)
    #[clap(long = "sigmaR", default_value_t = 0.0)]
    pub sigma_r: f32,

    /// Standard deviation of the Gaussian filter (σ_filter) in output pixels
    #[clap(long = "sigmaFilter", default_value_t = 0.8)]
    pub sigma_filter: f32,

    /// Zoom factor for the output image
    #[clap(long = "zoom", default_value_t = 1.0)]
    pub zoom: f32,

    /// Rendering algorithm to use: grain-wise, pixel-wise, or automatic
    #[clap(long = "algorithm", default_value = "automatic")]
    pub algorithm: RenderingAlgorithm,

    /// Number of Monte Carlo iterations
    #[clap(long = "NmonteCarlo", default_value_t = 800)]
    pub n_monte_carlo: u32,

    /// x-coordinate of the top-left corner of the rendering region (in input image coordinates)
    #[clap(long = "xA", default_value_t = 0.0)]
    pub x_a: f32,

    /// y-coordinate of the top-left corner of the rendering region (in input image coordinates)
    #[clap(long = "yA", default_value_t = 0.0)]
    pub y_a: f32,

    /// x-coordinate of the bottom-right corner of the rendering region (if omitted, defaults to image width)
    #[clap(long = "xB")]
    pub x_b: Option<f32>,

    /// y-coordinate of the bottom-right corner of the rendering region (if omitted, defaults to image height)
    #[clap(long = "yB")]
    pub y_b: Option<f32>,

    /// Output image width (number of columns; if not provided, computed from zoom)
    #[clap(long = "width")]
    pub width: Option<u32>,

    /// Output image height (number of rows; if not provided, computed from zoom)
    #[clap(long = "height")]
    pub height: Option<u32>,

    /// Seed for the pseudo–random number generator used throughout the algorithm
    #[clap(long = "seed", default_value_t = 42)]
    pub seed: u32,
}
```

