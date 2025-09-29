pub mod cpu;
pub mod gpu;

use clap::ValueEnum;

/// Specifies which rendering algorithm to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum RenderingAlgorithm {
    GrainWise,
    PixelWise,
    GpuPixelWise,
    GpuGrainWise,
    Automatic,
}

/// Options for film grain rendering. These parameters correspond to the model in Newson et al. (2017).
#[derive(Debug, Clone)]
pub struct FilmGrainOptions {
    /// Average grain radius (µ_r) in input image units.
    pub grain_radius: f32,
    /// Standard deviation of the grain radius (σ_r) as a fraction of µ_r.
    pub grain_radius_stddev_factor: f32,
    /// Standard deviation of the Gaussian filter (σ_filter) in output pixels.
    pub gaussian_filter_stddev: f32,
    /// Number of Monte Carlo iterations.
    pub monte_carlo_sample_count: usize,
    /// Seed offset used for seeding cell–specific PRNGs.
    pub grain_seed_offset: u32,
    /// Rendering region in the input image (top-left: (x_a, y_a), bottom–right: (x_b, y_b)).
    pub input_region_min_x: f32,
    pub input_region_min_y: f32,
    pub input_region_max_x: f32,
    pub input_region_max_y: f32,
    /// Output image dimensions (rows: m_out, columns: n_out).
    pub output_height: usize,
    pub output_width: usize,
}

/// Returns the squared Euclidean distance between (x1, y1) and (x2, y2).
#[inline]
pub fn sq_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (x1 - x2).powi(2) + (y1 - y2).powi(2)
}
