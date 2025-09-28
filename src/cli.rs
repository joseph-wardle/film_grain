use crate::rendering::RenderingAlgorithm;
use clap::Parser;
use clap::ValueHint;
use std::path::PathBuf;

/// Film Grain Rendering based on Newson et al. (2017)
#[derive(Debug, Parser)]
#[clap(
    author,
    version,
    about = "Apply a realistic film grain rendering to an input image",
    long_about = None,
    rename_all = "kebab-case"
)]
pub struct Cli {
    /// Input image file path
    #[arg(value_name = "INPUT", value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output image file path.
    #[arg(value_name = "OUTPUT", value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Average grain radius (µ_r) in input image units
    #[arg(short = 'r', default_value_t = 0.1, value_name = "R")]
    pub average_grain_radius: f32,

    /// Grain standard deviation factor (σ_r, as a fraction of µ_r)
    #[clap(long = "sigma-r", default_value_t = 0.0, value_name = "FACTOR")]
    pub grain_radius_stddev_factor: f32,

    /// Standard deviation of the Gaussian filter (σ_filter) in output pixels
    #[clap(long = "sigma-filter", default_value_t = 0.8, value_name = "SIGMA")]
    pub gaussian_filter_stddev: f32,

    /// Zoom factor for the output image
    #[clap(short = 'z', long = "zoom", default_value_t = 1.0)]
    pub zoom: f32,

    /// Rendering algorithm to use
    #[clap(short = 'a', long = "algorithm", default_value = "automatic")]
    pub algorithm: RenderingAlgorithm,

    /// Number of Monte Carlo Samples to evaluate per pixel
    #[clap(
        short = 's',
        long = "sample-count",
        value_name = "N",
        default_value_t = 800
    )]
    pub monte_carlo_sample_count: u32,

    /// x-coordinate of the top-left corner of the input
    #[clap(long = "x-min", value_name = "XMIN", default_value_t = 0.0)]
    pub region_min_x: f32,

    /// y-coordinate of the top-left corner of the input
    #[clap(long = "y-min", value_name = "YMIN", default_value_t = 0.0)]
    pub region_min_y: f32,

    /// x-coordinate of the bottom-right corner of the input (defaults to image width)
    #[clap(long = "x-max", value_name = "XMAX")]
    pub region_max_x: Option<f32>,

    /// y-coordinate of the bottom-right corner of the input (defaults to image height)
    #[clap(long = "y-max", value_name = "YMAX")]
    pub region_max_y: Option<f32>,

    /// Output image width (number of columns; if not provided, computed from zoom)
    #[clap(long = "width")]
    pub output_width: Option<u32>,

    /// Output image height (number of rows; if not provided, computed from zoom)
    #[clap(long = "height")]
    pub output_height: Option<u32>,

    /// Seed for the pseudo–random number generator used throughout the algorithm
    #[clap(long = "seed", default_value_t = 42)]
    pub rng_seed: u32,
}
