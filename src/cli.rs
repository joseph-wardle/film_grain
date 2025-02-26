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
