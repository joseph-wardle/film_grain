use serde::{Deserialize, Serialize};

pub const MAX_GREY_LEVEL: usize = 255;
pub const EPSILON_GREY_LEVEL: f32 = 0.1;

/// Rendering backend.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Backend {
    CpuSingle,
    CpuMulti,
    Gpu,
}

/// Algorithmic mode.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FilmGrainMode {
    PixelWise,
    GrainWise,
    Auto,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilmGrainParams {
    pub grain_size: f32,          // average grain radius (mu_R)
    pub grain_size_std_dev: f32,  // standard deviation of grain radius (sigma_R)
    pub blur_sigma: f32,          // standard deviation of the low-pass filter (sigmaFilter)
    pub n_monte_carlo: u32,       // number of Monte Carlo simulations (NMonteCarlo)
    pub mode: FilmGrainMode,
    pub backend: Backend,
    pub seed: u64,                // seed for random number generator (for reproducibility)
    pub x_start: u32,             // xA: left bound of input region (inclusive)
    pub y_start: u32,             // yA: top bound of input region (inclusive)
    pub x_end: u32,               // xB: right bound of input region (exclusive)
    pub y_end: u32,               // yB: bottom bound of input region (exclusive)
    pub output_width: u32,      // nOut: output image width (pixels)
    pub output_height: u32,     // mOut: output image height (pixels)
    pub threads: Option<usize>,   // for rayon threadpool
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            grain_size: 0.05,
            grain_size_std_dev: 0.0,
            blur_sigma: 0.8,
            n_monte_carlo: 800,
            mode: FilmGrainMode::Auto,
            backend: Backend::Gpu,
            seed: 0,
            x_start: 0, y_start: 0,
            x_end: 0, y_end: 0,                // 0 means “use full image dims”
            output_width: 0, output_height: 0, // 0 means “use zoom=1”
            threads: None,
        }
    }
}

impl FilmGrainParams {
    /// Validate and finalize parameters given an input image size.
    /// Sets default region (full image) and output size (same as region, unless zoom specified).
    pub fn finalize(&mut self, input_width: u32, input_height: u32) {
        if self.x_end == 0 || self.y_end == 0 {
            // If region end not set, default to full image
            self.x_start = 0;
            self.y_start = 0;
            self.x_end = input_width;
            self.y_end = input_height;
        }
        // Clamp region to image bounds
        self.x_start = self.x_start.min(input_width);
        self.y_start = self.y_start.min(input_height);
        self.x_end = self.x_end.min(input_width);
        self.y_end = self.y_end.min(input_height);
        if self.x_end <= self.x_start || self.y_end <= self.y_start {
            panic!("Invalid region of interest specified (x_end <= x_start or y_end <= y_start)");
        }
        let region_width = self.x_end - self.x_start;
        let region_height = self.y_end - self.y_start;
        // If output size not set, default to same as region (no scaling)
        if self.output_width == 0 || self.output_height == 0 {
            self.output_width = region_width;
            self.output_height = region_height;
        }
    }
}

/// Paper-inspired heuristic: pixel-wise faster when σ_r/μ_r small; grain-wise for larger.
/// TODO: Decide boundary based on empirical data
pub fn choose_mode(mu_r: f32, sigma_r: f32) -> FilmGrainMode {
    let ratio = if mu_r > 0.0 { sigma_r / mu_r } else { 1.0 };
    if ratio <= 0.35 { FilmGrainMode::PixelWise } else { FilmGrainMode::GrainWise }
}
