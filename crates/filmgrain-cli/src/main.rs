use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::eyre::Result;
use filmgrain_core::{Backend, FilmGrainMode, FilmGrainParams};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Copy, Clone, ValueEnum, Debug)]
enum BackendOpt {
    CpuSingle,
    CpuMulti,
    Gpu,
}

#[derive(Parser, Debug)]
#[command(
    name = "filmgrain",
    version,
    about = "Realistic film grain rendering (CPU + WGPU)"
)]
struct Cli {
    /// Log level
    #[arg(long, value_name="LEVEL", default_value="info",
          value_parser=["error","warn","info","debug","trace"])]
    log_level: String,

    /// Global seed
    #[arg(long, default_value_t = 1)]
    seed: u64,

    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Pixel-wise film grain
    Pixelwise(SharedArgs),
    /// Grain-wise film grain
    Grainwise(SharedArgs),
    /// Automatically choose the optimal algorithm
    Auto(SharedArgs),
}

#[derive(Parser, Debug)]
struct SharedArgs {
    #[arg(short = 'i', long)]
    input: PathBuf,
    #[arg(short = 'o', long)]
    output: PathBuf,

    #[arg(long, value_enum)]
    backend: Option<BackendOpt>,

    /// Threads (CPU)
    #[arg(long)]
    threads: Option<usize>,

    /// Mean grain radius (input pixels)
    #[arg(long = "grain-size", default_value_t = 0.1)]
    grain_size: f32,

    /// Stddev of grain radius (absolute, not fraction)
    #[arg(long = "grain-size-std-dev", default_value_t = 0.0)]
    grain_size_std_dev: f32,

    /// Gaussian σ (output pixels)
    #[arg(long = "blur-sigma", default_value_t = 0.8)]
    blur_sigma: f32,

    /// Monte-Carlo samples
    #[arg(long = "n-monte-carlo", default_value_t = 800)]
    n_monte_carlo: u32,

    /// ROI (input coordinates)
    #[arg(long = "x-start", default_value_t = 0)]
    x_start: u32,
    #[arg(long = "y-start", default_value_t = 0)]
    y_start: u32,
    #[arg(long = "x-end", default_value_t = 0)]
    x_end: u32,
    #[arg(long = "y-end", default_value_t = 0)]
    y_end: u32,

    /// Output size (rows/cols). If 0, uses ROI size (zoom=1).
    #[arg(long = "output-width", default_value_t = 0)]
    output_width: u32,
    #[arg(long = "output-height", default_value_t = 0)]
    output_height: u32,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    // logging
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(cli.log_level));
    fmt().with_env_filter(filter).init();

    let (mode, args) = match &cli.cmd {
        Commands::Pixelwise(a) => (FilmGrainMode::PixelWise, a),
        Commands::Grainwise(a) => (FilmGrainMode::GrainWise, a),
        Commands::Auto(a) => (FilmGrainMode::Auto, a),
    };

    let img = image::open(&args.input)?;

    let backend = match args.backend.unwrap_or(BackendOpt::Gpu) {
        BackendOpt::CpuSingle => Backend::CpuSingle,
        BackendOpt::CpuMulti => Backend::CpuMulti,
        BackendOpt::Gpu => Backend::Gpu,
    };

    let mut params = FilmGrainParams {
        grain_size: args.grain_size,
        grain_size_std_dev: args.grain_size_std_dev,
        blur_sigma: args.blur_sigma,
        n_monte_carlo: args.n_monte_carlo,
        mode,
        backend,
        seed: cli.seed,
        x_start: args.x_start,
        y_start: args.y_start,
        x_end: args.x_end,
        y_end: args.y_end,
        output_width: args.output_width,
        output_height: args.output_height,
        threads: args.threads,
    };

    let pb = {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{spinner} {msg}")?);
        pb.set_message("Rendering…");
        Some(pb)
    };

    if let Some(pb) = &pb {
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
    }

    params.finalize(img.width(), img.height());
    let out = filmgrain_core::render_image(&img, &params)?;

    if let Some(pb) = &pb {
        pb.finish_with_message("Done");
    }

    out.save(&args.output)?;
    Ok(())
}
