use clap::Parser;
use std::path::PathBuf;

use film_grain::{Algo, CliArgs, ColorMode, Device, RadiusDist, RenderStats};

fn main() {
    let args = Cli::parse();
    handle_render(args);
}

fn handle_render(args: Cli) {
    if matches!(args.radius_dist, RadiusDist::Const) && args.radius_stddev > 0.0 {
        eprintln!("warning: --radius-sd is ignored when --radius-dist=const");
    }

    let cli_args = CliArgs {
        input_path: args.input.clone(),
        output_path: args.output.clone(),
        radius_dist: args.radius_dist,
        radius_mean: args.radius,
        radius_stddev: args.radius_stddev,
        zoom: args.zoom,
        sigma_px: args.sigma,
        n_samples: args.iters,
        algo: args.algo,
        max_radius: args.max_radius.clone(),
        cell: args.cell.clone(),
        color_mode: args.color_mode,
        device: args.device,
        roi: args.roi.clone(),
        size: args.size.clone(),
        seed: args.seed,
        dry_run: args.dry_run,
        explain: args.explain,
        output_format: args.format.clone(),
    };

    let params = match film_grain::build_params(cli_args) {
        Ok(params) => params,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    };

    if params.dry_run {
        match film_grain::dry_run(&params) {
            Ok(stats) => {
                println!("{:#?}", params);
                print_stats(&stats, params.explain, true, &params.output_path);
            }
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    match film_grain::render(&params) {
        Ok(stats) => {
            print_stats(&stats, params.explain, false, &params.output_path);
        }
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "filmgrain",
    bin_name = "filmgrain",
    version,
    about = "Render physically-based film grain via a filtered Boolean model.",
    arg_required_else_help = true
)]
struct Cli {
    /// Path to input image
    #[arg(value_name = "INPUT", help_heading = "REQUIRED")]
    input: PathBuf,

    /// Output path
    #[arg(
        short,
        long,
        value_name = "OUTPUT",
        required = true,
        help_heading = "REQUIRED"
    )]
    output: PathBuf,

    #[arg(
        long,
        value_name = "R",
        default_value = "0.10",
        value_parser = parse_positive_f64,
        help_heading = "MODEL & UNITS",
        help = "Mean grain radius mu_r in input pixels (0 < R)"
    )]
    radius: f64,

    #[arg(
        long = "radius-dist",
        value_name = "KIND",
        value_enum,
        default_value_t = RadiusDist::Const,
        help_heading = "MODEL & UNITS",
        help = "Grain radius distribution"
    )]
    radius_dist: RadiusDist,

    #[arg(
        long = "radius-stddev",
        value_name = "SD",
        default_value = "0.00",
        value_parser = parse_non_negative_f64,
        help_heading = "MODEL & UNITS",
        help = "Stddev sigma_r in input pixels (only if lognorm)"
    )]
    radius_stddev: f64,

    #[arg(
        long,
        value_name = "S",
        default_value = "1.0",
        value_parser = parse_positive_f64,
        help_heading = "MODEL & UNITS",
        help = "Output zoom factor s (output size = s * input)"
    )]
    zoom: f64,

    #[arg(
        long,
        value_name = "PX",
        default_value = "0.8",
        value_parser = parse_positive_f64,
        help_heading = "MODEL & UNITS",
        help = "Gaussian filter sigma in output pixels"
    )]
    sigma: f64,

    #[arg(
        long,
        value_name = "N",
        default_value = "32",
        value_parser = parse_iters,
        help_heading = "MODEL & UNITS",
        help = "Monte Carlo samples N (quality vs speed)"
    )]
    iters: u32,

    #[arg(
        long = "color-mode",
        value_name = "M",
        value_enum,
        default_value_t = ColorMode::Luma,
        help_heading = "COLOR",
        help = "'rgb' (independent per channel) | 'luma' (apply in Y)"
    )]
    color_mode: ColorMode,

    #[arg(
        long,
        value_name = "DEVICE",
        value_enum,
        default_value_t = Device::Gpu,
        help_heading = "DEVICE",
        help = "'cpu' | 'gpu'"
    )]
    device: Device,

    #[arg(
        long,
        value_name = "A",
        value_enum,
        default_value_t = Algo::Auto,
        help_heading = "ALGORITHM",
        help = "'auto' | 'grain' | 'pixel'"
    )]
    algo: Algo,

    #[arg(
        long = "max-radius",
        value_name = "RM",
        default_value = "quantile 0.999",
        help_heading = "ALGORITHM",
        help = "Clamp radius rm (pixel-wise only; input pixels)"
    )]
    max_radius: String,

    #[arg(
        long,
        value_name = "DELTA",
        default_value = "approx mu_r",
        help_heading = "ALGORITHM",
        help = "Cell size delta (pixel-wise only; input pixels)"
    )]
    cell: String,

    #[arg(
        long,
        value_name = "X0,Y0:X1,Y1",
        help_heading = "ROI & OUTPUT SIZING",
        help = "Crop region in input pixel coords (before zoom)"
    )]
    roi: Option<String>,

    #[arg(
        long,
        value_name = "WxH",
        help_heading = "ROI & OUTPUT SIZING",
        help = "Force output WxH (overrides --zoom; preserves aspect if one side given)"
    )]
    size: Option<String>,

    #[arg(
        long,
        value_name = "EXT",
        help_heading = "I/O & QUALITY",
        help = "Force output format by extension (png/jpg/exr/tif) [default: from path]"
    )]
    format: Option<String>,

    #[arg(
        long,
        help_heading = "I/O & QUALITY",
        help = "Validate and print derived parameters; no rendering"
    )]
    dry_run: bool,

    #[arg(
        long,
        help_heading = "I/O & QUALITY",
        help = "Print chosen algorithm and why (incl. estimated cost)"
    )]
    explain: bool,
    #[arg(
        long,
        value_name = "SEED",
        default_value_t = 5489,
        help_heading = "I/O & QUALITY",
        help = "Seed for all stochastic sampling"
    )]
    seed: u32,
}

fn print_stats(stats: &RenderStats, explain: bool, dry_run: bool, output: &PathBuf) {
    if dry_run {
        println!(
            "dry-run: {:?} algorithm, input {}x{} → output {}x{}, samples {}",
            stats.algorithm,
            stats.input_size.0,
            stats.input_size.1,
            stats.output_size.0,
            stats.output_size.1,
            stats.n_samples
        );
    } else {
        println!(
            "rendered {:?} algorithm → {} ({}x{})",
            stats.algorithm,
            output.display(),
            stats.output_size.0,
            stats.output_size.1
        );
    }
    if explain {
        println!(
            "sigma/mean {:.3}, rm/mean {:.3}, samples {} on {:?}",
            stats.sigma_ratio, stats.rm_ratio, stats.n_samples, stats.device
        );
    }
}

fn parse_positive_f64(arg: &str) -> Result<f64, String> {
    let value: f64 = arg
        .parse()
        .map_err(|err| format!("invalid number: {err}"))?;
    if value <= 0.0 {
        Err("value must be greater than 0".into())
    } else {
        Ok(value)
    }
}

fn parse_non_negative_f64(arg: &str) -> Result<f64, String> {
    let value: f64 = arg
        .parse()
        .map_err(|err| format!("invalid number: {err}"))?;
    if value < 0.0 {
        Err("value must be >= 0".into())
    } else {
        Ok(value)
    }
}

fn parse_iters(arg: &str) -> Result<u32, String> {
    let value: u32 = arg
        .parse()
        .map_err(|err| format!("invalid integer: {err}"))?;
    if value < 1 {
        Err("iters must be >= 1".into())
    } else {
        Ok(value)
    }
}
