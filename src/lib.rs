use choose::choose_algorithm;
use color::Workspace;
use grainwise::render_grainwise;
use model::{Derived, derive_common, lambda_plane, normalize_plane};
use pixelwise::render_pixelwise;
use std::fs;
use thiserror::Error;

pub mod params;

mod choose;
mod color;
mod grainwise;
mod model;
mod pixelwise;
mod rng;
pub mod wgpu;

pub use params::{
    Algo, CliArgs, ColorMode, Device, MaxRadius, Params, ParamsError, ParamsResult, RadiusDist,
    Roi, build_params, default_cell_delta,
};

pub type RenderResult<T> = Result<T, RenderError>;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("parameter error: {0}")]
    Params(#[from] ParamsError),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error("{0}")]
    Message(String),
}

#[derive(Debug)]
pub struct RenderStats {
    pub algorithm: Algo,
    pub device: Device,
    pub input_size: (usize, usize),
    pub output_size: (usize, usize),
    pub n_samples: u32,
    pub sigma_ratio: f32,
    pub rm_ratio: f32,
}

pub fn render(params: &Params) -> RenderResult<RenderStats> {
    ensure_cpu(params)?;
    let mut workspace = Workspace::load(params)?;
    let input_size = workspace.dimensions();
    let derived = derive_common(params, input_size).map_err(RenderError::Message)?;
    let algorithm = choose_algorithm(params, &derived);

    workspace.for_each_plane(|plane, _| {
        let (normalized, _) = normalize_plane(plane);
        let lambda = lambda_plane(&normalized, derived.inv_e_pi_r2);
        let output = match algorithm {
            Algo::Pixel => render_pixelwise(&lambda, params, &derived),
            Algo::Grain => render_grainwise(&lambda, params, &derived),
            Algo::Auto => unreachable!("auto selection resolved before rendering"),
        };
        Ok(output)
    })?;

    if let Some(parent) = params.output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let format = resolve_format(params)?;
    workspace.save(&params.output_path, format)?;

    Ok(make_stats(params, &derived, algorithm))
}

pub fn dry_run(params: &Params) -> RenderResult<RenderStats> {
    ensure_cpu(params)?;
    let workspace = Workspace::load(params)?;
    let input_size = workspace.dimensions();
    let derived = derive_common(params, input_size).map_err(RenderError::Message)?;
    let algorithm = choose_algorithm(params, &derived);
    Ok(make_stats(params, &derived, algorithm))
}

fn ensure_cpu(params: &Params) -> RenderResult<()> {
    if matches!(params.device, Device::Gpu) {
        return Err(RenderError::Unsupported(
            "GPU backend is not implemented yet",
        ));
    }
    Ok(())
}

fn resolve_format(params: &Params) -> RenderResult<image::ImageFormat> {
    if let Some(token) = params.output_format.as_deref() {
        parse_format_token(token)
    } else if let Some(ext) = params.output_path.extension() {
        parse_format_token(ext.to_string_lossy().as_ref())
    } else {
        Ok(image::ImageFormat::Png)
    }
}

fn parse_format_token(token: &str) -> RenderResult<image::ImageFormat> {
    let trimmed = token.trim().trim_start_matches('.');
    image::ImageFormat::from_extension(trimmed).ok_or_else(|| {
        RenderError::Message(format!("unsupported or unknown image format '{trimmed}'"))
    })
}

fn make_stats(params: &Params, derived: &Derived, algorithm: Algo) -> RenderStats {
    let mean = params.radius_mean.max(1e-6);
    let sigma_ratio = if mean > 0.0 {
        params.radius_stddev / mean
    } else {
        0.0
    };
    let rm_ratio = derived.rm / mean;
    RenderStats {
        algorithm,
        device: params.device,
        input_size: (derived.input_width, derived.input_height),
        output_size: (derived.output_width, derived.output_height),
        n_samples: params.n_samples,
        sigma_ratio,
        rm_ratio,
    }
}
