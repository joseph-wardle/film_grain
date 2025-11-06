use choose::choose_algorithm;
use color::Workspace;
use grainwise::render_grainwise;
use image::RgbImage;
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

pub use color::InputImage;
pub use params::{
    Algo, CliArgs, ColorMode, Device, MaxRadius, Params, ParamsBuilder, ParamsError, ParamsResult,
    RadiusDist, Roi, build_params, default_cell_delta,
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
    #[error("gpu error: {0}")]
    Gpu(String),
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
    let (image, stats) = render_to_image(params)?;

    if let Some(parent) = params.output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let format = resolve_format(params)?;
    image
        .save_with_format(&params.output_path, format)
        .map_err(RenderError::from)?;

    Ok(stats)
}

pub fn render_to_image(params: &Params) -> RenderResult<(RgbImage, RenderStats)> {
    let workspace = Workspace::load(params)?;
    render_from_workspace(workspace, params)
}

pub fn render_with_input_image(
    input: &InputImage,
    params: &Params,
) -> RenderResult<(RgbImage, RenderStats)> {
    let workspace = input.to_workspace();
    render_from_workspace(workspace, params)
}

pub fn dry_run(params: &Params) -> RenderResult<RenderStats> {
    let workspace = Workspace::load(params)?;
    let (derived, algorithm) = derive_for_workspace(&workspace, params)?;
    Ok(make_stats(params, &derived, algorithm))
}

pub fn dry_run_with_input_image(input: &InputImage, params: &Params) -> RenderResult<RenderStats> {
    let workspace = input.to_workspace();
    let (derived, algorithm) = derive_for_workspace(&workspace, params)?;
    Ok(make_stats(params, &derived, algorithm))
}

fn render_from_workspace(
    mut workspace: Workspace,
    params: &Params,
) -> RenderResult<(RgbImage, RenderStats)> {
    let (derived, algorithm) = derive_for_workspace(&workspace, params)?;
    let gpu_ctx = match params.device {
        Device::Gpu => Some(wgpu::context()?),
        Device::Cpu => None,
    };

    workspace.for_each_plane(|plane, _| {
        let (normalized, _) = normalize_plane(plane);
        let lambda = lambda_plane(&normalized, derived.inv_e_pi_r2);
        match (params.device, algorithm) {
            (Device::Cpu, Algo::Pixel) => Ok(render_pixelwise(&lambda, params, &derived)),
            (Device::Cpu, Algo::Grain) => Ok(render_grainwise(&lambda, params, &derived)),
            (Device::Gpu, Algo::Pixel) => {
                let ctx = gpu_ctx.expect("gpu context initialized");
                wgpu::render_pixelwise_gpu(ctx, &lambda, params, &derived)
            }
            (Device::Gpu, Algo::Grain) => {
                let ctx = gpu_ctx.expect("gpu context initialized");
                wgpu::render_grainwise_gpu(ctx, &lambda, params, &derived)
            }
            (_, Algo::Auto) => unreachable!("auto selection resolved before rendering"),
        }
    })?;

    let stats = make_stats(params, &derived, algorithm);
    let image = workspace.into_rgb_image()?;
    Ok((image, stats))
}

fn derive_for_workspace(workspace: &Workspace, params: &Params) -> RenderResult<(Derived, Algo)> {
    let input_size = workspace.dimensions();
    let derived = derive_common(params, input_size).map_err(RenderError::Message)?;
    let algorithm = choose_algorithm(params, &derived);
    Ok((derived, algorithm))
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
