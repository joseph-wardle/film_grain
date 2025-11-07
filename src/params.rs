use std::fmt;
use std::path::PathBuf;

use clap::ValueEnum;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum RadiusDist {
    Const,
    Lognorm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Algo {
    Auto,
    Grain,
    Pixel,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ColorMode {
    Luma,
    Rgb,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Device {
    Cpu,
    Gpu,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MaxRadius {
    Absolute(f32),
    Quantile(f32),
}

#[derive(Debug, Clone)]
pub struct Params {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub radius_dist: RadiusDist,
    pub radius_mean: f32,
    pub radius_stddev: f32,
    pub radius_log_mu: Option<f32>,
    pub radius_log_sigma: Option<f32>,
    pub zoom: f32,
    pub sigma_px: f32,
    pub n_samples: u32,
    pub algo: Algo,
    pub max_radius: MaxRadius,
    pub cell_delta: Option<f32>,
    pub color_mode: ColorMode,
    pub roi: Option<Roi>,
    pub size: Option<(u32, Option<u32>)>,
    pub seed: u64,
    pub dry_run: bool,
    pub explain: bool,
    pub device: Device,
    pub output_format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParamsBuilder {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub radius_dist: RadiusDist,
    pub radius_mean: f32,
    pub radius_stddev: f32,
    pub zoom: f32,
    pub sigma_px: f32,
    pub n_samples: u32,
    pub algo: Algo,
    pub max_radius: MaxRadius,
    pub cell_delta: Option<f32>,
    pub color_mode: ColorMode,
    pub roi: Option<Roi>,
    pub size: Option<(u32, Option<u32>)>,
    pub seed: u64,
    pub dry_run: bool,
    pub explain: bool,
    pub device: Device,
    pub output_format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CliArgs {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub radius_dist: RadiusDist,
    pub radius_mean: f64,
    pub radius_stddev: f64,
    pub zoom: f64,
    pub sigma_px: f64,
    pub n_samples: u32,
    pub algo: Algo,
    pub max_radius: String,
    pub cell: String,
    pub color_mode: ColorMode,
    pub device: Device,
    pub roi: Option<String>,
    pub size: Option<String>,
    pub seed: u32,
    pub dry_run: bool,
    pub explain: bool,
    pub output_format: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParamsError {
    pub field: &'static str,
    pub message: String,
}

impl ParamsError {
    pub fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

pub type ParamsResult<T> = Result<T, ParamsError>;

impl fmt::Display for ParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ParamsError {}

impl ParamsBuilder {
    pub fn build(self) -> ParamsResult<Params> {
        let radius_mean = ensure_positive(self.radius_mean, "radius")?;
        let radius_stddev_input = ensure_non_negative(self.radius_stddev, "radius-stddev")?;
        let zoom = ensure_positive(self.zoom, "zoom")?;
        let sigma_px = ensure_positive(self.sigma_px, "sigma")?;
        let n_samples = self.n_samples.max(1);
        let max_radius = ensure_max_radius(self.max_radius)?;
        let roi = ensure_roi(self.roi)?;
        let size = ensure_size(self.size)?;
        let cell_delta = ensure_cell_delta(self.cell_delta, radius_mean)?;

        let (radius_stddev, radius_log_mu, radius_log_sigma) =
            derive_radius_parameters(radius_mean, radius_stddev_input, self.radius_dist)?;

        Ok(Params {
            input_path: self.input_path,
            output_path: self.output_path,
            radius_dist: self.radius_dist,
            radius_mean,
            radius_stddev,
            radius_log_mu,
            radius_log_sigma,
            zoom,
            sigma_px,
            n_samples,
            algo: self.algo,
            max_radius,
            cell_delta,
            color_mode: self.color_mode,
            roi,
            size,
            seed: self.seed,
            dry_run: self.dry_run,
            explain: self.explain,
            device: self.device,
            output_format: self.output_format,
        })
    }
}

impl TryFrom<CliArgs> for ParamsBuilder {
    type Error = ParamsError;

    fn try_from(args: CliArgs) -> ParamsResult<Self> {
        let radius_mean = to_positive_f32(args.radius_mean, "radius")?;
        let radius_stddev = to_non_negative_f32(args.radius_stddev, "radius-stddev")?;
        let zoom = to_positive_f32(args.zoom, "zoom")?;
        let sigma_px = to_positive_f32(args.sigma_px, "sigma")?;
        let cell_delta = parse_cell_delta(&args.cell, radius_mean)?;
        let max_radius = parse_max_radius(&args.max_radius)?;
        let roi = parse_roi(args.roi.as_deref())?;
        let size = parse_size(args.size.as_deref())?;

        Ok(Self {
            input_path: args.input_path,
            output_path: args.output_path,
            radius_dist: args.radius_dist,
            radius_mean,
            radius_stddev,
            zoom,
            sigma_px,
            n_samples: args.n_samples,
            algo: args.algo,
            max_radius,
            cell_delta,
            color_mode: args.color_mode,
            roi,
            size,
            seed: args.seed as u64,
            dry_run: args.dry_run,
            explain: args.explain,
            device: args.device,
            output_format: args.output_format,
        })
    }
}

pub fn build_params(args: CliArgs) -> ParamsResult<Params> {
    ParamsBuilder::try_from(args)?.build()
}

fn derive_radius_parameters(
    radius_mean: f32,
    radius_stddev: f32,
    radius_dist: RadiusDist,
) -> ParamsResult<(f32, Option<f32>, Option<f32>)> {
    match radius_dist {
        RadiusDist::Const => Ok((0.0, None, None)),
        RadiusDist::Lognorm => {
            if radius_mean <= 0.0 {
                return Err(ParamsError::new(
                    "radius",
                    "mean radius must be positive for lognormal distribution",
                ));
            }
            if radius_stddev < 0.0 {
                return Err(ParamsError::new(
                    "radius-stddev",
                    "stddev must be non-negative",
                ));
            }
            if radius_stddev == 0.0 {
                return Ok((0.0, Some(radius_mean.ln()), Some(0.0)));
            }
            let variance_ratio = (radius_stddev * radius_stddev) / (radius_mean * radius_mean);
            let sigma_sq = (1.0 + variance_ratio).ln();
            let sigma = sigma_sq.sqrt();
            let mu = radius_mean.ln() - 0.5 * sigma_sq;
            Ok((radius_stddev, Some(mu), Some(sigma)))
        }
    }
}

pub fn default_cell_delta(radius_mean: f32) -> f32 {
    if radius_mean <= 0.0 {
        return 1.0;
    }
    let inv = (1.0 / radius_mean).ceil().max(1.0);
    1.0 / inv
}

fn ensure_positive(value: f32, field: &'static str) -> ParamsResult<f32> {
    if !value.is_finite() {
        return Err(ParamsError::new(field, "value must be finite"));
    }
    if value <= 0.0 {
        return Err(ParamsError::new(field, "value must be greater than 0"));
    }
    Ok(value)
}

fn ensure_non_negative(value: f32, field: &'static str) -> ParamsResult<f32> {
    if !value.is_finite() {
        return Err(ParamsError::new(field, "value must be finite"));
    }
    if value < 0.0 {
        return Err(ParamsError::new(field, "value must be >= 0"));
    }
    Ok(value)
}

fn ensure_cell_delta(value: Option<f32>, radius_mean: f32) -> ParamsResult<Option<f32>> {
    match value {
        Some(delta) => {
            if !delta.is_finite() {
                return Err(ParamsError::new("cell", "cell size must be finite"));
            }
            if delta <= 0.0 {
                return Err(ParamsError::new("cell", "cell size must be > 0"));
            }
            Ok(Some(delta))
        }
        None => Ok(Some(default_cell_delta(radius_mean))),
    }
}

fn ensure_max_radius(spec: MaxRadius) -> ParamsResult<MaxRadius> {
    match spec {
        MaxRadius::Absolute(value) => {
            if !value.is_finite() {
                return Err(ParamsError::new(
                    "max-radius",
                    "absolute radius must be finite",
                ));
            }
            if value <= 0.0 {
                return Err(ParamsError::new(
                    "max-radius",
                    "absolute radius must be > 0",
                ));
            }
            Ok(MaxRadius::Absolute(value))
        }
        MaxRadius::Quantile(value) => {
            if !value.is_finite() {
                return Err(ParamsError::new("max-radius", "quantile must be finite"));
            }
            if !(0.0 < value && value < 1.0) {
                return Err(ParamsError::new(
                    "max-radius",
                    "quantile must lie in the open interval (0,1)",
                ));
            }
            Ok(MaxRadius::Quantile(value))
        }
    }
}

fn ensure_roi(roi: Option<Roi>) -> ParamsResult<Option<Roi>> {
    if let Some(r) = roi {
        if r.x1 <= r.x0 || r.y1 <= r.y0 {
            return Err(ParamsError::new(
                "roi",
                "roi end must be greater than start (exclusive bounds)",
            ));
        }
        Ok(Some(r))
    } else {
        Ok(None)
    }
}

fn ensure_size(size: Option<(u32, Option<u32>)>) -> ParamsResult<Option<(u32, Option<u32>)>> {
    if let Some((width, maybe_height)) = size {
        if width == 0 {
            return Err(ParamsError::new("size", "output width must be > 0"));
        }
        if let Some(height) = maybe_height
            && height == 0
        {
            return Err(ParamsError::new("size", "output height must be > 0"));
        }
        Ok(Some((width, maybe_height)))
    } else {
        Ok(None)
    }
}

fn to_positive_f32(value: f64, field: &'static str) -> ParamsResult<f32> {
    if !value.is_finite() {
        return Err(ParamsError::new(field, "value must be finite"));
    }
    if value <= 0.0 {
        return Err(ParamsError::new(field, "value must be greater than 0"));
    }
    if value > f32::MAX as f64 {
        return Err(ParamsError::new(
            field,
            "value is too large for single precision",
        ));
    }
    Ok(value as f32)
}

fn to_non_negative_f32(value: f64, field: &'static str) -> ParamsResult<f32> {
    if !value.is_finite() {
        return Err(ParamsError::new(field, "value must be finite"));
    }
    if value < 0.0 {
        return Err(ParamsError::new(field, "value must be >= 0"));
    }
    if value > f32::MAX as f64 {
        return Err(ParamsError::new(
            field,
            "value is too large for single precision",
        ));
    }
    Ok(value as f32)
}

fn parse_cell_delta(cell: &str, radius_mean: f32) -> ParamsResult<Option<f32>> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if trimmed.eq_ignore_ascii_case("approx mu_r")
        || trimmed.eq_ignore_ascii_case("auto")
        || trimmed.eq_ignore_ascii_case("mu_r")
    {
        return Ok(Some(default_cell_delta(radius_mean)));
    }

    let value: f32 = trimmed
        .parse()
        .map_err(|_| ParamsError::new("cell", "expected a positive number or 'approx mu_r'"))?;
    if value <= 0.0 {
        return Err(ParamsError::new("cell", "cell size must be > 0"));
    }
    Ok(Some(value))
}

fn parse_max_radius(value: &str) -> ParamsResult<MaxRadius> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(MaxRadius::Quantile(0.999));
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("quantile") {
        let tail = trimmed[8..].trim();
        let tail = tail.trim_start_matches(['=', ':']).trim();
        if tail.is_empty() {
            return Ok(MaxRadius::Quantile(0.999));
        }
        let quantile: f32 = tail.parse().map_err(|_| {
            ParamsError::new("max-radius", "expected numeric quantile value in (0,1)")
        })?;
        if !(0.0 < quantile && quantile < 1.0) {
            return Err(ParamsError::new(
                "max-radius",
                "quantile must lie in the open interval (0,1)",
            ));
        }
        return Ok(MaxRadius::Quantile(quantile));
    }
    let absolute: f32 = trimmed
        .parse()
        .map_err(|_| ParamsError::new("max-radius", "expected a positive number or 'quantile'"))?;
    if absolute <= 0.0 {
        return Err(ParamsError::new(
            "max-radius",
            "absolute radius must be > 0",
        ));
    }
    Ok(MaxRadius::Absolute(absolute))
}

fn parse_roi(raw: Option<&str>) -> ParamsResult<Option<Roi>> {
    let Some(text) = raw else {
        return Ok(None);
    };
    let mut parts = text.split(':');
    let start = parts
        .next()
        .ok_or_else(|| ParamsError::new("roi", "expected X0,Y0:X1,Y1"))?;
    let end = parts
        .next()
        .ok_or_else(|| ParamsError::new("roi", "expected X0,Y0:X1,Y1"))?;
    if parts.next().is_some() {
        return Err(ParamsError::new("roi", "unexpected extra ':'"));
    }
    let (x0, y0) = parse_pair(start, "roi")?;
    let (x1, y1) = parse_pair(end, "roi")?;
    if x1 <= x0 || y1 <= y0 {
        return Err(ParamsError::new(
            "roi",
            "roi end must be greater than start (exclusive bounds)",
        ));
    }
    Ok(Some(Roi { x0, y0, x1, y1 }))
}

fn parse_pair(raw: &str, field: &'static str) -> ParamsResult<(u32, u32)> {
    let mut parts = raw.split(',');
    let first = parts
        .next()
        .ok_or_else(|| ParamsError::new(field, "expected two comma-separated integers"))?;
    let second = parts
        .next()
        .ok_or_else(|| ParamsError::new(field, "expected two comma-separated integers"))?;
    if parts.next().is_some() {
        return Err(ParamsError::new(field, "too many commas"));
    }
    let a = parse_u32(first, field)?;
    let b = parse_u32(second, field)?;
    Ok((a, b))
}

fn parse_u32(raw: &str, field: &'static str) -> ParamsResult<u32> {
    raw.trim()
        .parse()
        .map_err(|_| ParamsError::new(field, "expected an unsigned integer without suffixes"))
}

fn parse_size(raw: Option<&str>) -> ParamsResult<Option<(u32, Option<u32>)>> {
    let Some(text) = raw else {
        return Ok(None);
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ParamsError::new("size", "size string cannot be empty"));
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(idx) = lower.find('x') {
        let (left, right) = trimmed.split_at(idx);
        let width_str = left.trim();
        let height_str = right[1..].trim();
        match (width_str.is_empty(), height_str.is_empty()) {
            (false, false) => {
                let w = parse_u32(width_str, "size")?;
                let h = parse_u32(height_str, "size")?;
                Ok(Some((w, Some(h))))
            }
            (false, true) => {
                let w = parse_u32(width_str, "size")?;
                Ok(Some((w, None)))
            }
            (true, false) => Err(ParamsError::new(
                "size",
                "height-only form is not supported; provide width or WxH",
            )),
            (true, true) => Err(ParamsError::new(
                "size",
                "missing width and height around separator",
            )),
        }
    } else {
        let width = parse_u32(trimmed, "size")?;
        Ok(Some((width, None)))
    }
}
