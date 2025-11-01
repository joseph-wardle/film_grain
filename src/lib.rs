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
    pub mean_r: f32,
    pub std_r: f32,
    pub zoom_s: f32,
    pub sigma_px: f32,
    pub n_samples: u32,
    pub algo: Algo,
    pub max_radius: MaxRadius,
    pub cell_delta: Option<f32>,
    pub color_mode: ColorMode,
    pub roi: Option<Roi>,
    pub size: Option<(u32, Option<u32>)>,
    pub seed: u32,
    pub dry_run: bool,
    pub explain: bool,
}

impl Params {
    pub fn mean_radius_linear(&self) -> f32 {
        match self.radius_dist {
            RadiusDist::Const => self.mean_r,
            RadiusDist::Lognorm => (self.mean_r + 0.5 * self.std_r.powi(2)).exp(),
        }
    }

    pub fn std_radius_linear(&self) -> f32 {
        match self.radius_dist {
            RadiusDist::Const => self.std_r,
            RadiusDist::Lognorm => {
                let sigma_sq = self.std_r * self.std_r;
                let mu = self.mean_r;
                let exp_sigma_sq = sigma_sq.exp();
                let variance = (exp_sigma_sq - 1.0) * (2.0 * mu + sigma_sq).exp();
                variance.max(0.0).sqrt()
            }
        }
    }
}

#[derive(Debug)]
pub struct CliArgs {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub radius_dist: RadiusDist,
    pub radius_mean: f64,
    pub radius_stddev: f64,
    pub zoom_s: f64,
    pub sigma_px: f64,
    pub n_samples: u32,
    pub algo: Algo,
    pub max_radius: String,
    pub cell: String,
    pub color_mode: ColorMode,
    pub roi: Option<String>,
    pub size: Option<String>,
    pub seed: u32,
    pub dry_run: bool,
    pub explain: bool,
}

#[derive(Debug, Clone)]
pub struct ParamsError {
    field: &'static str,
    message: String,
}

impl ParamsError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ParamsError {}

pub type ParamsResult<T> = Result<T, ParamsError>;

pub fn build_params(args: CliArgs) -> ParamsResult<Params> {
    let radius_mean = to_f32(args.radius_mean, "radius")?;
    let radius_stddev = to_f32(args.radius_stddev, "radius-stddev")?;
    let sigma_px = to_f32(args.sigma_px, "sigma")?;
    let zoom_s = to_f32(args.zoom_s, "zoom")?;

    let (mean_r, std_r, radius_dist) =
        derive_radius_params(radius_mean, radius_stddev, args.radius_dist)?;

    let cell_delta = parse_cell_delta(&args.cell, radius_mean)?;
    let max_radius = parse_max_radius(&args.max_radius)?;
    let roi = parse_roi(args.roi.as_deref())?;
    let size = parse_size(args.size.as_deref())?;

    Ok(Params {
        input_path: args.input_path,
        output_path: args.output_path,
        radius_dist,
        mean_r,
        std_r,
        zoom_s,
        sigma_px,
        n_samples: args.n_samples,
        algo: args.algo,
        max_radius,
        cell_delta,
        color_mode: args.color_mode,
        roi,
        size,
        seed: args.seed,
        dry_run: args.dry_run,
        explain: args.explain,
    })
}

fn to_f32(value: f64, field: &'static str) -> ParamsResult<f32> {
    if value.is_nan() || value.is_infinite() {
        return Err(ParamsError::new(field, "value must be finite"));
    }
    if value > f32::MAX as f64 {
        return Err(ParamsError::new(
            field,
            "value is too large for single precision",
        ));
    }
    Ok(value as f32)
}

fn derive_radius_params(
    radius_mean: f32,
    radius_stddev: f32,
    radius_dist: RadiusDist,
) -> ParamsResult<(f32, f32, RadiusDist)> {
    match radius_dist {
        RadiusDist::Const => Ok((radius_mean, 0.0, RadiusDist::Const)),
        RadiusDist::Lognorm => {
            if radius_mean <= 0.0 {
                return Err(ParamsError::new(
                    "radius",
                    "mean radius must be positive for lognormal distribution",
                ));
            }
            let variance_ratio = if radius_stddev == 0.0 {
                0.0
            } else {
                (radius_stddev * radius_stddev) / (radius_mean * radius_mean)
            };
            let sigma_sq = (1.0 + variance_ratio).ln();
            let sigma = sigma_sq.sqrt();
            let mu = radius_mean.ln() - 0.5 * sigma_sq;
            Ok((mu, sigma, RadiusDist::Lognorm))
        }
    }
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

fn default_cell_delta(radius_mean: f32) -> f32 {
    if radius_mean <= 0.0 {
        return 1.0;
    }
    let inv = (1.0 / radius_mean).ceil().max(1.0);
    1.0 / inv
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_delta_matches_simple_cases() {
        assert_eq!(default_cell_delta(0.25), 0.25);
        assert_eq!(default_cell_delta(1.0), 1.0);
        assert_eq!(default_cell_delta(2.0), 1.0);
    }

    #[test]
    fn parse_quantile_defaults() {
        assert_eq!(
            parse_max_radius("quantile").unwrap(),
            MaxRadius::Quantile(0.999)
        );
        assert_eq!(
            parse_max_radius("quantile 0.9").unwrap(),
            MaxRadius::Quantile(0.9)
        );
    }

    #[test]
    fn parse_roi_ok() {
        let roi = parse_roi(Some("1,2:10,20")).unwrap().unwrap();
        assert_eq!(roi.x0, 1);
        assert_eq!(roi.y0, 2);
        assert_eq!(roi.x1, 10);
        assert_eq!(roi.y1, 20);
    }
}
