use std::f32::consts::PI;

use rand::Rng;
use rand_distr::{Distribution, LogNormal};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::params::{MaxRadius, Params, RadiusDist, default_cell_delta};
use crate::rng;
#[cfg(target_arch = "wasm32")]
use crate::wgpu::WEBGPU_MAX_OUTPUT_PIXELS;

const EPSILON: f32 = 1e-6;
const MAX_LAMBDA: f32 = 1.0e6;

#[derive(Debug, Clone)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(width * height, data.len());
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.index(x, y)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        let idx = self.index(x, y);
        self.data[idx] = value;
    }

    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let xi = x.clamp(0, self.width as isize - 1) as usize;
        let yi = y.clamp(0, self.height as isize - 1) as usize;
        self.get(xi, yi)
    }

    pub fn pixels(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn pixels_mut(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }

    pub fn resize_nearest(&self, new_width: usize, new_height: usize) -> Plane {
        if new_width == self.width && new_height == self.height {
            return self.clone();
        }
        if new_width == 0 || new_height == 0 || self.width == 0 || self.height == 0 {
            return Plane::new(new_width, new_height);
        }
        let mut result = Plane::new(new_width, new_height);
        let scale_x = self.width as f32 / new_width as f32;
        let scale_y = self.height as f32 / new_height as f32;
        for y in 0..new_height {
            let src_y = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (self.height - 1) as f32);
            let sy = src_y.round() as usize;
            for x in 0..new_width {
                let src_x = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (self.width - 1) as f32);
                let sx = src_x.round() as usize;
                let value = self.get(sx, sy);
                result.set(x, y, value);
            }
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct RadiusProfile {
    pub dist: RadiusDist,
    pub mean_linear: f32,
    log_mu: Option<f64>,
    log_sigma: Option<f64>,
    lognormal: Option<LogNormal<f64>>,
}

impl RadiusProfile {
    pub fn new(params: &Params) -> Result<Self, String> {
        let log_mu = params.radius_log_mu.map(|v| v as f64);
        let log_sigma = params.radius_log_sigma.map(|v| v as f64);

        let lognormal = match params.radius_dist {
            RadiusDist::Const => None,
            RadiusDist::Lognorm => {
                let mu = log_mu.ok_or_else(|| {
                    "missing log-normal mean; parameters were not derived".to_string()
                })?;
                let sigma = log_sigma.ok_or_else(|| {
                    "missing log-normal sigma; parameters were not derived".to_string()
                })?;
                Some(LogNormal::new(mu, sigma).map_err(|err| err.to_string())?)
            }
        };

        Ok(Self {
            dist: params.radius_dist,
            mean_linear: params.radius_mean,
            log_mu,
            log_sigma,
            lognormal,
        })
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        match self.dist {
            RadiusDist::Const => self.mean_linear,
            RadiusDist::Lognorm => {
                if let Some(dist) = &self.lognormal {
                    dist.sample(rng) as f32
                } else {
                    self.mean_linear
                }
            }
        }
    }

    pub fn quantile(&self, p: f32) -> f32 {
        match self.dist {
            RadiusDist::Const => self.mean_linear,
            RadiusDist::Lognorm => {
                let mu = self.log_mu.unwrap_or(self.mean_linear as f64);
                let sigma = self.log_sigma.unwrap_or(0.0);
                if sigma == 0.0 {
                    return mu.exp() as f32;
                }
                let normal = Normal::new(0.0, 1.0).expect("unit normal");
                let z = normal.inverse_cdf(p as f64);
                (mu + sigma * z).exp() as f32
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Derived {
    pub input_width: usize,
    pub input_height: usize,
    pub output_width: usize,
    pub output_height: usize,
    pub inv_e_pi_r2: f32,
    pub rm: f32,
    pub delta: f32,
    pub offsets: Vec<[f32; 2]>,
    pub offsets_input: Vec<[f32; 2]>,
    pub radius: RadiusProfile,
}

pub fn derive_common(params: &Params, input_size: (usize, usize)) -> Result<Derived, String> {
    let (input_width, input_height) = input_size;
    if input_width == 0 || input_height == 0 {
        return Err("input image is empty after ROI".into());
    }

    let (output_width, output_height) = resolve_output_size(params, input_size)?;
    if output_width == 0 || output_height == 0 {
        return Err("output dimensions must be positive".into());
    }

    let mean_sq = params.radius_mean * params.radius_mean;
    let variance = params.radius_stddev * params.radius_stddev;
    let inv_e_pi_r2 = 1.0 / (PI * (mean_sq + variance).max(EPSILON));

    let radius = RadiusProfile::new(params)?;
    let rm = match params.max_radius {
        MaxRadius::Absolute(value) => value,
        MaxRadius::Quantile(p) => radius.quantile(p),
    }
    .max(EPSILON);

    let delta = params
        .cell_delta
        .unwrap_or_else(|| default_cell_delta(params.radius_mean))
        .max(EPSILON);

    let offsets = rng::make_offsets(params.seed, params.n_samples as usize, params.sigma_px);
    let offsets_input: Vec<[f32; 2]> = offsets
        .iter()
        .map(|offset| [offset[0] / params.zoom, offset[1] / params.zoom])
        .collect();

    Ok(Derived {
        input_width,
        input_height,
        output_width,
        output_height,
        inv_e_pi_r2,
        rm,
        delta,
        offsets,
        offsets_input,
        radius,
    })
}

pub fn normalize_plane(plane: &Plane) -> (Plane, f32) {
    let max_value = plane
        .pixels()
        .iter()
        .copied()
        .fold(0.0_f32, |acc, value| acc.max(value));
    let already_normalized = max_value <= 1.0 + EPSILON;
    let mut normalized = Plane::new(plane.width, plane.height);
    for (dst, &src) in normalized
        .pixels_mut()
        .iter_mut()
        .zip(plane.pixels().iter())
    {
        let value = if already_normalized {
            src
        } else {
            let denom = (max_value + EPSILON).max(EPSILON);
            src / denom
        };
        *dst = value.clamp(0.0, 1.0 - EPSILON);
    }
    (normalized, max_value)
}

pub fn lambda_plane(normalized: &Plane, inv_e_pi_r2: f32) -> Plane {
    let mut lambda = Plane::new(normalized.width, normalized.height);
    for (dst, &value) in lambda
        .pixels_mut()
        .iter_mut()
        .zip(normalized.pixels().iter())
    {
        let clamped = value.clamp(0.0, 1.0 - EPSILON);
        let safe = (1.0 - clamped).max(EPSILON);
        let activity = -inv_e_pi_r2 * safe.ln();
        *dst = activity.min(MAX_LAMBDA);
    }
    lambda
}

fn resolve_output_size(
    params: &Params,
    input_size: (usize, usize),
) -> Result<(usize, usize), String> {
    if let Some((width, maybe_height)) = params.size {
        let width = width as usize;
        if width == 0 {
            return Err("output width must be > 0".into());
        }
        let height = maybe_height
            .map(|h| h as usize)
            .or_else(|| {
                let (in_w, in_h) = (input_size.0 as f32, input_size.1 as f32);
                if in_w <= 0.0 || in_h <= 0.0 {
                    None
                } else {
                    let aspect = in_h / in_w;
                    let computed = (width as f32 * aspect).round().max(1.0);
                    Some(computed as usize)
                }
            })
            .ok_or_else(|| "could not infer output height".to_string())?;
        if height == 0 {
            return Err("output height must be > 0".into());
        }
        #[cfg(target_arch = "wasm32")]
        validate_webgpu_output_dims(width, height)?;
        return Ok((width, height));
    }

    let width = ((input_size.0 as f32) * params.zoom).ceil().max(1.0) as usize;
    let height = ((input_size.1 as f32) * params.zoom).ceil().max(1.0) as usize;
    #[cfg(target_arch = "wasm32")]
    validate_webgpu_output_dims(width, height)?;
    Ok((width, height))
}

#[cfg(target_arch = "wasm32")]
fn validate_webgpu_output_dims(width: usize, height: usize) -> Result<(), String> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "output size is too large".to_string())?;
    if pixels > WEBGPU_MAX_OUTPUT_PIXELS {
        let limit_mp = WEBGPU_MAX_OUTPUT_PIXELS as f32 / 1_000_000.0;
        return Err(format!(
            "output size {width}×{height} exceeds the WebGPU browser budget (~{limit_mp:.1} MP)"
        ));
    }
    Ok(())
}
