//! filmgrain-core: library for film grain synthesis (CPU + GPU).
//!
//! Implements the inhomogeneous Boolean model with Monte-Carlo Gaussian filtering
//! in two algorithmic modes: **PixelWise** and **GrainWise**.
//!
//! # Determinism
//! CPU is bit-exact given seed & params. GPU is deterministic for a fixed backend/driver
//! to the extent the backend guarantees, using integer PRNG & explicit barriers.

mod config;
mod error;
mod rng;
mod util;

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use config::{Backend, FilmGrainMode, FilmGrainParams};
pub use error::{Error, Result};

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use tracing::{debug, info, instrument, trace};

/// Render a single-channel (luma) or 3-channel image with film grain.
/// If input is RGB, grain is applied **independently per channel** per §5.2 of the paper.
#[instrument(skip(img))]
pub fn render_image(img: &DynamicImage, params: &FilmGrainParams) -> Result<DynamicImage> {
    let (w, h) = img.dimensions();
    let mut out: RgbImage = ImageBuffer::new(params.output_width, params.output_height);

    info!(width = w, height = h, "starting film grain render");
    debug!(
        backend = ?params.backend,
        mode = ?params.mode,
        output_width = params.output_width,
        output_height = params.output_height,
        "render configuration captured"
    );

    // Split channels; process independently (color grain)
    let rgb = img.to_rgb8();
    let mut chan_outs: [Vec<f32>; 3] = [
        vec![0.0; (params.output_height * params.output_width) as usize],
        vec![0.0; (params.output_height * params.output_width) as usize],
        vec![0.0; (params.output_height * params.output_width) as usize],
    ];

    for c in 0..3 {
        trace!(channel = c, "processing color channel");
        // Normalize to [0, 1) like u/(umax + ε)
        let mut gray = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = rgb.get_pixel(x, y).0;
                let v = p[c] as f32;
                gray[(y * w + x) as usize] = v / (255.0 + config::EPSILON_GREY_LEVEL);
            }
        }

        let out_buf = match params.backend {
            Backend::CpuSingle => cpu::render_single(&gray, w, h, params)?,
            Backend::CpuMulti => cpu::render_parallel(&gray, w, h, params)?,
            #[cfg(feature = "gpu")]
            Backend::Gpu => gpu::render_gpu(&gray, w as usize, h as usize, params)?,
            #[cfg(not(feature = "gpu"))]
            Backend::Gpu => return Err(error::Error::GpuDisabled),
        };

        chan_outs[c] = out_buf;
    }

    // Recompose & scale back to 8-bit
    trace!("recomposing RGB image from channels");
    for y in 0..params.output_height {
        for x in 0..params.output_width {
            let idx = (y * params.output_width + x) as usize;
            let r =
                (chan_outs[0][idx] * (255.0 + config::EPSILON_GREY_LEVEL)).clamp(0.0, 255.0) as u8;
            let g =
                (chan_outs[1][idx] * (255.0 + config::EPSILON_GREY_LEVEL)).clamp(0.0, 255.0) as u8;
            let b =
                (chan_outs[2][idx] * (255.0 + config::EPSILON_GREY_LEVEL)).clamp(0.0, 255.0) as u8;
            out.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    info!("film grain render completed");
    Ok(DynamicImage::ImageRgb8(out))
}
