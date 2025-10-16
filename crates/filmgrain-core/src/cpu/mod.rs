//! CPU rendering (reference single-thread + Rayon multithread).

mod grainwise;
mod pixelwise;

use tracing::instrument;

use crate::config::choose_mode;
use crate::error::Result;
use crate::util::precompute_lambda_tables;
use crate::{FilmGrainMode, FilmGrainParams};

#[instrument(level = "debug", skip(gray01, params))]
pub fn render_single(gray01: &[f32], w: u32, h: u32, params: &FilmGrainParams) -> Result<Vec<f32>> {
    let mut p = params.clone();
    p.finalize(w, h);

    let mode = match p.mode {
        FilmGrainMode::Auto => choose_mode(p.grain_size, p.grain_size_std_dev),
        m => m,
    };

    let (lambda_lut, exp_lut) = precompute_lambda_tables(p.grain_size, p.grain_size_std_dev);

    let out = match mode {
        FilmGrainMode::PixelWise => pixelwise::render_pixelwise_cpu(
            gray01,
            w,
            h,
            &p,
            &lambda_lut,
            &exp_lut,
            /*threads*/ None,
        ),
        FilmGrainMode::GrainWise => {
            grainwise::render_grainwise_cpu(gray01, w, h, &p, &lambda_lut, /*threads*/ None)
        }
        FilmGrainMode::Auto => unreachable!(),
    }?;
    Ok(out)
}

pub fn render_parallel(
    gray01: &[f32],
    w: u32,
    h: u32,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    let mut p = params.clone();
    p.finalize(w, h);

    // Optional threads
    if let Some(t) = p.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }

    let mode = match p.mode {
        FilmGrainMode::Auto => choose_mode(p.grain_size, p.grain_size_std_dev),
        m => m,
    };
    let (lambda_lut, exp_lut) = precompute_lambda_tables(p.grain_size, p.grain_size_std_dev);

    let out = match mode {
        FilmGrainMode::PixelWise => pixelwise::render_pixelwise_cpu(
            gray01,
            w,
            h,
            &p,
            &lambda_lut,
            &exp_lut,
            /*threads*/ p.threads,
        ),
        FilmGrainMode::GrainWise => grainwise::render_grainwise_cpu(
            gray01,
            w,
            h,
            &p,
            &lambda_lut,
            /*threads*/ p.threads,
        ),
        FilmGrainMode::Auto => unreachable!(),
    }?;
    Ok(out)
}
