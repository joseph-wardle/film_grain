//! CPU rendering (reference single-thread + Rayon multithread).

use tracing::{debug, instrument, warn};

use crate::FilmGrainParams;
use crate::error::Result;

#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
#[instrument(level = "debug", skip(gray01, params))]
pub fn render_single(
    gray01: &[f32],
    w: usize,
    h: usize,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    debug!(width = w, height = h, "CPU single-thread renderer invoked");
    warn!("CPU single-thread renderer not yet implemented");
    todo!()
}

#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
pub fn render_parallel(
    gray01: &[f32],
    w: usize,
    h: usize,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    debug!(width = w, height = h, "CPU parallel renderer invoked");
    warn!("CPU parallel renderer not yet implemented");
    todo!()
}
