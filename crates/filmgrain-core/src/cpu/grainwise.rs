use crate::{config::*, error::*};
use tracing::instrument;

/// Grain-wise: draw grains per input pixel, rasterize onto N binary images v_k, then average.
/// Mirrors Algorithm 1 in the paper.
#[instrument(skip(gray01, params, lambda_lut))]
#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
pub fn render_grainwise_cpu(
    gray01: &[f32],
    w: u32,
    h: u32,
    params: &FilmGrainParams,
    lambda_lut: &[f32],
    _threads: Option<usize>,
) -> Result<Vec<f32>> {
    todo!()
}
