use crate::{config::*, error::*};
use tracing::instrument;

/// Evaluate v(y) = (1/N) Σ 1_Z((y - ξ_k)/s) for each output pixel y.
/// Locally generate Poisson grains for overlapping cells, using hashed cell seeds.
/// See Algorithm 2/3 in the paper.
#[instrument(skip(gray01, params, lambda_lut, exp_lut))]
#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
pub fn render_pixelwise_cpu(
    gray01: &[f32],
    w: u32,
    h: u32,
    params: &FilmGrainParams,
    lambda_lut: &[f32],
    exp_lut: &[f32],
    _threads: Option<usize>,
) -> Result<Vec<f32>> {
    todo!()
}
