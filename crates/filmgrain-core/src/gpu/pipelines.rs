use crate::FilmGrainParams;
use crate::error::Result;
use tracing::instrument;

#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
#[instrument(skip(gray01, params))]
pub fn render_gpu(
    gray01: &[f32],
    w: usize,
    h: usize,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    todo!()
}
