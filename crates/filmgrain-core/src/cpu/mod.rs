//! CPU rendering (reference single-thread + Rayon multithread).

use crate::FilmGrainParams;
use crate::error::Result;

#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
pub fn render_single(
    gray01: &[f32],
    w: usize,
    h: usize,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    todo!()
}

#[expect(unused_variables, reason = "scaffolding signature; impl coming")]
pub fn render_parallel(
    gray01: &[f32],
    w: usize,
    h: usize,
    params: &FilmGrainParams,
) -> Result<Vec<f32>> {
    todo!()
}
