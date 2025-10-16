//! CPU rendering (reference single-thread + Rayon multithread).
pub fn render_single(
    gray01: &[f32], w: usize, h: usize, params: &FilmGrainParams
) -> Result<Vec<f32>> {
}

pub fn render_parallel(
    gray01: &[f32], w: usize, h: usize, params: &FilmGrainParams
) -> Result<Vec<f32>> {
}
