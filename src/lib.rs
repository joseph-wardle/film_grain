//! Realistic Film Grain Rendering
//!
//! This library implements a film grain rendering algorithm based on the physically–motivated
//! model described in Newson et al. (2017). Two implementations are provided – a pixel–wise
//! approach and a grain–wise approach – with an automatic selection mechanism.

pub mod cli;
pub mod io;
mod prng;
pub mod rendering;
