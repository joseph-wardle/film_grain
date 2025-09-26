//! Realistic Film Grain Rendering
//!
//! This library implements a film grain rendering algorithm based on the physically–motivated
//! model described in Newson et al. (2017). Two implementations are provided – a pixel–wise
//! approach and a grain–wise approach – with an automatic selection mechanism.

pub mod cli;
pub mod io;
mod prng;
pub mod rendering;

use std::sync::Once;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

/// Initializes global tracing subscribers for the crate.
///
/// When the environment variable is absent the default filter enables `INFO` level
/// logging for this crate and keeps everything else at `WARN` or above.
pub fn init_logging() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("warn,film_grain=info"));

        fmt()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .init();

        info!("Tracing subscriber initialised");
    });
}
