//! Realistic Film Grain Rendering
//!
//! This library implements a film grain rendering algorithm based on the physically–motivated
//! model described in Newson et al. (2017). Two implementations are provided – a pixel–wise
//! approach and a grain–wise approach – with an automatic selection mechanism.

pub mod cli;
pub mod io;
mod prng;
pub mod rendering;

use std::fs;
use std::path::PathBuf;
use std::sync::{Once, OnceLock};

use tracing::info;
use tracing_appender::non_blocking::{self, WorkerGuard};
use tracing_subscriber::{
    filter::LevelFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

pub fn init_logging() {
    static INIT: Once = Once::new();
    static GUARD: OnceLock<WorkerGuard> = OnceLock::new();

    INIT.call_once(|| {
        // Ensure logs/ exists
        let mut path = PathBuf::from("logs");
        let _ = fs::create_dir_all(&path);

        // Unique file per run: app-YYYY-mm-dd_HH-MM-SS-PID.log
        let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let pid = std::process::id();
        path.push(format!("app-{ts}-{pid}.log"));

        // Non-blocking file writer
        let file = fs::File::create(&path)
            .unwrap_or_else(|e| panic!("failed to create log file {}: {e}", path.display()));
        let (file_writer, guard) = non_blocking::non_blocking(file);
        let _ = GUARD.set(guard); // keep worker thread alive for the program lifetime

        // Console layer: INFO and above
        let console_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_ansi(true)
            .with_writer(std::io::stdout)  // print to console
            .with_filter(LevelFilter::INFO);

        // File layer: TRACE and above (everything)
        let file_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_ansi(false)               // no color codes in file
            .with_writer(file_writer)       // write to our file
            .with_filter(LevelFilter::TRACE);

        tracing_subscriber::registry()
            .with(console_layer)
            .with(file_layer)
            .init();

        info!("Tracing initialized. Console: INFO+. File (TRACE+): {}", path.display());
    });
}
