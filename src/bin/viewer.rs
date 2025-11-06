use std::path::{Path, PathBuf};
use std::time::Instant;

use eframe::egui;

use film_grain::{
    Algo, ColorMode, Device, InputImage, MaxRadius, ParamsBuilder, RadiusDist, RenderStats, Roi,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Film Grain Viewer",
        options,
        Box::new(|cc| Ok(Box::new(FilmGrainViewer::new(cc)))),
    )
}

struct FilmGrainViewer {
    state: ViewerState,
}

impl FilmGrainViewer {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: ViewerState::default(),
        }
    }
}

impl eframe::App for FilmGrainViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("viewer_toolbar").show(ctx, |ui| {
            ui.heading("Film Grain Viewer");
            if let Some(source) = &self.state.source {
                ui.label(format!(
                    "Source: {} ({} × {})",
                    source.path.display(),
                    source.width,
                    source.height
                ));
                let builder = self.state.params.to_builder(source);
                ui.label(format!("Color mode: {:?}", source.cache.color_mode()));
                ui.label(format!("Output path: {}", builder.output_path.display()));
            } else {
                ui.label("No image loaded");
            }
            ui.label(format!("Worker: {}", self.state.worker.status_text()));
            if let Some(path) = &self.state.pending_save_path {
                ui.label(format!("Pending save: {}", path.display()));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Preview");
            if let Some(texture) = self.state.preview.texture.as_ref() {
                let size = texture.size_vec2();
                ui.add(egui::Image::new((texture.id(), size)));
                ui.label(format!("Preview size: {:.0} × {:.0}", size.x, size.y));
            } else {
                ui.label("Preview not available");
            }
            ui.separator();
            ui.heading("Parameters");
            self.state.params.show(ui);
            ui.separator();
            ui.heading("Render stats");
            if let Some(stats) = &self.state.last_stats {
                ui.label(stats_summary(stats));
            } else {
                ui.label("No render has completed yet.");
            }
        });
    }
}

#[derive(Default)]
struct ViewerState {
    source: Option<SourceImage>,
    params: InteractiveParams,
    last_stats: Option<RenderStats>,
    preview: PreviewState,
    worker: WorkerState,
    pending_save_path: Option<PathBuf>,
}

struct SourceImage {
    path: PathBuf,
    cache: InputImage,
    width: usize,
    height: usize,
}

#[derive(Default)]
struct PreviewState {
    texture: Option<egui::TextureHandle>,
}

struct InteractiveParams {
    radius_dist: RadiusDist,
    radius_mean: f32,
    radius_stddev: f32,
    zoom: f32,
    sigma_px: f32,
    n_samples: u32,
    algo: Algo,
    max_radius: MaxRadius,
    cell_delta: Option<f32>,
    color_mode: ColorMode,
    roi: Option<Roi>,
    size: Option<(u32, Option<u32>)>,
    seed: u64,
    dry_run: bool,
    explain: bool,
    device: Device,
    output_path: Option<PathBuf>,
    output_format: Option<String>,
}

impl Default for InteractiveParams {
    fn default() -> Self {
        Self {
            radius_dist: RadiusDist::Const,
            radius_mean: 0.10,
            radius_stddev: 0.00,
            zoom: 1.0,
            sigma_px: 0.8,
            n_samples: 32,
            algo: Algo::Auto,
            max_radius: MaxRadius::Quantile(0.999),
            cell_delta: None,
            color_mode: ColorMode::Luma,
            roi: None,
            size: None,
            seed: 5489,
            dry_run: false,
            explain: false,
            device: Device::Cpu,
            output_path: None,
            output_format: None,
        }
    }
}

impl InteractiveParams {
    fn to_builder(&self, source: &SourceImage) -> ParamsBuilder {
        ParamsBuilder {
            input_path: source.path.clone(),
            output_path: self
                .output_path
                .clone()
                .unwrap_or_else(|| default_output_path(&source.path)),
            radius_dist: self.radius_dist,
            radius_mean: self.radius_mean,
            radius_stddev: self.radius_stddev,
            zoom: self.zoom,
            sigma_px: self.sigma_px,
            n_samples: self.n_samples,
            algo: self.algo,
            max_radius: self.max_radius,
            cell_delta: self.cell_delta,
            color_mode: self.color_mode,
            roi: self.roi,
            size: self.size,
            seed: self.seed,
            dry_run: self.dry_run,
            explain: self.explain,
            device: self.device,
            output_format: self.output_format.clone(),
        }
    }

    fn show(&self, ui: &mut egui::Ui) {
        ui.label(format!("Radius distribution: {:?}", self.radius_dist));
        ui.label(format!("Radius mean: {:.3}", self.radius_mean));
        ui.label(format!("Radius stddev: {:.3}", self.radius_stddev));
        ui.label(format!("Zoom: {:.3}", self.zoom));
        ui.label(format!("Sigma (px): {:.3}", self.sigma_px));
        ui.label(format!("Samples: {}", self.n_samples));
        ui.label(format!("Algorithm: {:?}", self.algo));
        ui.label(format!("Device: {:?}", self.device));
        ui.label(format!(
            "Max radius: {}",
            match self.max_radius {
                MaxRadius::Absolute(v) => format!("absolute {:.3}", v),
                MaxRadius::Quantile(q) => format!("quantile {:.3}", q),
            }
        ));
        ui.label(format!(
            "Cell delta: {}",
            self.cell_delta
                .map(|d| format!("{:.3}", d))
                .unwrap_or_else(|| "auto".to_owned())
        ));
        ui.label(format!("Color mode: {:?}", self.color_mode));
        if let Some(roi) = self.roi {
            ui.label(format!("ROI: {} {} {} {}", roi.x0, roi.y0, roi.x1, roi.y1));
        } else {
            ui.label("ROI: full image");
        }
        if let Some((width, maybe_height)) = self.size {
            let text = maybe_height
                .map(|h| format!("{} × {}", width, h))
                .unwrap_or_else(|| format!("width {}", width));
            ui.label(format!("Output size override: {text}"));
        } else {
            ui.label("Output size override: none");
        }
        ui.label(format!("Seed: {}", self.seed));
        ui.label(format!("Dry run: {}", self.dry_run));
        ui.label(format!("Explain: {}", self.explain));
        if let Some(fmt) = &self.output_format {
            ui.label(format!("Output format override: {fmt}"));
        } else {
            ui.label("Output format override: auto from path");
        }
    }
}

#[derive(Default)]
struct WorkerState {
    status: WorkerStatus,
}

impl WorkerState {
    fn status_text(&self) -> String {
        match &self.status {
            WorkerStatus::Idle => "idle".to_owned(),
            WorkerStatus::Rendering { started_at } => {
                format!("rendering (started {}s ago)", elapsed_seconds(*started_at))
            }
            WorkerStatus::Completed { finished_at } => {
                format!("completed {}s ago", elapsed_seconds(*finished_at))
            }
            WorkerStatus::Failed { message } => format!("failed: {message}"),
        }
    }
}

#[allow(dead_code)]
enum WorkerStatus {
    Idle,
    Rendering { started_at: Instant },
    Completed { finished_at: Instant },
    Failed { message: String },
}

impl Default for WorkerStatus {
    fn default() -> Self {
        WorkerStatus::Idle
    }
}

fn elapsed_seconds(instant: Instant) -> u64 {
    instant.elapsed().as_secs()
}

fn stats_summary(stats: &RenderStats) -> String {
    format!(
        "Algorithm: {:?}, device: {:?}, output: {} × {}, samples: {}, σ/mean {:.3}, rm/mean {:.3}",
        stats.algorithm,
        stats.device,
        stats.output_size.0,
        stats.output_size.1,
        stats.n_samples,
        stats.sigma_ratio,
        stats.rm_ratio
    )
}

fn default_output_path(source: &Path) -> PathBuf {
    let mut result = source.to_path_buf();
    result.set_extension("filmgrain.png");
    result
}
