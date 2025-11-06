use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use eframe::egui;

use film_grain::{
    Algo, ColorMode, Device, InputImage, MaxRadius, Params, ParamsBuilder, RadiusDist, RenderError,
    RenderStats, Roi, render_with_input_image,
};
use image::RgbImage;

fn main() -> eframe::Result<()> {
    let mut options = eframe::NativeOptions::default();
    options.renderer = eframe::Renderer::Glow;
    options.follow_system_theme = false;
    options.default_theme = eframe::Theme::Dark;
    options.viewport = egui::viewport::ViewportBuilder::default()
        .with_inner_size([1280.0, 800.0])
        .with_title("Film Grain Viewer");
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
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        cc.egui_ctx.set_pixels_per_point(1.0);
        Self {
            state: ViewerState::new(),
        }
    }
}

impl eframe::App for FilmGrainViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.state.poll_worker(ctx);
        self.state.draw(ctx);
    }
}

struct ViewerState {
    source: Option<SourceImage>,
    params: InteractiveParams,
    last_stats: Option<RenderStats>,
    preview: PreviewState,
    worker_status: WorkerState,
    pending_save_path: Option<PathBuf>,
    worker_runtime: RenderWorker,
}

impl ViewerState {
    fn new() -> Self {
        Self {
            source: None,
            params: InteractiveParams::default(),
            last_stats: None,
            preview: PreviewState::default(),
            worker_status: WorkerState::default(),
            pending_save_path: None,
            worker_runtime: RenderWorker::new(),
        }
    }

    fn draw(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("viewer_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Film Grain Viewer");
                ui.separator();
                ui.label(self.top_bar_text());
                ui.separator();
                ui.label(self.worker_status.status_text());
            });
            if let Some(path) = &self.pending_save_path {
                ui.label(format!("Pending save target: {}", path.display()));
            }
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .min_width(260.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.params.show(ui);
                    ui.separator();
                    if let Some(stats) = &self.last_stats {
                        ui.heading("Last render stats");
                        ui.label(stats_summary(stats));
                    } else {
                        ui.heading("Last render stats");
                        ui.label("No render has completed yet.");
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Preview");
            });
            ui.separator();
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                ui.set_min_height(ui.available_height());
                ui.set_min_width(ui.available_width());
                ui.centered_and_justified(|ui| {
                    if let Some(texture) = self.preview.texture.as_ref() {
                        let size = texture.size_vec2();
                        ui.image((texture.id(), size));
                    } else {
                        ui.label("No preview yet – load an image to begin.");
                    }
                });
            });
        });
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        let mut updated = false;
        while let Some(result) = self.worker_runtime.try_recv() {
            if self.worker_status.active_job != Some(result.id) {
                continue;
            }
            match result.outcome {
                Ok(success) => {
                    self.last_stats = Some(success.stats);
                    self.preview.set_image(ctx, success.image);
                    self.worker_status.complete();
                }
                Err(err) => {
                    self.worker_status.fail(err.to_string());
                }
            }
            updated = true;
        }
        if updated {
            ctx.request_repaint();
        }
    }

    fn top_bar_text(&self) -> String {
        if let Some(source) = &self.source {
            format!(
                "{} ({} × {}, {:?})",
                source.path.display(),
                source.width,
                source.height,
                source.cache.color_mode()
            )
        } else {
            "No image loaded".to_owned()
        }
    }

    #[allow(dead_code)]
    fn request_render(&mut self) -> Result<(), String> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| "no source loaded".to_owned())?;
        let builder = self.params.to_builder(source);
        let params = builder.build().map_err(|err| err.to_string())?;
        let job_id = self
            .worker_runtime
            .request(source.cache.clone(), params)
            .map_err(|err| err)?;
        self.worker_status.begin_render(job_id);
        Ok(())
    }
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
    generation: u64,
}

impl PreviewState {
    fn set_image(&mut self, ctx: &egui::Context, image: RgbImage) {
        let color_image = color_image_from_rgb(image);
        let name = format!("filmgrain-preview-{}", self.generation);
        self.generation = self.generation.wrapping_add(1);
        self.texture = Some(ctx.load_texture(name, color_image, egui::TextureOptions::default()));
    }
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

struct RenderWorker {
    job_tx: mpsc::Sender<RenderJob>,
    result_rx: mpsc::Receiver<RenderOutcome>,
    latest_request: Arc<AtomicU64>,
    next_job_id: u64,
}

impl RenderWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<RenderJob>();
        let (result_tx, result_rx) = mpsc::channel::<RenderOutcome>();
        let latest_request = Arc::new(AtomicU64::new(0));
        let worker_latest = latest_request.clone();
        thread::Builder::new()
            .name("film-grain-render".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let current = worker_latest.load(Ordering::SeqCst);
                    if job.id != current {
                        continue;
                    }
                    let outcome = render_with_input_image(&job.input, &job.params)
                        .map(|(image, stats)| RenderSuccess { image, stats });
                    if job.id != worker_latest.load(Ordering::SeqCst) {
                        continue;
                    }
                    if result_tx
                        .send(RenderOutcome {
                            id: job.id,
                            outcome,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("spawn render worker");

        Self {
            job_tx,
            result_rx,
            latest_request,
            next_job_id: 1,
        }
    }

    fn request(&mut self, input: InputImage, params: Params) -> Result<u64, String> {
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        if self.next_job_id == 0 {
            self.next_job_id = 1;
        }
        self.latest_request.store(id, Ordering::SeqCst);
        let job = RenderJob { id, input, params };
        self.job_tx.send(job).map_err(|err| err.to_string())?;
        Ok(id)
    }

    fn try_recv(&self) -> Option<RenderOutcome> {
        self.result_rx.try_recv().ok()
    }
}

struct RenderJob {
    id: u64,
    input: InputImage,
    params: Params,
}

struct RenderOutcome {
    id: u64,
    outcome: Result<RenderSuccess, RenderError>,
}

struct RenderSuccess {
    image: RgbImage,
    stats: RenderStats,
}

fn color_image_from_rgb(image: RgbImage) -> egui::ColorImage {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let raw = image.into_raw();
    let mut pixels = Vec::with_capacity(width * height);
    for chunk in raw.chunks_exact(3) {
        pixels.push(egui::Color32::from_rgb(chunk[0], chunk[1], chunk[2]));
    }
    egui::ColorImage {
        size: [width, height],
        pixels,
    }
}

#[derive(Default)]
struct WorkerState {
    status: WorkerStatus,
    active_job: Option<u64>,
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

    fn begin_render(&mut self, job_id: u64) {
        self.active_job = Some(job_id);
        self.status = WorkerStatus::Rendering {
            started_at: Instant::now(),
        };
    }

    fn complete(&mut self) {
        self.active_job = None;
        self.status = WorkerStatus::Completed {
            finished_at: Instant::now(),
        };
    }

    fn fail(&mut self, message: String) {
        self.active_job = None;
        self.status = WorkerStatus::Failed { message };
    }

    #[allow(dead_code)]
    fn is_busy(&self) -> bool {
        matches!(self.status, WorkerStatus::Rendering { .. })
    }
}

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
