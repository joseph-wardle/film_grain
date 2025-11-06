use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui;

use film_grain::{
    Algo, ColorMode, Device, InputImage, MaxRadius, Params, ParamsBuilder, RadiusDist, RenderError,
    RenderStats, Roi, default_cell_delta, dry_run_with_input_image, render_with_input_image,
};
use image::RgbImage;
use rand::Rng;
use rfd::FileDialog;

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
    last_error: Option<String>,
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
            last_error: None,
        }
    }

    fn draw(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("viewer_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Load Image…").clicked() {
                    self.prompt_and_load();
                }
                let can_save = self.preview.has_image();
                if ui
                    .add_enabled(can_save, egui::Button::new("Save Preview…"))
                    .clicked()
                {
                    self.prompt_and_save();
                }
                if ui.button("Render Now").clicked() {
                    let mode = if self.params.dry_run_mode {
                        RenderMode::StatsOnly
                    } else {
                        RenderMode::Preview
                    };
                    if let Err(err) = self.schedule_render(mode) {
                        self.set_error(err);
                    }
                }
                ui.separator();
                ui.heading("Film Grain Viewer");
                ui.separator();
                ui.label(self.top_bar_text());
                ui.separator();
                ui.label(self.worker_status.status_text());
            });
            if let Some(err) = &self.last_error {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("Error: {err}"));
            }
            if let Some(path) = &self.pending_save_path {
                ui.label(format!("Last saved: {}", path.display()));
            }
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .min_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let change = self.params.show(
                        ui,
                        self.source.as_ref(),
                        self.worker_runtime.gpu_available(),
                    );
                    self.apply_param_change(change);
                    ui.separator();
                    ui.heading("Last render stats");
                    if let Some(stats) = &self.last_stats {
                        ui.label(stats_summary(stats));
                        if let Some(duration) = self.worker_status.last_duration() {
                            ui.label(format!("Last run time: {:.2}s", duration.as_secs_f32()));
                        }
                    } else {
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
                        ui.label(format!(
                            "{} × {} px",
                            size.x.round() as u32,
                            size.y.round() as u32
                        ));
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
                Ok(JobResult::Preview { image, stats }) => {
                    self.last_stats = Some(stats);
                    self.preview.set_image(ctx, image);
                    self.worker_status.complete();
                    self.clear_error();
                }
                Ok(JobResult::Stats { stats }) => {
                    self.last_stats = Some(stats);
                    self.worker_status.complete();
                    self.clear_error();
                }
                Err(err) => {
                    let message = err.to_string();
                    self.worker_status.fail(message.clone());
                    self.set_error(message);
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

    fn apply_param_change(&mut self, change: ParamChange) {
        if !change.require_reload && !change.request_preview && !change.request_stats_only {
            return;
        }
        if self.source.is_none() {
            return;
        }
        if change.require_reload
            && let Err(err) = self.reload_source_from_params()
        {
            self.set_error(err);
            return;
        }
        if change.request_preview {
            if let Err(err) = self.schedule_render(RenderMode::Preview) {
                self.set_error(err);
            }
        } else if change.request_stats_only
            && let Err(err) = self.schedule_render(RenderMode::StatsOnly)
        {
            self.set_error(err);
        }
    }

    fn schedule_render(&mut self, mode: RenderMode) -> Result<(), String> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| "no image loaded".to_owned())?;
        let mut params = self
            .params
            .to_builder(source, matches!(mode, RenderMode::StatsOnly))
            .build()
            .map_err(|err| err.to_string())?;
        if matches!(mode, RenderMode::StatsOnly) {
            params.dry_run = true;
        }
        let kind = match mode {
            RenderMode::Preview => RenderTaskKind::Preview,
            RenderMode::StatsOnly => RenderTaskKind::StatsOnly,
        };
        let job_id = self
            .worker_runtime
            .request(source.cache.clone(), params, kind)?;
        self.worker_status.begin_render(job_id);
        Ok(())
    }

    fn reload_source_from_params(&mut self) -> Result<(), String> {
        let source = self
            .source
            .as_mut()
            .ok_or_else(|| "no image loaded".to_owned())?;
        let roi = self.params.current_roi();
        let cache = InputImage::from_path(&source.path, self.params.color_mode, roi.as_ref())
            .map_err(|err| err.to_string())?;
        let cache = Arc::new(cache);
        source.set_cache(cache.clone());
        self.params.sync_with_source(source);
        self.preview.clear();
        self.last_stats = None;
        self.worker_status.reset();
        Ok(())
    }

    fn prompt_and_load(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter(
                "Images",
                &["png", "jpg", "jpeg", "tif", "tiff", "bmp", "hdr", "gif"],
            )
            .pick_file()
        {
            if let Err(err) = self.load_image_from_path(path.clone()) {
                self.set_error(err);
            } else {
                self.pending_save_path = None;
            }
        }
    }

    fn load_image_from_path(&mut self, path: PathBuf) -> Result<(), String> {
        let cache = InputImage::from_path(&path, self.params.color_mode, None)
            .map_err(|err| err.to_string())?;
        let cache = Arc::new(cache);
        let source = SourceImage::new(path.clone(), cache.clone());
        self.params.sync_with_source(&source);
        self.params.update_output_defaults(&path);
        self.source = Some(source);
        self.preview.clear();
        self.last_stats = None;
        self.worker_status.reset();
        self.clear_error();
        let mode = if self.params.dry_run_mode {
            RenderMode::StatsOnly
        } else {
            RenderMode::Preview
        };
        self.schedule_render(mode)?;
        Ok(())
    }

    fn prompt_and_save(&mut self) {
        let Some(image) = self.preview.latest_image() else {
            self.set_error("no preview image available to save".to_owned());
            return;
        };
        let default_name = self
            .source
            .as_ref()
            .map(|source| self.params.resolve_output_path(&source.path))
            .and_then(|path| {
                path.file_name()
                    .and_then(|s| s.to_str().map(|t| t.to_owned()))
            })
            .unwrap_or_else(|| "filmgrain.png".to_owned());
        if let Some(path) = FileDialog::new()
            .set_file_name(&default_name)
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
            .save_file()
        {
            if let Err(err) = image.clone().save(&path) {
                self.set_error(err.to_string());
            } else {
                self.pending_save_path = Some(path);
                self.clear_error();
            }
        }
    }

    fn set_error(&mut self, message: String) {
        self.last_error = Some(message);
    }

    fn clear_error(&mut self) {
        self.last_error = None;
    }
}

struct SourceImage {
    path: PathBuf,
    cache: Arc<InputImage>,
    width: usize,
    height: usize,
}

impl SourceImage {
    fn new(path: PathBuf, cache: Arc<InputImage>) -> Self {
        let (width, height) = cache.dimensions();
        Self {
            path,
            cache,
            width,
            height,
        }
    }

    fn set_cache(&mut self, cache: Arc<InputImage>) {
        let (width, height) = cache.dimensions();
        self.cache = cache;
        self.width = width;
        self.height = height;
    }
}

#[derive(Default)]
struct PreviewState {
    texture: Option<egui::TextureHandle>,
    generation: u64,
    last_image: Option<RgbImage>,
}

impl PreviewState {
    fn set_image(&mut self, ctx: &egui::Context, image: RgbImage) {
        self.last_image = Some(image.clone());
        let color_image = color_image_from_rgb(image);
        let name = format!("filmgrain-preview-{}", self.generation);
        self.generation = self.generation.wrapping_add(1);
        self.texture = Some(ctx.load_texture(name, color_image, egui::TextureOptions::default()));
    }

    fn has_image(&self) -> bool {
        self.last_image.is_some()
    }

    fn latest_image(&self) -> Option<&RgbImage> {
        self.last_image.as_ref()
    }

    fn clear(&mut self) {
        self.texture = None;
        self.last_image = None;
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
    cell_auto: bool,
    cell_value: f32,
    color_mode: ColorMode,
    roi_enabled: bool,
    roi_x0: u32,
    roi_y0: u32,
    roi_x1: u32,
    roi_y1: u32,
    size_enabled: bool,
    size_width: u32,
    size_height_enabled: bool,
    size_height: u32,
    seed: u64,
    dry_run_mode: bool,
    explain: bool,
    device: Device,
    output_path_override: String,
    output_format_override: String,
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
            cell_auto: true,
            cell_value: 0.1,
            color_mode: ColorMode::Luma,
            roi_enabled: false,
            roi_x0: 0,
            roi_y0: 0,
            roi_x1: 1,
            roi_y1: 1,
            size_enabled: false,
            size_width: 1,
            size_height_enabled: false,
            size_height: 1,
            seed: 5489,
            dry_run_mode: false,
            explain: false,
            device: Device::Cpu,
            output_path_override: String::new(),
            output_format_override: String::new(),
        }
    }
}

impl InteractiveParams {
    fn to_builder(&self, source: &SourceImage, dry_run: bool) -> ParamsBuilder {
        ParamsBuilder {
            input_path: source.path.clone(),
            output_path: self.resolve_output_path(&source.path),
            radius_dist: self.radius_dist,
            radius_mean: self.radius_mean,
            radius_stddev: self.radius_stddev,
            zoom: self.zoom,
            sigma_px: self.sigma_px,
            n_samples: self.n_samples,
            algo: self.algo,
            max_radius: self.max_radius,
            cell_delta: self.current_cell_delta(),
            color_mode: self.color_mode,
            roi: self.current_roi(),
            size: self.current_size_override(),
            seed: self.seed,
            dry_run,
            explain: self.explain,
            device: self.device,
            output_format: {
                let trimmed = self.output_format_override.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_owned())
                }
            },
        }
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        source: Option<&SourceImage>,
        gpu_available: bool,
    ) -> ParamChange {
        let mut change = ParamChange::new(self.dry_run_mode);

        ui.heading("Model");
        if ui
            .add(
                egui::Slider::new(&mut self.radius_mean, 0.01..=5.0)
                    .logarithmic(true)
                    .text("Mean radius (px)"),
            )
            .changed()
        {
            if self.cell_auto {
                self.cell_value = default_cell_delta(self.radius_mean);
            }
            change.mark_changed();
        }

        let radius_before = self.radius_dist;
        egui::ComboBox::from_label("Radius distribution")
            .selected_text(match self.radius_dist {
                RadiusDist::Const => "Const",
                RadiusDist::Lognorm => "Log-normal",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.radius_dist, RadiusDist::Const, "Const");
                ui.selectable_value(&mut self.radius_dist, RadiusDist::Lognorm, "Log-normal");
            });
        if self.radius_dist != radius_before {
            if self.radius_dist == RadiusDist::Const {
                self.radius_stddev = 0.0;
            }
            change.mark_changed();
        }

        if ui
            .add_enabled(
                matches!(self.radius_dist, RadiusDist::Lognorm),
                egui::Slider::new(&mut self.radius_stddev, 0.0..=2.0)
                    .text("Radius stddev (px)")
                    .logarithmic(true),
            )
            .changed()
        {
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Sampling");
        if ui
            .add(egui::Slider::new(&mut self.zoom, 0.25..=4.0).text("Zoom (output scale)"))
            .changed()
        {
            change.mark_changed();
        }
        if ui
            .add(
                egui::Slider::new(&mut self.sigma_px, 0.1..=5.0)
                    .logarithmic(true)
                    .text("Sigma (px)"),
            )
            .changed()
        {
            change.mark_changed();
        }
        if ui
            .add(
                egui::Slider::new(&mut self.n_samples, 1..=512)
                    .text("Samples")
                    .logarithmic(true),
            )
            .changed()
        {
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Algorithm");
        let algo_before = self.algo;
        egui::ComboBox::from_label("Algorithm override")
            .selected_text(format!("{:?}", self.algo))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.algo, Algo::Auto, "Auto");
                ui.selectable_value(&mut self.algo, Algo::Pixel, "Pixel-wise");
                ui.selectable_value(&mut self.algo, Algo::Grain, "Grain-wise");
            });
        if self.algo != algo_before {
            change.mark_changed();
        }

        enum MaxSetting {
            Absolute(f32),
            Quantile(f32),
        }
        let mut max_setting = match self.max_radius {
            MaxRadius::Absolute(value) => MaxSetting::Absolute(value),
            MaxRadius::Quantile(prob) => MaxSetting::Quantile(prob),
        };
        let mut max_changed = false;
        let mut new_setting: Option<MaxSetting> = None;
        match &mut max_setting {
            MaxSetting::Absolute(value) => {
                ui.horizontal(|ui| {
                    ui.label("Max radius (absolute)");
                    max_changed |= ui
                        .add(egui::DragValue::new(value).range(0.01..=10.0).speed(0.05))
                        .changed();
                    if ui.button("Switch to quantile").clicked() {
                        new_setting = Some(MaxSetting::Quantile(0.999));
                    }
                });
            }
            MaxSetting::Quantile(prob) => {
                ui.horizontal(|ui| {
                    ui.label("Max radius quantile");
                    max_changed |= ui
                        .add(
                            egui::Slider::new(prob, 0.5..=0.99999)
                                .logarithmic(true)
                                .text("Quantile"),
                        )
                        .changed();
                    if ui.button("Switch to absolute").clicked() {
                        new_setting = Some(MaxSetting::Absolute(1.0));
                    }
                });
            }
        }
        if let Some(setting) = new_setting {
            max_setting = setting;
            max_changed = true;
        }
        if max_changed {
            self.max_radius = match max_setting {
                MaxSetting::Absolute(value) => MaxRadius::Absolute(value.max(0.01)),
                MaxSetting::Quantile(prob) => MaxRadius::Quantile(prob.clamp(0.5, 0.99999)),
            };
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Cell size");
        let cell_auto_prev = self.cell_auto;
        if ui
            .checkbox(&mut self.cell_auto, "Automatic (≈ μᵣ)")
            .changed()
        {
            if self.cell_auto {
                self.cell_value = default_cell_delta(self.radius_mean);
            }
            change.mark_changed();
        }
        if !self.cell_auto
            && ui
                .add(
                    egui::Slider::new(&mut self.cell_value, 0.01..=5.0)
                        .logarithmic(true)
                        .text("Manual cell size"),
                )
                .changed()
        {
            change.mark_changed();
        }
        if cell_auto_prev && !self.cell_auto {
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Colour & ROI");
        let color_before = self.color_mode;
        egui::ComboBox::from_label("Colour mode")
            .selected_text(match self.color_mode {
                ColorMode::Luma => "Luma (Y only)",
                ColorMode::Rgb => "RGB (per channel)",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.color_mode, ColorMode::Luma, "Luma (Y only)");
                ui.selectable_value(&mut self.color_mode, ColorMode::Rgb, "RGB (per channel)");
            });
        if self.color_mode != color_before {
            change.mark_reload();
        }

        ui.collapsing("Region of interest", |ui| {
            if ui.checkbox(&mut self.roi_enabled, "Enable ROI").changed() {
                change.mark_reload();
            }
            if self.roi_enabled {
                let mut roi_changed = false;
                roi_changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.roi_x0)
                            .range(0..=u32::MAX)
                            .prefix("x₀ "),
                    )
                    .changed();
                roi_changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.roi_y0)
                            .range(0..=u32::MAX)
                            .prefix("y₀ "),
                    )
                    .changed();
                roi_changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.roi_x1)
                            .range(1..=u32::MAX)
                            .prefix("x₁ "),
                    )
                    .changed();
                roi_changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.roi_y1)
                            .range(1..=u32::MAX)
                            .prefix("y₁ "),
                    )
                    .changed();
                if roi_changed {
                    change.mark_reload();
                }
                if let Some(source) = source {
                    self.clamp_roi_to_source(source);
                }
            }
        });

        ui.separator();
        ui.heading("Output size");
        if ui
            .checkbox(&mut self.size_enabled, "Override output size")
            .changed()
        {
            change.mark_changed();
        }
        if self.size_enabled {
            if ui
                .add(
                    egui::DragValue::new(&mut self.size_width)
                        .range(1..=16384)
                        .prefix("Width "),
                )
                .changed()
            {
                change.mark_changed();
            }
            if ui
                .checkbox(&mut self.size_height_enabled, "Set explicit height")
                .changed()
            {
                change.mark_changed();
            }
            if self.size_height_enabled
                && ui
                    .add(
                        egui::DragValue::new(&mut self.size_height)
                            .range(1..=16384)
                            .prefix("Height "),
                    )
                    .changed()
            {
                change.mark_changed();
            }
        }

        ui.separator();
        ui.heading("Rendering");
        let device_before = self.device;
        egui::ComboBox::from_label("Device")
            .selected_text(match self.device {
                Device::Cpu => "CPU",
                Device::Gpu => "GPU",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.device, Device::Cpu, "CPU");
                ui.add_enabled_ui(gpu_available, |ui| {
                    ui.selectable_value(&mut self.device, Device::Gpu, "GPU");
                });
            });
        if !gpu_available && self.device == Device::Gpu {
            self.device = Device::Cpu;
        }
        if self.device != device_before {
            change.mark_changed();
        }
        if ui.button("Reset device to CPU").clicked() {
            self.device = Device::Cpu;
            change.mark_changed();
        }
        if ui
            .checkbox(&mut self.dry_run_mode, "Dry run (stats only)")
            .changed()
        {
            change.mark_mode_switch(self.dry_run_mode);
        }
        if ui
            .checkbox(&mut self.explain, "Explain selection")
            .changed()
        {
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Randomness");
        let mut seed_changed = false;
        seed_changed |= ui
            .add(
                egui::DragValue::new(&mut self.seed)
                    .range(1..=u64::MAX)
                    .speed(64.0)
                    .prefix("Seed "),
            )
            .changed();
        if ui.button("Randomise").clicked() {
            self.seed = rand::thread_rng().r#gen::<u64>();
            seed_changed = true;
        }
        if seed_changed {
            change.mark_changed();
        }

        ui.separator();
        ui.heading("Output");
        if ui
            .text_edit_singleline(&mut self.output_path_override)
            .changed()
        {
            // Changing output path does not require a re-render.
        }
        if ui
            .text_edit_singleline(&mut self.output_format_override)
            .changed()
        {
            // Format change affects only export.
        }

        change
    }

    fn resolve_output_path(&self, source_path: &Path) -> PathBuf {
        let trimmed = self.output_path_override.trim();
        if trimmed.is_empty() {
            default_output_path(source_path)
        } else {
            PathBuf::from(trimmed)
        }
    }

    fn current_cell_delta(&self) -> Option<f32> {
        if self.cell_auto {
            None
        } else {
            Some(self.cell_value.max(0.0001))
        }
    }

    fn current_roi(&self) -> Option<Roi> {
        if self.roi_enabled && self.roi_x1 > self.roi_x0 && self.roi_y1 > self.roi_y0 {
            Some(Roi {
                x0: self.roi_x0,
                y0: self.roi_y0,
                x1: self.roi_x1,
                y1: self.roi_y1,
            })
        } else {
            None
        }
    }

    fn current_size_override(&self) -> Option<(u32, Option<u32>)> {
        if !self.size_enabled {
            return None;
        }
        let width = self.size_width.max(1);
        let height_opt = if self.size_height_enabled {
            Some(self.size_height.max(1))
        } else {
            None
        };
        Some((width, height_opt))
    }

    fn sync_with_source(&mut self, source: &SourceImage) {
        let max_width = source.width.max(1) as u32;
        let max_height = source.height.max(1) as u32;
        if !self.roi_enabled {
            self.roi_x0 = 0;
            self.roi_y0 = 0;
            self.roi_x1 = max_width;
            self.roi_y1 = max_height;
        } else {
            self.roi_x0 = self.roi_x0.min(max_width.saturating_sub(1));
            self.roi_y0 = self.roi_y0.min(max_height.saturating_sub(1));
            self.roi_x1 = self.roi_x1.clamp(self.roi_x0 + 1, max_width);
            self.roi_y1 = self.roi_y1.clamp(self.roi_y0 + 1, max_height);
        }
        if !self.size_enabled {
            self.size_width = max_width;
            self.size_height = max_height;
        } else {
            self.size_width = self.size_width.max(1);
            if self.size_height_enabled {
                self.size_height = self.size_height.max(1);
            }
        }
        if self.cell_auto {
            self.cell_value = default_cell_delta(self.radius_mean);
        }
    }

    fn clamp_roi_to_source(&mut self, source: &SourceImage) {
        let max_width = source.width.max(1) as u32;
        let max_height = source.height.max(1) as u32;
        self.roi_x0 = self.roi_x0.min(max_width.saturating_sub(1));
        self.roi_y0 = self.roi_y0.min(max_height.saturating_sub(1));
        self.roi_x1 = self.roi_x1.clamp(self.roi_x0 + 1, max_width);
        self.roi_y1 = self.roi_y1.clamp(self.roi_y0 + 1, max_height);
    }

    fn update_output_defaults(&mut self, source_path: &Path) {
        if self.output_path_override.trim().is_empty()
            && let Some(path) = default_output_path(source_path).to_str()
        {
            self.output_path_override = path.to_owned();
        }
    }
}

#[derive(Default)]
struct ParamChange {
    request_preview: bool,
    request_stats_only: bool,
    require_reload: bool,
    dry_run_mode: bool,
}

impl ParamChange {
    fn new(dry_run_mode: bool) -> Self {
        Self {
            request_preview: false,
            request_stats_only: false,
            require_reload: false,
            dry_run_mode,
        }
    }

    fn mark_changed(&mut self) {
        if self.dry_run_mode {
            self.request_stats_only = true;
            self.request_preview = false;
        } else {
            self.request_preview = true;
            self.request_stats_only = false;
        }
    }

    fn mark_reload(&mut self) {
        self.require_reload = true;
        self.mark_changed();
    }

    fn mark_mode_switch(&mut self, dry_run_mode: bool) {
        self.dry_run_mode = dry_run_mode;
        self.mark_changed();
    }
}

struct RenderWorker {
    job_tx: mpsc::Sender<RenderJob>,
    result_rx: mpsc::Receiver<RenderOutcome>,
    latest_request: Arc<AtomicU64>,
    next_job_id: u64,
    gpu_available: bool,
}

impl RenderWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<RenderJob>();
        let (result_tx, result_rx) = mpsc::channel::<RenderOutcome>();
        let latest_request = Arc::new(AtomicU64::new(0));
        let worker_latest = latest_request.clone();
        let gpu_available = film_grain::wgpu::context().is_ok();

        thread::Builder::new()
            .name("film-grain-render".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let current = worker_latest.load(Ordering::SeqCst);
                    if job.id != current {
                        continue;
                    }
                    let outcome = match job.kind {
                        RenderTaskKind::Preview => {
                            render_with_input_image(job.input.as_ref(), &job.params)
                                .map(|(image, stats)| JobResult::Preview { image, stats })
                        }
                        RenderTaskKind::StatsOnly => {
                            let stats = dry_run_with_input_image(job.input.as_ref(), &job.params);
                            stats.map(|stats| JobResult::Stats { stats })
                        }
                    };
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
            gpu_available,
        }
    }

    fn request(
        &mut self,
        input: Arc<InputImage>,
        params: Params,
        kind: RenderTaskKind,
    ) -> Result<u64, String> {
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        if self.next_job_id == 0 {
            self.next_job_id = 1;
        }
        self.latest_request.store(id, Ordering::SeqCst);
        let job = RenderJob {
            id,
            input,
            params,
            kind,
        };
        self.job_tx.send(job).map_err(|err| err.to_string())?;
        Ok(id)
    }

    fn try_recv(&self) -> Option<RenderOutcome> {
        self.result_rx.try_recv().ok()
    }

    fn gpu_available(&self) -> bool {
        self.gpu_available
    }
}

struct RenderJob {
    id: u64,
    input: Arc<InputImage>,
    params: Params,
    kind: RenderTaskKind,
}

struct RenderOutcome {
    id: u64,
    outcome: Result<JobResult, RenderError>,
}

enum JobResult {
    Preview { image: RgbImage, stats: RenderStats },
    Stats { stats: RenderStats },
}

#[derive(Copy, Clone)]
enum RenderMode {
    Preview,
    StatsOnly,
}

#[derive(Copy, Clone)]
enum RenderTaskKind {
    Preview,
    StatsOnly,
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

struct WorkerState {
    status: WorkerStatus,
    active_job: Option<u64>,
    last_duration: Option<Duration>,
}

impl Default for WorkerState {
    fn default() -> Self {
        Self {
            status: WorkerStatus::Idle,
            active_job: None,
            last_duration: None,
        }
    }
}

impl WorkerState {
    fn status_text(&self) -> String {
        match &self.status {
            WorkerStatus::Idle => "idle".to_owned(),
            WorkerStatus::Rendering { started_at } => {
                format!(
                    "rendering ({:.2}s elapsed)",
                    started_at.elapsed().as_secs_f32()
                )
            }
            WorkerStatus::Completed { finished_at } => {
                let ago = finished_at.elapsed().as_secs_f32();
                if let Some(duration) = self.last_duration {
                    format!(
                        "completed {:.2}s ago (took {:.2}s)",
                        ago,
                        duration.as_secs_f32()
                    )
                } else {
                    format!("completed {:.2}s ago", ago)
                }
            }
            WorkerStatus::Failed { message } => format!("failed: {message}"),
        }
    }

    fn begin_render(&mut self, job_id: u64) {
        self.active_job = Some(job_id);
        self.status = WorkerStatus::Rendering {
            started_at: Instant::now(),
        };
        self.last_duration = None;
    }

    fn complete(&mut self) {
        if let WorkerStatus::Rendering { started_at } = self.status {
            self.last_duration = Some(started_at.elapsed());
        }
        self.active_job = None;
        self.status = WorkerStatus::Completed {
            finished_at: Instant::now(),
        };
    }

    fn fail(&mut self, message: String) {
        if let WorkerStatus::Rendering { started_at } = self.status {
            self.last_duration = Some(started_at.elapsed());
        }
        self.active_job = None;
        self.status = WorkerStatus::Failed { message };
    }

    fn reset(&mut self) {
        self.active_job = None;
        self.status = WorkerStatus::Idle;
        self.last_duration = None;
    }

    fn last_duration(&self) -> Option<Duration> {
        self.last_duration
    }
}

#[derive(Default)]
enum WorkerStatus {
    #[default]
    Idle,
    Rendering {
        started_at: Instant,
    },
    Completed {
        finished_at: Instant,
    },
    Failed {
        message: String,
    },
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
