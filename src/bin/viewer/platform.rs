use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use film_grain::{ColorMode, InputImage, RenderError, Roi};
#[cfg(target_arch = "wasm32")]
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use image::RgbImage;

use super::luma_image;

pub const OPEN_FILE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "hdr", "gif",
];
pub const SAVE_FILE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff", "bmp"];

#[derive(Clone, Copy, Debug)]
pub struct LoadDialogOptions {
    pub color_mode: ColorMode,
    pub roi: Option<Roi>,
}

impl LoadDialogOptions {
    pub fn new(color_mode: ColorMode) -> Self {
        Self {
            color_mode,
            roi: None,
        }
    }

}

#[derive(Clone)]
pub struct PreviewSaveRequest {
    pub image: RgbImage,
    pub color_mode: ColorMode,
    pub default_file_name: String,
}

impl PreviewSaveRequest {
    pub fn new(
        image: RgbImage,
        color_mode: ColorMode,
        default_file_name: impl Into<String>,
    ) -> Self {
        Self {
            image,
            color_mode,
            default_file_name: default_file_name.into(),
        }
    }
}

#[derive(Clone)]
pub struct LoadedImage {
    pub origin: SourceOrigin,
    pub cache: Arc<InputImage>,
}

impl LoadedImage {
    pub fn new(origin: SourceOrigin, cache: Arc<InputImage>) -> Self {
        Self { origin, cache }
    }
}

#[derive(Clone)]
pub enum SourceOrigin {
    FilePath { path: PathBuf },
    #[cfg(target_arch = "wasm32")]
    BrowserFile { file_name: String, bytes: Arc<Vec<u8>> },
}

impl SourceOrigin {
    pub fn file_path(path: PathBuf) -> Self {
        Self::FilePath { path }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn browser_file(file_name: String, bytes: Arc<Vec<u8>>) -> Self {
        Self::BrowserFile { file_name, bytes }
    }

    pub fn display_name(&self) -> String {
        match self {
            SourceOrigin::FilePath { path } => path.display().to_string(),
            #[cfg(target_arch = "wasm32")]
            SourceOrigin::BrowserFile { file_name, .. } => file_name.clone(),
        }
    }

    pub fn input_path(&self) -> PathBuf {
        match self {
            SourceOrigin::FilePath { path } => path.clone(),
            #[cfg(target_arch = "wasm32")]
            SourceOrigin::BrowserFile { file_name, .. } => PathBuf::from(file_name),
        }
    }

    pub fn default_output_path(&self) -> PathBuf {
        let mut path = self.input_path();
        path.set_extension("filmgrain.png");
        path
    }

    pub fn reload_image(
        &self,
        color_mode: ColorMode,
        roi: Option<&Roi>,
    ) -> Result<InputImage, RenderError> {
        match self {
            SourceOrigin::FilePath { path } => InputImage::from_path(path, color_mode, roi),
            #[cfg(target_arch = "wasm32")]
            SourceOrigin::BrowserFile { bytes, .. } => InputImage::from_bytes(bytes, color_mode, roi),
        }
    }
}

pub enum SaveOutcome {
    File(PathBuf),
    #[cfg(target_arch = "wasm32")]
    Downloaded { file_name: String },
}

pub enum PlatformEvent {
    ImageLoaded(Result<Option<LoadedImage>, String>),
    PreviewSaved(Result<Option<SaveOutcome>, String>),
}

pub trait PlatformIo {
    fn request_load_image(&self, options: LoadDialogOptions);
    fn request_save_preview(&self, request: PreviewSaveRequest);
    fn try_recv(&self) -> Option<PlatformEvent>;
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::NativePlatformIo as ActivePlatformIo;
#[cfg(target_arch = "wasm32")]
pub use web::WebPlatformIo as ActivePlatformIo;

pub fn create_platform_io() -> ActivePlatformIo {
    ActivePlatformIo::new()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use rfd::FileDialog;
    use std::sync::Mutex;

    pub struct NativePlatformIo {
        events: Mutex<VecDeque<PlatformEvent>>,
    }

    impl NativePlatformIo {
        pub fn new() -> Self {
            Self {
                events: Mutex::new(VecDeque::new()),
            }
        }

        fn push_event(&self, event: PlatformEvent) {
            if let Ok(mut queue) = self.events.lock() {
                queue.push_back(event);
            }
        }
    }

    impl PlatformIo for NativePlatformIo {
        fn request_load_image(&self, options: LoadDialogOptions) {
            let outcome = load_image_blocking(options);
            self.push_event(PlatformEvent::ImageLoaded(outcome));
        }

        fn request_save_preview(&self, request: PreviewSaveRequest) {
            let outcome = save_preview_blocking(request);
            self.push_event(PlatformEvent::PreviewSaved(outcome));
        }

        fn try_recv(&self) -> Option<PlatformEvent> {
            self.events.lock().ok().and_then(|mut q| q.pop_front())
        }
    }

    fn load_image_blocking(options: LoadDialogOptions) -> Result<Option<LoadedImage>, String> {
        let dialog = FileDialog::new().add_filter("Images", OPEN_FILE_EXTENSIONS);
        let Some(path) = dialog.pick_file() else {
            return Ok(None);
        };
        let roi = options.roi.as_ref();
        let cache = InputImage::from_path(&path, options.color_mode, roi)
            .map_err(|err| err.to_string())?;
        let origin = SourceOrigin::file_path(path);
        Ok(Some(LoadedImage::new(origin, Arc::new(cache))))
    }

    fn save_preview_blocking(request: PreviewSaveRequest) -> Result<Option<SaveOutcome>, String> {
        let dialog = FileDialog::new()
            .set_file_name(&request.default_file_name)
            .add_filter("Images", SAVE_FILE_EXTENSIONS);
        let Some(path) = dialog.save_file() else {
            return Ok(None);
        };
        save_preview_to_path(&request.image, request.color_mode, &path)?;
        Ok(Some(SaveOutcome::File(path)))
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use js_sys::Uint8Array;
    use rfd::AsyncFileDialog;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::spawn_local;
    use web_sys::{window, Blob, BlobPropertyBag, HtmlAnchorElement, Url};

    use std::cell::RefCell;
    use std::rc::Rc;

    pub struct WebPlatformIo {
        events: Rc<RefCell<VecDeque<PlatformEvent>>>,
    }

    impl WebPlatformIo {
        pub fn new() -> Self {
            Self {
                events: Rc::new(RefCell::new(VecDeque::new())),
            }
        }

        fn push_event(&self, event: PlatformEvent) {
            self.events.borrow_mut().push_back(event);
        }
    }

    impl PlatformIo for WebPlatformIo {
        fn request_load_image(&self, options: LoadDialogOptions) {
            let queue = self.events.clone();
            spawn_local(async move {
                let event = load_image_async(options).await;
                queue.borrow_mut().push_back(PlatformEvent::ImageLoaded(event));
            });
        }

        fn request_save_preview(&self, request: PreviewSaveRequest) {
            let queue = self.events.clone();
            spawn_local(async move {
                let event = save_preview_async(request).await;
                queue
                    .borrow_mut()
                    .push_back(PlatformEvent::PreviewSaved(event));
            });
        }

        fn try_recv(&self) -> Option<PlatformEvent> {
            self.events.borrow_mut().pop_front()
        }
    }

    async fn load_image_async(options: LoadDialogOptions) -> Result<Option<LoadedImage>, String> {
        let dialog = AsyncFileDialog::new().add_filter("Images", OPEN_FILE_EXTENSIONS);
        let Some(handle) = dialog.pick_file().await else {
            return Ok(None);
        };
        let data = handle.read().await;
        let bytes = Arc::new(data);
        let roi = options.roi.as_ref();
        let cache = InputImage::from_bytes(bytes.as_ref(), options.color_mode, roi)
            .map_err(|err| err.to_string())?;
        let origin = SourceOrigin::browser_file(handle.file_name(), bytes);
        Ok(Some(LoadedImage::new(origin, Arc::new(cache))))
    }

    async fn save_preview_async(
        request: PreviewSaveRequest,
    ) -> Result<Option<SaveOutcome>, String> {
        let png = encode_preview_png(&request.image, request.color_mode)
            .map_err(|err| err.to_string())?;
        trigger_download(png, &request.default_file_name)?;
        Ok(Some(SaveOutcome::Downloaded {
            file_name: request.default_file_name,
        }))
    }

    fn encode_preview_png(image: &RgbImage, mode: ColorMode) -> image::ImageResult<Vec<u8>> {
        let mut buffer = Vec::new();
        match mode {
            ColorMode::Rgb => {
                let encoder = PngEncoder::new(&mut buffer);
                encoder.write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ColorType::Rgb8.into(),
                )?;
            }
            ColorMode::Luma => {
                let gray = luma_image(image);
                let encoder = PngEncoder::new(&mut buffer);
                encoder.write_image(
                    gray.as_raw(),
                    gray.width(),
                    gray.height(),
                    ColorType::L8.into(),
                )?;
            }
        }
        Ok(buffer)
    }

    fn trigger_download(bytes: Vec<u8>, file_name: &str) -> Result<(), String> {
        let window = window().ok_or_else(|| "missing window".to_owned())?;
        let document = window
            .document()
            .ok_or_else(|| "missing document".to_owned())?;
        let body = document.body().ok_or_else(|| "missing body".to_owned())?;
        let array = Uint8Array::from(bytes.as_slice());
        let mut options = BlobPropertyBag::new();
        options.set_type("image/png");
        let blob = Blob::new_with_u8_array_sequence_and_options(&js_sys::Array::of1(&array.into()), &options)
            .map_err(|err| format!("blob error: {err:?}"))?;
        let url = Url::create_object_url_with_blob(&blob)
            .map_err(|err| format!("url error: {err:?}"))?;
        let element = document
            .create_element("a")
            .map_err(|err| format!("anchor error: {err:?}"))?;
        let anchor: HtmlAnchorElement = element
            .dyn_into()
            .map_err(|_| "failed to cast anchor".to_owned())?;
        anchor.set_href(&url);
        anchor.set_download(file_name);
        anchor.style().set_property("display", "none").ok();
        body.append_child(&anchor).ok();
        anchor.click();
        body.remove_child(&anchor).ok();
        Url::revoke_object_url(&url).ok();
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn save_preview_to_path(image: &RgbImage, mode: ColorMode, path: &Path) -> Result<(), String> {
    match mode {
        ColorMode::Rgb => image.save(path).map_err(|err| err.to_string()),
        ColorMode::Luma => {
            let gray = luma_image(image);
            gray.save(path).map_err(|err| err.to_string())
        }
    }
}
