use std::path::Path;

use image::{DynamicImage, GenericImageView, ImageFormat, RgbImage};

use crate::RenderError;
use crate::model::Plane;
use crate::params::{ColorMode, Params, Roi};

const Y_COEFF_R: f32 = 0.2126;
const Y_COEFF_G: f32 = 0.7152;
const Y_COEFF_B: f32 = 0.0722;
const CB_DENOM: f32 = 1.8556;
const CR_DENOM: f32 = 1.5748;

pub struct Workspace {
    kind: WorkspaceKind,
}

enum WorkspaceKind {
    Luma { y: Plane, cb: Plane, cr: Plane },
    Rgb { planes: [Plane; 3] },
}

impl Workspace {
    pub fn load(params: &Params) -> Result<Self, RenderError> {
        let image = image::open(&params.input_path)?;
        let cropped = apply_roi(image, params.roi.as_ref())?;
        match params.color_mode {
            ColorMode::Luma => load_luma_workspace(cropped),
            ColorMode::Rgb => load_rgb_workspace(cropped),
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        match &self.kind {
            WorkspaceKind::Luma { y, .. } => (y.width, y.height),
            WorkspaceKind::Rgb { planes } => (planes[0].width, planes[0].height),
        }
    }

    pub fn for_each_plane<F>(&mut self, mut f: F) -> Result<(), RenderError>
    where
        F: FnMut(&Plane, usize) -> Result<Plane, RenderError>,
    {
        match &mut self.kind {
            WorkspaceKind::Luma { y, .. } => {
                let updated = f(y, 0)?;
                *y = updated;
            }
            WorkspaceKind::Rgb { planes } => {
                for (idx, plane) in planes.iter_mut().enumerate() {
                    let updated = f(plane, idx)?;
                    *plane = updated;
                }
            }
        }
        Ok(())
    }

    pub fn save(self, output_path: &Path, format: ImageFormat) -> Result<(), RenderError> {
        match self.kind {
            WorkspaceKind::Luma { y, cb, cr } => {
                let width = y.width;
                let height = y.height;
                let mut cb_plane = cb;
                let mut cr_plane = cr;
                if cb_plane.width != width || cb_plane.height != height {
                    cb_plane = cb_plane.resize_nearest(width, height);
                }
                if cr_plane.width != width || cr_plane.height != height {
                    cr_plane = cr_plane.resize_nearest(width, height);
                }
                let mut buffer = vec![0u8; width * height * 3];
                for y_idx in 0..height {
                    for x_idx in 0..width {
                        let y_val = y.get(x_idx, y_idx);
                        let cb_val = cb_plane.get(x_idx, y_idx);
                        let cr_val = cr_plane.get(x_idx, y_idx);
                        let r = clamp01(y_val + CR_DENOM * cr_val);
                        let b = clamp01(y_val + CB_DENOM * cb_val);
                        let g_unclamped = (y_val - Y_COEFF_R * r - Y_COEFF_B * b) / Y_COEFF_G;
                        let g = clamp01(g_unclamped);
                        let idx = (y_idx * width + x_idx) * 3;
                        buffer[idx] = to_u8(r);
                        buffer[idx + 1] = to_u8(g);
                        buffer[idx + 2] = to_u8(b);
                    }
                }
                let image = RgbImage::from_vec(width as u32, height as u32, buffer)
                    .ok_or_else(|| RenderError::Message("failed to create RGB image".into()))?;
                save_image(image, output_path, format)
            }
            WorkspaceKind::Rgb { planes } => {
                let width = planes[0].width;
                let height = planes[0].height;
                let mut buffer = vec![0u8; width * height * 3];
                for y_idx in 0..height {
                    for x_idx in 0..width {
                        let idx = (y_idx * width + x_idx) * 3;
                        buffer[idx] = to_u8(planes[0].get(x_idx, y_idx));
                        buffer[idx + 1] = to_u8(planes[1].get(x_idx, y_idx));
                        buffer[idx + 2] = to_u8(planes[2].get(x_idx, y_idx));
                    }
                }
                let image = RgbImage::from_vec(width as u32, height as u32, buffer)
                    .ok_or_else(|| RenderError::Message("failed to create RGB image".into()))?;
                save_image(image, output_path, format)
            }
        }
    }
}

fn load_rgb_workspace(image: DynamicImage) -> Result<Workspace, RenderError> {
    let rgb = image.to_rgb32f();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let data = rgb.into_raw();
    let mut planes = [
        Plane::new(width, height),
        Plane::new(width, height),
        Plane::new(width, height),
    ];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            planes[0].set(x, y, clamp01(data[idx]));
            planes[1].set(x, y, clamp01(data[idx + 1]));
            planes[2].set(x, y, clamp01(data[idx + 2]));
        }
    }
    Ok(Workspace {
        kind: WorkspaceKind::Rgb { planes },
    })
}

fn load_luma_workspace(image: DynamicImage) -> Result<Workspace, RenderError> {
    let rgb = image.to_rgb32f();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let data = rgb.into_raw();

    let mut y_plane = Plane::new(width, height);
    let mut cb_plane = Plane::new(width, height);
    let mut cr_plane = Plane::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = clamp01(data[idx]);
            let g = clamp01(data[idx + 1]);
            let b = clamp01(data[idx + 2]);
            let luma = Y_COEFF_R * r + Y_COEFF_G * g + Y_COEFF_B * b;
            let cb = (b - luma) / CB_DENOM;
            let cr = (r - luma) / CR_DENOM;
            y_plane.set(x, y, clamp01(luma));
            cb_plane.set(x, y, cb);
            cr_plane.set(x, y, cr);
        }
    }

    Ok(Workspace {
        kind: WorkspaceKind::Luma {
            y: y_plane,
            cb: cb_plane,
            cr: cr_plane,
        },
    })
}

fn apply_roi(image: DynamicImage, roi: Option<&Roi>) -> Result<DynamicImage, RenderError> {
    let Some(roi) = roi else {
        return Ok(image);
    };
    let (width, height) = image.dimensions();
    if roi.x1 > width || roi.y1 > height {
        return Err(RenderError::Message("ROI exceeds image bounds".to_owned()));
    }
    let w = roi.x1 - roi.x0;
    let h = roi.y1 - roi.y0;
    if w == 0 || h == 0 {
        return Err(RenderError::Message(
            "ROI width and height must be positive".to_owned(),
        ));
    }
    Ok(image.crop_imm(roi.x0, roi.y0, w, h))
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn to_u8(value: f32) -> u8 {
    (clamp01(value) * 255.0 + 0.5).floor() as u8
}

fn save_image(image: RgbImage, path: &Path, format: ImageFormat) -> Result<(), RenderError> {
    image
        .save_with_format(path, format)
        .map_err(RenderError::from)
}
