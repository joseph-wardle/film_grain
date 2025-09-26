use image::{GrayImage, Luma};
use ndarray::Array2;
use std::path::Path;
use tracing::{debug, error, info, trace, warn};

/// Reads a grayscale image from the given file path and normalizes pixel values to [0, 1].
#[tracing::instrument(level = "info")]
pub fn read_image(path: &str) -> Array2<f32> {
    debug!("Opening image from disk");
    let img = image::open(path).unwrap_or_else(|err| {
        error!(?err, "Failed to open input image");
        panic!("Failed to open input image");
    });
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    info!(width, height, "Normalising grayscale image");
    let data: Vec<f32> = gray
        .pixels()
        .enumerate()
        .map(|(idx, p)| {
            let value = p.0[0] as f32 / 255.0;
            value
        })
        .collect();
    Array2::from_shape_vec((height as usize, width as usize), data).unwrap_or_else(|err| {
        error!(?err, "Failed to build ndarray from image data");
        panic!("Error creating image array");
    })
}

/// Writes an image (as an Array2<f32> with values in [0, 1]) to the given file path.
#[tracing::instrument(level = "info", fields(path = %path.as_ref().display()))]
pub fn write_image<P: AsRef<Path>>(path: P, img: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = (img.shape()[0], img.shape()[1]);
    debug!(rows, cols, "Preparing to write image");
    if rows == 0 || cols == 0 {
        warn!("Output image has no pixels");
    }
    let mut gray_img = GrayImage::new(cols as u32, rows as u32);
    for ((row, col), &val) in img.indexed_iter() {
        if !(0.0..=1.0).contains(&val) {
            trace!(row, col, value = val, "Clamping out-of-range pixel value");
        }
        let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        gray_img.put_pixel(col as u32, row as u32, Luma([pixel_val]));
    }
    gray_img.save(&path).map_err(|err| {
        error!(?err, "Failed to persist grayscale image");
        err.to_string()
    })
}
