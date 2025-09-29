use image::{GrayImage, Luma};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, trace, warn};

/// Reads a grayscale image from the given file path and normalizes pixel values to [0, 1].
#[tracing::instrument(level = "info")]
pub fn read_image(path: &PathBuf) -> Array2<f32> {
    debug!("Opening image from disk");
    let opened_image = image::open(path).unwrap_or_else(|err| {
        error!(?err, "Failed to open input image");
        panic!("Failed to open input image");
    });
    let grayscale_image = opened_image.to_luma8();
    let (width, height) = grayscale_image.dimensions();
    info!(width, height, "Normalising grayscale image");
    let normalized_pixels: Vec<f32> = grayscale_image
        .pixels()
        .enumerate()
        .map(|(idx, p)| {
            let value = p.0[0] as f32 / 255.0;
            value
        })
        .collect();
    Array2::from_shape_vec((height as usize, width as usize), normalized_pixels).unwrap_or_else(
        |err| {
            error!(?err, "Failed to build ndarray from image data");
            panic!("Error creating image array");
        },
    )
}

/// Writes an image (as an Array2<f32> with values in [0, 1]) to the given file path.
#[tracing::instrument(level = "info", fields(path = %path.as_ref().display()))]
pub fn write_image<P: AsRef<Path>>(path: P, img: &Array2<f32>) -> Result<(), String> {
    let (row_count, column_count) = (img.shape()[0], img.shape()[1]);
    debug!(row_count, column_count, "Preparing to write image");
    if row_count == 0 || column_count == 0 {
        warn!("Output image has no pixels");
    }
    let mut grayscale_image = GrayImage::new(column_count as u32, row_count as u32);
    for ((row_index, column_index), &val) in img.indexed_iter() {
        if !(0.0..=1.0).contains(&val) {
            trace!(
                row_index,
                column_index,
                value = val,
                "Clamping out-of-range pixel value"
            );
        }
        let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        grayscale_image.put_pixel(column_index as u32, row_index as u32, Luma([pixel_val]));
    }
    grayscale_image.save(&path).map_err(|err| {
        error!(?err, "Failed to persist grayscale image");
        err.to_string()
    })
}
