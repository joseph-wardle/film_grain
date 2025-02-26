use image::{GrayImage, Luma};
use ndarray::Array2;
use std::path::Path;

/// Reads a grayscale image from the given file path and normalizes pixel values to [0, 1].
pub fn read_image(path: &str) -> Array2<f32> {
    let img = image::open(path).expect("Failed to open input image");
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let data: Vec<f32> = gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    Array2::from_shape_vec((height as usize, width as usize), data)
        .expect("Error creating image array")
}

/// Writes an image (as an Array2<f32> with values in [0, 1]) to the given file path.
pub fn write_image<P: AsRef<Path>>(path: P, img: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = (img.shape()[0], img.shape()[1]);
    let mut gray_img = GrayImage::new(cols as u32, rows as u32);
    for ((row, col), &val) in img.indexed_iter() {
        let pixel_val = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        gray_img.put_pixel(col as u32, row as u32, Luma([pixel_val]));
    }
    gray_img.save(path).map_err(|e| e.to_string())
}
