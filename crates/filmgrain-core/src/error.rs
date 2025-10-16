use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("GPU backend requested but `gpu` feature not enabled")]
    GpuDisabled,

    #[error("WGPU initialization failed: {0}")]
    WgpuInit(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(&'static str),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Other: {0}")]
    Other(String),
}
