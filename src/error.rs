use thiserror::Error;

/// Errors identify the failed processing stage.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("geometry estimation failed: {0}")]
    Geometry(String),
    #[error("quality gate failed: {0}")]
    Quality(String),
    #[error("incompatible templates: {0}")]
    IncompatibleTemplate(String),
    #[error("no sufficient valid bit overlap")]
    NoOverlap,
    #[error("ONNX inference failed: {0}")]
    Onnx(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}
pub type Result<T> = std::result::Result<T, Error>;
