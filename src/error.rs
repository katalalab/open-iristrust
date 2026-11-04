use thiserror::Error;

/// Historical successful publications before a later output failed.
///
/// Paths may have been changed or removed by another process after publication.
/// Display and Debug omit paths and the underlying error text; callers with access
/// to the value can inspect both explicitly.
pub struct PartialPublish {
    published_paths: Vec<std::path::PathBuf>,
    cause: std::io::Error,
}

impl PartialPublish {
    pub(crate) fn new(published_paths: Vec<std::path::PathBuf>, cause: std::io::Error) -> Self {
        Self {
            published_paths,
            cause,
        }
    }

    pub fn published_paths(&self) -> &[std::path::PathBuf] {
        &self.published_paths
    }

    pub fn cause(&self) -> &std::io::Error {
        &self.cause
    }
}

impl std::fmt::Display for PartialPublish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "output publication incomplete after {} path(s); outputs may remain ({:?})",
            self.published_paths.len(),
            self.cause.kind()
        )
    }
}

impl std::fmt::Debug for PartialPublish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartialPublish")
            .field("published_count", &self.published_paths.len())
            .field("cause_kind", &self.cause.kind())
            .finish()
    }
}

impl std::error::Error for PartialPublish {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

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
    #[error(transparent)]
    PartialPublish(#[from] PartialPublish),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_publication_format_hides_paths_and_cause_text() {
        let partial = PartialPublish::new(
            vec![std::path::PathBuf::from("SYNTHETIC_PRIVATE_PATH")],
            std::io::Error::other("SYNTHETIC_PRIVATE_CAUSE"),
        );
        assert_eq!(
            partial.published_paths()[0],
            std::path::Path::new("SYNTHETIC_PRIVATE_PATH")
        );
        assert!(
            partial
                .cause()
                .to_string()
                .contains("SYNTHETIC_PRIVATE_CAUSE")
        );
        let error = Error::PartialPublish(partial);
        let source = std::error::Error::source(&error).expect("underlying cause remains available");
        assert!(source.to_string().contains("SYNTHETIC_PRIVATE_CAUSE"));
        for text in [format!("{error}"), format!("{error:?}")] {
            assert!(!text.contains("SYNTHETIC_PRIVATE_PATH"));
            assert!(!text.contains("SYNTHETIC_PRIVATE_CAUSE"));
            assert!(text.contains("1"));
        }
    }
}
