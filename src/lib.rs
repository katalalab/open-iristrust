//! Open Iristrust: a native Rust iris recognition pipeline.
//!
//! Images and templates stay local. ONNX inference is an optional feature.
pub mod aggregation;
pub mod encoder;
pub mod error;
pub mod filters;
pub mod geometry;
pub mod io;
pub mod matcher;
pub mod normalization;
pub mod pipeline;
mod profile;
pub mod quality;
pub mod segmentation;
pub mod types;
pub use aggregation::*;
pub use error::{Error, Result};
pub use geometry::*;
pub use matcher::*;
pub use normalization::*;
pub use pipeline::*;
pub use quality::*;
pub use types::*;
