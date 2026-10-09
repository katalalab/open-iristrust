//! Rubber-sheet normalization of fitted iris boundaries.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Error, Geometry, GrayImage, Mask, NormalizedIris, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Sampling {
    Nearest,
    #[default]
    Bilinear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct NormalizationConfig {
    pub width: usize,
    pub height: usize,
    pub sampling: Sampling,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 128,
            sampling: Sampling::Bilinear,
        }
    }
}

pub fn normalize(
    image: &GrayImage,
    geometry: &Geometry,
    noise: &Mask,
    visible: Option<&Mask>,
    config: &NormalizationConfig,
) -> Result<NormalizedIris> {
    image.validate()?;
    geometry.validate()?;
    noise.validate()?;
    if (noise.width, noise.height) != (image.width, image.height) {
        return Err(Error::InvalidInput(
            "noise mask dimensions differ from image".into(),
        ));
    }
    if let Some(mask) = visible {
        mask.validate()?;
        if (mask.width, mask.height) != (image.width, image.height) {
            return Err(Error::InvalidInput(
                "visible mask dimensions differ from image".into(),
            ));
        }
    }
    let pixels = config
        .width
        .checked_mul(config.height)
        .filter(|&n| n > 0 && n <= crate::MAX_PIXELS)
        .ok_or_else(|| {
            Error::InvalidInput("normalized dimensions are zero, overflowed, or too large".into())
        })?;
    let source_len = geometry.pupil.len();
    let values: Vec<(u8, bool)> = (0..pixels)
        .into_par_iter()
        .map(|index| {
            let x = index % config.width;
            let y = index / config.width;
            let source = (x as f64 / config.width as f64
                + geometry.orientation_radians / std::f64::consts::TAU)
                .rem_euclid(1.0)
                * source_len as f64;
            let (pupil, iris) = interpolate_boundary(geometry, source);
            let radial = (y as f64 + 0.5) / config.height as f64;
            let point = (
                pupil.x + (iris.x - pupil.x) * radial,
                pupil.y + (iris.y - pupil.y) * radial,
            );
            let valid = point.0.is_finite()
                && point.1.is_finite()
                && point.0 >= 0.0
                && point.1 >= 0.0
                && point.0 <= (image.width - 1) as f64
                && point.1 <= (image.height - 1) as f64
                && source_is_visible(noise, visible, point.0, point.1, config.sampling);
            if valid {
                (sample(image, point.0, point.1, config.sampling), true)
            } else {
                (0, false)
            }
        })
        .collect();
    let (data, mask): (Vec<_>, Vec<_>) = values.into_iter().unzip();
    Ok(NormalizedIris {
        image: GrayImage::new(config.width, config.height, data)?,
        mask: Mask::new(config.width, config.height, mask)?,
    })
}

fn interpolate_boundary(geometry: &Geometry, source: f64) -> (crate::Point, crate::Point) {
    let lower = source.floor() as usize % geometry.pupil.len();
    let upper = (lower + 1) % geometry.pupil.len();
    let fraction = source.fract();
    let interpolate = |left: crate::Point, right: crate::Point| crate::Point {
        x: left.x + (right.x - left.x) * fraction,
        y: left.y + (right.y - left.y) * fraction,
    };
    (
        interpolate(geometry.pupil[lower], geometry.pupil[upper]),
        interpolate(geometry.iris[lower], geometry.iris[upper]),
    )
}

fn source_is_visible(
    noise: &Mask,
    visible: Option<&Mask>,
    x: f64,
    y: f64,
    sampling: Sampling,
) -> bool {
    match sampling {
        Sampling::Nearest => {
            let (x, y) = nearest_coordinates(x, y);
            !mask_at(noise, x, y) && visible.is_none_or(|mask| mask_at(mask, x, y))
        }
        Sampling::Bilinear => {
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(noise.width - 1);
            let y1 = (y0 + 1).min(noise.height - 1);
            let dx = x - x0 as f64;
            let dy = y - y0 as f64;
            [
                (x0, y0, (1.0 - dx) * (1.0 - dy)),
                (x1, y0, dx * (1.0 - dy)),
                (x0, y1, (1.0 - dx) * dy),
                (x1, y1, dx * dy),
            ]
            .into_iter()
            .filter(|(_, _, weight)| *weight > 0.0)
            .all(|(x, y, _)| {
                !mask_at(noise, x, y) && visible.is_none_or(|mask| mask_at(mask, x, y))
            })
        }
    }
}

fn nearest_coordinates(x: f64, y: f64) -> (usize, usize) {
    (x.round() as usize, y.round() as usize)
}

fn mask_at(mask: &Mask, x: usize, y: usize) -> bool {
    mask.data[y * mask.width + x]
}

fn sample(image: &GrayImage, x: f64, y: f64, sampling: Sampling) -> u8 {
    match sampling {
        Sampling::Nearest => {
            let (x, y) = nearest_coordinates(x, y);
            image.data[y * image.width + x]
        }
        Sampling::Bilinear => {
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(image.width - 1);
            let y1 = (y0 + 1).min(image.height - 1);
            let dx = x - x0 as f64;
            let dy = y - y0 as f64;
            let top = image.data[y0 * image.width + x0] as f64 * (1.0 - dx)
                + image.data[y0 * image.width + x1] as f64 * dx;
            let bottom = image.data[y1 * image.width + x0] as f64 * (1.0 - dx)
                + image.data[y1 * image.width + x1] as f64 * dx;
            (top * (1.0 - dy) + bottom * dy).round() as u8
        }
    }
}
