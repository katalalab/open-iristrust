//! Quality measurements and configurable extraction gate.

use serde::{Deserialize, Serialize};

use crate::{Error, Geometry, NormalizedIris, Result};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub visible_fraction: f64,
    pub sharpness: f64,
    pub pupil_iris_ratio: f64,
    pub center_distance_ratio: f64,
    pub offgaze: f64,
    pub orientation_radians: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QualityConfig {
    pub min_visible_fraction: f64,
    pub min_sharpness: f64,
    pub min_pupil_iris_ratio: f64,
    pub max_pupil_iris_ratio: f64,
    pub max_center_distance_ratio: f64,
    pub max_offgaze: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_visible_fraction: 0.25,
            min_sharpness: 0.0,
            min_pupil_iris_ratio: 0.1,
            max_pupil_iris_ratio: 0.7,
            max_center_distance_ratio: 0.9,
            max_offgaze: 0.8,
        }
    }
}

pub fn measure_quality(normalized: &NormalizedIris, geometry: &Geometry) -> Result<QualityMetrics> {
    normalized.validate()?;
    geometry.validate()?;
    let valid = normalized.mask.data.iter().filter(|value| **value).count();
    let visible_fraction = valid as f64 / normalized.mask.data.len() as f64;
    let mut differences = Vec::new();
    for y in 0..normalized.image.height {
        for x in 1..normalized.image.width {
            let index = y * normalized.image.width + x;
            if normalized.mask.data[index] && normalized.mask.data[index - 1] {
                differences.push(
                    (normalized.image.data[index] as f64 - normalized.image.data[index - 1] as f64)
                        .powi(2),
                );
            }
        }
    }
    let sharpness = if differences.is_empty() {
        0.0
    } else {
        (differences.iter().sum::<f64>() / differences.len() as f64).sqrt()
    };
    let pupil_radius = mean_radius(&geometry.pupil, geometry.pupil_center);
    let iris_radius = mean_radius(&geometry.iris, geometry.iris_center);
    if !pupil_radius.is_finite()
        || !iris_radius.is_finite()
        || pupil_radius <= 0.0
        || iris_radius <= pupil_radius
    {
        return Err(Error::Geometry(
            "geometry has invalid pupil/iris radii".into(),
        ));
    }
    let center_distance = ((geometry.pupil_center.x - geometry.iris_center.x).powi(2)
        + (geometry.pupil_center.y - geometry.iris_center.y).powi(2))
    .sqrt();
    let center_distance_ratio = center_distance / iris_radius;
    let (minimum_iris_radius, maximum_iris_radius) = geometry
        .iris
        .iter()
        .map(|point| {
            ((point.x - geometry.iris_center.x).powi(2)
                + (point.y - geometry.iris_center.y).powi(2))
            .sqrt()
        })
        .fold((f64::INFINITY, 0.0_f64), |(minimum, maximum), radius| {
            (minimum.min(radius), maximum.max(radius))
        });
    if !minimum_iris_radius.is_finite() || maximum_iris_radius <= 0.0 {
        return Err(Error::Geometry(
            "geometry has invalid iris boundary radii".into(),
        ));
    }
    // Boundary eccentricity is an off-gaze proxy independent of pupil decentration.
    let offgaze = (1.0 - (minimum_iris_radius / maximum_iris_radius).powi(2))
        .max(0.0)
        .sqrt();
    Ok(QualityMetrics {
        visible_fraction,
        sharpness,
        pupil_iris_ratio: pupil_radius / iris_radius,
        center_distance_ratio,
        offgaze,
        orientation_radians: geometry.orientation_radians,
    })
}

pub fn gate_quality(metrics: &QualityMetrics, config: &QualityConfig) -> Result<()> {
    validate_config(config)?;
    validate_metrics(metrics)?;
    let mut failures = Vec::new();
    if metrics.visible_fraction < config.min_visible_fraction {
        failures.push(format!(
            "visible_fraction {:.4} < {:.4}",
            metrics.visible_fraction, config.min_visible_fraction
        ));
    }
    if metrics.sharpness < config.min_sharpness {
        failures.push(format!(
            "sharpness {:.4} < {:.4}",
            metrics.sharpness, config.min_sharpness
        ));
    }
    if metrics.pupil_iris_ratio < config.min_pupil_iris_ratio
        || metrics.pupil_iris_ratio > config.max_pupil_iris_ratio
    {
        failures.push(format!(
            "pupil_iris_ratio {:.4} outside [{:.4}, {:.4}]",
            metrics.pupil_iris_ratio, config.min_pupil_iris_ratio, config.max_pupil_iris_ratio
        ));
    }
    if metrics.center_distance_ratio > config.max_center_distance_ratio {
        failures.push(format!(
            "center_distance_ratio {:.4} > {:.4}",
            metrics.center_distance_ratio, config.max_center_distance_ratio
        ));
    }
    if metrics.offgaze > config.max_offgaze {
        failures.push(format!(
            "offgaze {:.4} > {:.4}",
            metrics.offgaze, config.max_offgaze
        ));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::Quality(failures.join("; ")))
    }
}

fn validate_metrics(metrics: &QualityMetrics) -> Result<()> {
    let values = [
        metrics.visible_fraction,
        metrics.sharpness,
        metrics.pupil_iris_ratio,
        metrics.center_distance_ratio,
        metrics.offgaze,
        metrics.orientation_radians,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Quality("metrics contain a non-finite value".into()));
    }
    if !(0.0..=1.0).contains(&metrics.visible_fraction) {
        return Err(Error::Quality(
            "visible_fraction must be within [0, 1]".into(),
        ));
    }
    if metrics.sharpness < 0.0 {
        return Err(Error::Quality("sharpness must be non-negative".into()));
    }
    if !(0.0..=1.0).contains(&metrics.pupil_iris_ratio) {
        return Err(Error::Quality(
            "pupil_iris_ratio must be within [0, 1]".into(),
        ));
    }
    if metrics.center_distance_ratio < 0.0 {
        return Err(Error::Quality(
            "center_distance_ratio must be non-negative".into(),
        ));
    }
    if !(0.0..=1.0).contains(&metrics.offgaze) {
        return Err(Error::Quality("offgaze must be within [0, 1]".into()));
    }
    Ok(())
}

fn mean_radius(points: &[crate::Point], center: crate::Point) -> f64 {
    points
        .iter()
        .map(|point| ((point.x - center.x).powi(2) + (point.y - center.y).powi(2)).sqrt())
        .sum::<f64>()
        / points.len() as f64
}

fn validate_config(config: &QualityConfig) -> Result<()> {
    let values = [
        config.min_visible_fraction,
        config.min_sharpness,
        config.min_pupil_iris_ratio,
        config.max_pupil_iris_ratio,
        config.max_center_distance_ratio,
        config.max_offgaze,
    ];
    if values.iter().any(|value| !value.is_finite())
        || !(0.0..=1.0).contains(&config.min_visible_fraction)
        || config.min_sharpness < 0.0
        || !(0.0..=1.0).contains(&config.min_pupil_iris_ratio)
        || !(0.0..=1.0).contains(&config.max_pupil_iris_ratio)
        || config.min_pupil_iris_ratio > config.max_pupil_iris_ratio
        || !(0.0..=1.0).contains(&config.max_center_distance_ratio)
        || !(0.0..=1.0).contains(&config.max_offgaze)
    {
        return Err(Error::InvalidInput("invalid quality thresholds".into()));
    }
    Ok(())
}
