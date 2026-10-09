//! Native geometry estimation from segmentation masks.
//!
//! The estimator keeps the largest connected region for each class, extracts its
//! boundary, and fits a radial ellipse. This deliberately differs from the
//! upstream Python pipeline: no contour-library or Python numerical dependency is
//! used, and the fitted boundary is sampled at equal image-space angles.

use std::{cmp::Ordering, collections::VecDeque, f64::consts::TAU};

use serde::{Deserialize, Serialize};

use crate::{Error, Geometry, GeometryMasks, Mask, Point, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeometryConfig {
    /// Number of equal-angular samples emitted for both pupil and iris boundaries.
    pub samples: usize,
    /// Small regions are insufficient to fit a stable boundary.
    pub min_component_pixels: usize,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            samples: 512,
            min_component_pixels: 16,
        }
    }
}

pub fn geometry_from_circles(
    pupil_center: Point,
    pupil_radius: f64,
    iris_center: Point,
    iris_radius: f64,
    samples: usize,
) -> Result<Geometry> {
    if !pupil_center.is_finite()
        || !iris_center.is_finite()
        || !pupil_radius.is_finite()
        || !iris_radius.is_finite()
        || pupil_radius <= 0.0
        || iris_radius <= pupil_radius
    {
        return Err(Error::InvalidInput(
            "circle geometry requires finite nested positive radii".into(),
        ));
    }
    let center_distance = ((pupil_center.x - iris_center.x).powi(2)
        + (pupil_center.y - iris_center.y).powi(2))
    .sqrt();
    if center_distance + pupil_radius >= iris_radius {
        return Err(Error::InvalidInput(
            "pupil circle must be strictly contained by iris circle".into(),
        ));
    }
    validate_samples(samples)?;
    let pupil = circle_samples(pupil_center, pupil_radius, samples);
    let iris = circle_samples(iris_center, iris_radius, samples);
    let geometry = Geometry {
        pupil,
        iris,
        eyeball: Vec::new(),
        pupil_center,
        iris_center,
        orientation_radians: 0.0,
    };
    geometry.validate()?;
    Ok(geometry)
}

pub fn estimate_geometry(masks: &GeometryMasks, config: &GeometryConfig) -> Result<Geometry> {
    masks.validate()?;
    validate_samples(config.samples)?;
    if config.min_component_pixels < 8 {
        return Err(Error::InvalidInput(
            "min_component_pixels must be at least 8".into(),
        ));
    }

    let pupil = largest_component(&masks.pupil);
    let iris = largest_component(&masks.iris);
    if pupil.len() < config.min_component_pixels {
        return Err(Error::Geometry(
            "pupil mask has no sufficiently large connected component".into(),
        ));
    }
    if iris.len() < config.min_component_pixels {
        return Err(Error::Geometry(
            "iris mask has no sufficiently large connected component".into(),
        ));
    }

    let pupil_fit = fit_radial_ellipse(&pupil, masks.pupil.width, masks.pupil.height)?;
    let iris_fit = fit_radial_ellipse(&iris, masks.iris.width, masks.iris.height)?;
    let center_distance = ((pupil_fit.center.x - iris_fit.center.x).powi(2)
        + (pupil_fit.center.y - iris_fit.center.y).powi(2))
    .sqrt();
    if center_distance + pupil_fit.major >= iris_fit.minor {
        return Err(Error::Geometry(
            "iris boundary does not enclose pupil boundary".into(),
        ));
    }

    let eyeball = largest_component(&masks.eyeball);
    let eyeball_boundary = if eyeball.is_empty() {
        Vec::new()
    } else {
        boundary_points(&eyeball, masks.eyeball.width, masks.eyeball.height)
            .into_iter()
            .map(|(x, y)| Point {
                x: x as f64,
                y: y as f64,
            })
            .collect()
    };
    let geometry = Geometry {
        pupil: pupil_fit.samples(config.samples),
        iris: iris_fit.samples(config.samples),
        eyeball: eyeball_boundary,
        pupil_center: pupil_fit.center,
        iris_center: iris_fit.center,
        orientation_radians: iris_fit.angle,
    };
    geometry.validate()?;
    Ok(geometry)
}

#[derive(Clone, Copy)]
struct Ellipse {
    center: Point,
    major: f64,
    minor: f64,
    angle: f64,
}

impl Ellipse {
    fn samples(self, count: usize) -> Vec<Point> {
        (0..count)
            .map(|i| {
                let theta = TAU * i as f64 / count as f64;
                let local = theta - self.angle;
                let denominator =
                    (local.cos() / self.major).powi(2) + (local.sin() / self.minor).powi(2);
                let radius = denominator.sqrt().recip();
                Point {
                    x: self.center.x + radius * theta.cos(),
                    y: self.center.y + radius * theta.sin(),
                }
            })
            .collect()
    }
}

fn fit_radial_ellipse(
    component: &[(usize, usize)],
    width: usize,
    height: usize,
) -> Result<Ellipse> {
    let n = component.len() as f64;
    let center = Point {
        x: component.iter().map(|(x, _)| *x as f64).sum::<f64>() / n,
        y: component.iter().map(|(_, y)| *y as f64).sum::<f64>() / n,
    };
    let (mut xx, mut yy, xy) = component
        .iter()
        .fold((0.0, 0.0, 0.0), |(xx, yy, xy), (x, y)| {
            let dx = *x as f64 - center.x;
            let dy = *y as f64 - center.y;
            (xx + dx * dx, yy + dy * dy, xy + dx * dy)
        });
    xx /= n;
    yy /= n;
    let xy = xy / n;
    let discriminant = ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt();
    let major_variance = (xx + yy + discriminant) * 0.5;
    let minor_variance = (xx + yy - discriminant) * 0.5;
    if !major_variance.is_finite() || !minor_variance.is_finite() || minor_variance <= 0.0 {
        return Err(Error::Geometry("mask boundary is underdetermined".into()));
    }
    let mut angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    if major_variance / minor_variance < 1.05 {
        angle = 0.0;
    }

    let boundary = boundary_points(component, width, height);
    if boundary.len() < 8 {
        return Err(Error::Geometry("mask boundary is underdetermined".into()));
    }
    let covariance_major = (4.0 * major_variance).sqrt();
    let covariance_minor = (4.0 * minor_variance).sqrt();
    let major = directional_radius(&boundary, center, angle, covariance_major);
    let minor = directional_radius(
        &boundary,
        center,
        angle + std::f64::consts::FRAC_PI_2,
        covariance_minor,
    );
    if !major.is_finite() || !minor.is_finite() || major <= 1.0 || minor <= 1.0 {
        return Err(Error::Geometry(
            "mask boundary cannot form a finite ellipse".into(),
        ));
    }
    Ok(Ellipse {
        center,
        major,
        minor,
        angle,
    })
}

fn directional_radius(
    boundary: &[(usize, usize)],
    center: Point,
    angle: f64,
    fallback: f64,
) -> f64 {
    let direction = (angle.cos(), angle.sin());
    boundary
        .iter()
        .filter_map(|(x, y)| {
            let dx = *x as f64 - center.x;
            let dy = *y as f64 - center.y;
            let radius = (dx * dx + dy * dy).sqrt();
            (radius > 0.0 && ((dx * direction.0 + dy * direction.1) / radius).abs() >= 0.94)
                .then_some(radius)
        })
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .unwrap_or(fallback)
}

fn circle_samples(center: Point, radius: f64, samples: usize) -> Vec<Point> {
    (0..samples)
        .map(|i| {
            let angle = TAU * i as f64 / samples as f64;
            Point {
                x: center.x + radius * angle.cos(),
                y: center.y + radius * angle.sin(),
            }
        })
        .collect()
}

fn validate_samples(samples: usize) -> Result<()> {
    if !(8..=4096).contains(&samples) {
        return Err(Error::InvalidInput(
            "geometry sample count must be between 8 and 4096".into(),
        ));
    }
    Ok(())
}

fn largest_component(mask: &Mask) -> Vec<(usize, usize)> {
    let mut visited = vec![false; mask.data.len()];
    let mut largest = Vec::new();
    for start in 0..mask.data.len() {
        if visited[start] || !mask.data[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(index) = queue.pop_front() {
            let x = index % mask.width;
            let y = index / mask.width;
            component.push((x, y));
            for ny in y.saturating_sub(1)..=(y + 1).min(mask.height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(mask.width - 1) {
                    let neighbor = ny * mask.width + nx;
                    if !visited[neighbor] && mask.data[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if component.len() > largest.len() {
            largest = component;
        }
    }
    largest
}

fn boundary_points(
    component: &[(usize, usize)],
    width: usize,
    height: usize,
) -> Vec<(usize, usize)> {
    let mut inside = vec![false; width * height];
    for &(x, y) in component {
        inside[y * width + x] = true;
    }
    component
        .iter()
        .copied()
        .filter(|&(x, y)| {
            x == 0
                || y == 0
                || x + 1 == width
                || y + 1 == height
                || (y.saturating_sub(1)..=(y + 1).min(height - 1)).any(|ny| {
                    (x.saturating_sub(1)..=(x + 1).min(width - 1))
                        .any(|nx| !inside[ny * width + nx])
                })
        })
        .collect()
}
