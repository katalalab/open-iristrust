use iristrust::{
    GeometryConfig, GeometryMasks, Mask, NormalizationConfig, Point, QualityConfig, QualityMetrics,
    Raster, Sampling, estimate_geometry, gate_quality, geometry_from_circles, measure_quality,
    normalize,
};

fn mask(width: usize, height: usize, predicate: impl Fn(usize, usize) -> bool) -> Mask {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            data.push(predicate(x, y));
        }
    }
    Raster::new(width, height, data).unwrap()
}

fn circle_mask(width: usize, height: usize, center: Point, radius: f64) -> Mask {
    mask(width, height, |x, y| {
        let dx = x as f64 - center.x;
        let dy = y as f64 - center.y;
        dx * dx + dy * dy <= radius * radius
    })
}

fn geometry_masks(width: usize, height: usize, pupil: Mask, iris: Mask) -> GeometryMasks {
    GeometryMasks {
        pupil,
        iris,
        eyeball: mask(width, height, |_, _| true),
        noise: mask(width, height, |_, _| false),
    }
}

#[test]
fn estimates_equal_angular_boundaries_for_displaced_circles() {
    let pupil_center = Point { x: 31.0, y: 29.0 };
    let iris_center = Point { x: 34.0, y: 30.0 };
    let geometry = estimate_geometry(
        &geometry_masks(
            80,
            72,
            circle_mask(80, 72, pupil_center, 10.0),
            circle_mask(80, 72, iris_center, 25.0),
        ),
        &GeometryConfig::default(),
    )
    .unwrap();

    assert_eq!(geometry.pupil.len(), 512);
    assert_eq!(geometry.iris.len(), 512);
    assert!((geometry.pupil_center.x - pupil_center.x).abs() < 1.5);
    assert!((geometry.iris_center.y - iris_center.y).abs() < 1.5);
    assert!((geometry.pupil[0].x - (geometry.pupil_center.x + 10.0)).abs() < 2.0);
    assert!((geometry.iris[128].y - (geometry.iris_center.y + 25.0)).abs() < 2.0);
}

#[test]
fn geometry_uses_largest_connected_component() {
    let pupil = mask(80, 72, |x, y| {
        let main = ((x as f64 - 30.0).powi(2) + (y as f64 - 30.0).powi(2)) <= 9.0_f64.powi(2);
        let speck = ((x as f64 - 70.0).powi(2) + (y as f64 - 10.0).powi(2)) <= 2.0_f64.powi(2);
        main || speck
    });
    let geometry = estimate_geometry(
        &geometry_masks(
            80,
            72,
            pupil,
            circle_mask(80, 72, Point { x: 30.0, y: 30.0 }, 24.0),
        ),
        &GeometryConfig::default(),
    )
    .unwrap();
    assert!((geometry.pupil_center.x - 30.0).abs() < 1.5);
}

#[test]
fn geometry_rejects_missing_or_mismatched_masks() {
    let empty = mask(20, 20, |_, _| false);
    let iris = circle_mask(20, 20, Point { x: 10.0, y: 10.0 }, 8.0);
    assert!(
        estimate_geometry(
            &geometry_masks(20, 20, empty, iris),
            &GeometryConfig::default()
        )
        .is_err()
    );

    let masks = GeometryMasks {
        pupil: circle_mask(20, 20, Point { x: 10.0, y: 10.0 }, 3.0),
        iris: circle_mask(21, 20, Point { x: 10.0, y: 10.0 }, 8.0),
        eyeball: mask(20, 20, |_, _| true),
        noise: mask(20, 20, |_, _| false),
    };
    assert!(estimate_geometry(&masks, &GeometryConfig::default()).is_err());
}

#[test]
fn circle_geometry_validates_inputs() {
    let geometry = geometry_from_circles(
        Point { x: 10.0, y: 10.0 },
        3.0,
        Point { x: 11.0, y: 10.0 },
        9.0,
        16,
    )
    .unwrap();
    assert_eq!(geometry.pupil.len(), 16);
    assert!(
        geometry_from_circles(
            Point {
                x: f64::NAN,
                y: 0.0
            },
            2.0,
            Point { x: 0.0, y: 0.0 },
            4.0,
            16
        )
        .is_err()
    );
    assert!(
        geometry_from_circles(
            Point { x: 0.0, y: 0.0 },
            0.0,
            Point { x: 0.0, y: 0.0 },
            4.0,
            16
        )
        .is_err()
    );
    assert!(
        geometry_from_circles(
            Point { x: 0.0, y: 0.0 },
            4.0,
            Point { x: 0.0, y: 0.0 },
            4.0,
            16
        )
        .is_err()
    );
    assert!(
        geometry_from_circles(
            Point { x: 0.0, y: 0.0 },
            4.0,
            Point { x: 0.0, y: 0.0 },
            3.0,
            16
        )
        .is_err()
    );
    assert!(
        geometry_from_circles(
            Point { x: 0.0, y: 0.0 },
            1.0,
            Point { x: 0.0, y: 0.0 },
            4.0,
            7
        )
        .is_err()
    );
    assert!(
        geometry_from_circles(
            Point { x: 4.0, y: 0.0 },
            3.0,
            Point { x: 0.0, y: 0.0 },
            5.0,
            16
        )
        .is_err()
    );
}

#[test]
fn geometry_rejects_a_raster_past_the_pixel_limit_without_allocating_it() {
    let oversized = Raster {
        width: 4097,
        height: 4097,
        data: Vec::new(),
    };
    let masks = GeometryMasks {
        pupil: oversized.clone(),
        iris: oversized.clone(),
        eyeball: oversized.clone(),
        noise: oversized,
    };
    assert!(estimate_geometry(&masks, &GeometryConfig::default()).is_err());
}

#[test]
fn normalization_marks_noise_and_outside_samples_invalid() {
    let image = Raster::new(20, 20, (0..400).map(|i| (i % 251) as u8).collect()).unwrap();
    let geometry = geometry_from_circles(
        Point { x: 10.0, y: 10.0 },
        4.0,
        Point { x: 10.0, y: 10.0 },
        13.0,
        64,
    )
    .unwrap();
    let noisy = mask(20, 20, |x, y| x == 10 && y == 4);
    let visible = mask(20, 20, |x, _| x < 15);
    let normalized = normalize(
        &image,
        &geometry,
        &noisy,
        Some(&visible),
        &NormalizationConfig {
            width: 64,
            height: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((normalized.image.width, normalized.image.height), (64, 8));
    assert!(normalized.mask.data.iter().any(|valid| *valid));
    assert!(normalized.mask.data.iter().any(|valid| !*valid));
}

#[test]
fn bilinear_normalization_rejects_a_masked_nonzero_weight_neighbor() {
    let image = Raster::new(8, 8, (0..64).map(|value| value as u8).collect()).unwrap();
    let geometry = geometry_from_circles(
        Point { x: 2.25, y: 2.25 },
        1.0,
        Point { x: 2.25, y: 2.25 },
        3.0,
        8,
    )
    .unwrap();
    let noise = mask(8, 8, |x, y| x == 5 && y == 3);

    let normalized = normalize(
        &image,
        &geometry,
        &noise,
        None,
        &NormalizationConfig {
            width: 1,
            height: 1,
            sampling: Sampling::Bilinear,
        },
    )
    .unwrap();
    assert!(!normalized.mask.data[0]);

    let nearest = normalize(
        &image,
        &geometry,
        &noise,
        None,
        &NormalizationConfig {
            width: 1,
            height: 1,
            sampling: Sampling::Nearest,
        },
    )
    .unwrap();
    assert!(nearest.mask.data[0]);
    assert_eq!(nearest.image.data[0], image.data[2 * image.width + 4]);

    let clean_noise = mask(8, 8, |_, _| false);
    let visibility = mask(8, 8, |x, y| !(x == 5 && y == 3));
    let occluded = normalize(
        &image,
        &geometry,
        &clean_noise,
        Some(&visibility),
        &NormalizationConfig {
            width: 1,
            height: 1,
            sampling: Sampling::Bilinear,
        },
    )
    .unwrap();
    assert!(!occluded.mask.data[0]);
}

#[test]
fn bilinear_normalization_ignores_a_masked_zero_weight_corner() {
    let image = Raster::new(8, 8, vec![64_u8; 64]).unwrap();
    let geometry = geometry_from_circles(
        Point { x: 2.0, y: 2.25 },
        1.0,
        Point { x: 2.0, y: 2.25 },
        3.0,
        8,
    )
    .unwrap();
    let noise = mask(8, 8, |x, y| x == 5 && y == 3);

    let normalized = normalize(
        &image,
        &geometry,
        &noise,
        None,
        &NormalizationConfig {
            width: 1,
            height: 1,
            sampling: Sampling::Bilinear,
        },
    )
    .unwrap();
    assert!(normalized.mask.data[0]);
}

#[test]
fn quality_gate_reports_explicit_metric_rejection() {
    let image = Raster::new(16, 4, vec![0_u8; 64]).unwrap();
    let normalized = iristrust::NormalizedIris {
        image,
        mask: mask(16, 4, |_, _| true),
    };
    let geometry = geometry_from_circles(
        Point { x: 8.0, y: 8.0 },
        1.0,
        Point { x: 8.0, y: 8.0 },
        10.0,
        32,
    )
    .unwrap();
    let metrics = measure_quality(&normalized, &geometry).unwrap();
    let error = gate_quality(&metrics, &QualityConfig::default()).unwrap_err();
    assert!(error.to_string().contains("pupil_iris_ratio"));
}

#[test]
fn quality_gate_rejects_negative_physical_metrics() {
    let metrics = QualityMetrics {
        visible_fraction: 0.5,
        sharpness: 1.0,
        pupil_iris_ratio: 0.3,
        center_distance_ratio: -0.1,
        offgaze: 0.2,
        orientation_radians: 0.0,
    };
    let error = gate_quality(&metrics, &QualityConfig::default()).unwrap_err();
    assert!(error.to_string().contains("center_distance_ratio"));
}

#[test]
fn quality_offgaze_detects_a_concentric_elliptical_iris() {
    let image = Raster::new(32, 8, vec![64_u8; 256]).unwrap();
    let normalized = iristrust::NormalizedIris {
        image,
        mask: mask(32, 8, |_, _| true),
    };
    let mut geometry = geometry_from_circles(
        Point { x: 16.0, y: 16.0 },
        3.0,
        Point { x: 16.0, y: 16.0 },
        10.0,
        64,
    )
    .unwrap();
    let samples = geometry.iris.len();
    for (index, point) in geometry.iris.iter_mut().enumerate() {
        let angle = std::f64::consts::TAU * index as f64 / samples as f64;
        let radius = ((angle.cos() / 6.0).powi(2) + (angle.sin() / 10.0).powi(2))
            .sqrt()
            .recip();
        point.x = geometry.iris_center.x + radius * angle.cos();
        point.y = geometry.iris_center.y + radius * angle.sin();
    }

    let metrics = measure_quality(&normalized, &geometry).unwrap();
    assert_eq!(metrics.center_distance_ratio, 0.0);
    assert!((metrics.offgaze - 0.8).abs() < 0.03);
}
