use iristrust::*;

fn synthetic() -> (GrayImage, Geometry) {
    let center = Point { x: 80.0, y: 60.0 };
    let geometry = geometry_from_circles(center, 12.0, center, 40.0, 512).unwrap();
    let data = (0..160 * 120)
        .map(|i| {
            let x = (i % 160) as f64 - center.x;
            let y = (i / 160) as f64 - center.y;
            let r = x.hypot(y);
            let phi = y.atan2(x);
            if r < 12.0 {
                15
            } else if r < 42.0 {
                (110.0 + 10.0 * (phi * 17.0 + r * 0.4).sin() + 6.0 * (phi * 31.0 - r * 0.3).cos())
                    as u8
            } else {
                180
            }
        })
        .collect();
    (GrayImage::new(160, 120, data).unwrap(), geometry)
}

#[test]
fn native_extraction_serialization_matching_and_aggregation() {
    let (image, geometry) = synthetic();
    let pipeline = IrisPipeline::new(PipelineConfig::default()).unwrap();
    let extracted = pipeline
        .extract_with_geometry(&image, EyeSide::Left, &geometry, None, None)
        .unwrap();
    assert!(extracted.template.valid_bits() > 4096);
    let serialized = serde_json::to_vec(&extracted.template).unwrap();
    let decoded: IrisTemplate = serde_json::from_slice(&serialized).unwrap();
    let matcher = Matcher::new(MatcherConfig::default()).unwrap();
    let rotated = rotate_template(&decoded, 5).unwrap();
    let comparison = matcher.compare(&decoded, &rotated).unwrap();
    assert_eq!(comparison.distance, 0.0);
    assert_eq!(comparison.rotation, -5);
    let aggregate = aggregate(&[decoded.clone(), rotated], &AggregationConfig::default()).unwrap();
    assert_eq!(matcher.compare(&decoded, &aggregate).unwrap().distance, 0.0);
}

#[test]
fn blank_image_and_wrong_mask_dimensions_are_rejected() {
    let (mut image, geometry) = synthetic();
    image.data.fill(128);
    let pipeline = IrisPipeline::new(PipelineConfig::default()).unwrap();
    assert!(
        pipeline
            .extract_with_geometry(&image, EyeSide::Left, &geometry, None, None)
            .is_err()
    );
    let noise = Mask::new(1, 1, vec![false]).unwrap();
    assert!(
        pipeline
            .extract_with_geometry(&image, EyeSide::Left, &geometry, Some(&noise), None)
            .is_err()
    );
}

#[test]
fn normalization_config_is_part_of_extraction_profile() {
    let (image, geometry) = synthetic();
    let a = IrisPipeline::new(PipelineConfig::default())
        .unwrap()
        .extract_with_geometry(&image, EyeSide::Left, &geometry, None, None)
        .unwrap();
    let mut config = PipelineConfig::default();
    config.normalization.sampling = Sampling::Nearest;
    let b = IrisPipeline::new(config)
        .unwrap()
        .extract_with_geometry(&image, EyeSide::Left, &geometry, None, None)
        .unwrap();
    assert_ne!(a.template.encoding_profile, b.template.encoding_profile);
    assert!(
        Matcher::new(MatcherConfig::default())
            .unwrap()
            .compare(&a.template, &b.template)
            .is_err()
    );
}
