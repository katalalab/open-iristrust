use iristrust::segmentation::{SegmentationConfig, binarize, postprocess, preprocess};
use iristrust::{Error, GrayImage};

#[test]
fn preprocess_uses_channel_first_imagenet_normalization() {
    let image = GrayImage::new(2, 1, vec![0, 255]).unwrap();
    let config = SegmentationConfig {
        input_width: 2,
        input_height: 1,
        ..Default::default()
    };
    let input = preprocess(&image, &config).unwrap();
    let means = [0.485_f32, 0.456, 0.406];
    let stds = [0.229_f32, 0.224, 0.225];
    for c in 0..3 {
        assert!((input[c * 2] + means[c] / stds[c]).abs() < 1e-6);
        assert!((input[c * 2 + 1] - (1.0 - means[c]) / stds[c]).abs() < 1e-6);
    }
}

#[test]
fn postprocess_preserves_class_order_and_nearest_resize() {
    let predictions = vec![0.9, 0.2, 0.7, 0.8, 0.0, 0.9, 0.2, 0.7];
    let map = postprocess(&predictions, &[1, 4, 1, 2], 4, 1).unwrap();
    let masks = binarize(&map, &[0.5; 4]).unwrap();
    assert_eq!(masks.eyeball.data, vec![true, true, false, false]);
    assert_eq!(masks.iris.data, vec![true; 4]);
    assert_eq!(masks.pupil.data, vec![false, false, true, true]);
    assert_eq!(masks.noise.data, vec![false, false, true, true]);
}

#[test]
fn malformed_probabilities_are_rejected() {
    assert!(matches!(
        postprocess(&[f32::NAN; 4], &[1, 4, 1, 1], 1, 1),
        Err(Error::InvalidInput(_))
    ));
    assert!(postprocess(&[1.2; 4], &[1, 4, 1, 1], 1, 1).is_err());
    assert!(postprocess(&[0.5; 4], &[1, 1, 1, 4], 1, 1).is_err());
    assert!(postprocess(&[], &[1, 4, 0, 1], 1, 1).is_err());
}

#[test]
fn preprocessing_rejects_zero_and_overflowed_resolution() {
    let image = GrayImage::new(1, 1, vec![127]).unwrap();
    assert!(
        preprocess(
            &image,
            &SegmentationConfig {
                input_width: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        preprocess(
            &image,
            &SegmentationConfig {
                input_width: usize::MAX,
                ..Default::default()
            }
        )
        .is_err()
    );
}
