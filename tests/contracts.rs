use iristrust::*;
use iristrust::{
    encoder::IrisEncoder,
    filters::{FilterBank, FilterBankConfig, FilterResponse},
};

#[test]
fn crossing_and_misordered_boundaries_are_rejected() {
    let center = Point { x: 20.0, y: 20.0 };
    let mut geometry = geometry_from_circles(center, 2.0, center, 6.0, 8).unwrap();
    geometry.pupil[0].x = 27.0;
    assert!(geometry.validate().is_err());
    let mut geometry = geometry_from_circles(center, 2.0, center, 6.0, 8).unwrap();
    geometry.iris.swap(1, 2);
    assert!(geometry.validate().is_err());
}
#[test]
fn checked_bit_access_never_aliases_another_phase_or_panics() {
    let mut plane = CodePlane {
        rows: 1,
        cols: 2,
        code: vec![0],
        mask: vec![0],
    };
    assert!(plane.get_bit(0, 0, 2, false).is_err());
    assert!(plane.put_bit(0, 0, 2, false, true).is_err());
    assert!(plane.get_bit(usize::MAX, 0, 0, false).is_err());
    assert_eq!(plane.code, vec![0]);
    plane.code.clear();
    assert!(plane.get_bit(0, 0, 0, false).is_err());
}
#[test]
fn invalid_filter_confidences_fail_in_public_encoder() {
    let response = FilterResponse {
        rows: 1,
        cols: 1,
        real: vec![1.0],
        imag: vec![1.0],
        mask_real: vec![1.1],
        mask_imag: vec![1.0],
    };
    assert!(
        IrisEncoder::new(0.93)
            .unwrap()
            .encode_planes(&[response])
            .is_err()
    );
}
#[test]
fn different_gabor_and_normalization_configs_cannot_be_compared() {
    let image = NormalizedIris {
        image: GrayImage::new(64, 32, (0..64 * 32).map(|i| (i % 127 + 50) as u8).collect())
            .unwrap(),
        mask: Mask::new(64, 32, vec![true; 64 * 32]).unwrap(),
    };
    let a = FilterBankConfig {
        fragile_thresholds: None,
        ..Default::default()
    };
    let mut b = a.clone();
    b.filters[0].sigma_phi = 8.0;
    let a = FilterBank::new(a)
        .unwrap()
        .encode(&image, EyeSide::Left)
        .unwrap();
    let b = FilterBank::new(b)
        .unwrap()
        .encode(&image, EyeSide::Left)
        .unwrap();
    assert_ne!(a.encoding_profile, b.encoding_profile);
    assert!(
        Matcher::new(MatcherConfig::default())
            .unwrap()
            .compare(&a, &b)
            .is_err()
    );
}

#[test]
fn direct_filter_bank_profile_separates_normalized_image_dimensions() {
    let bank = FilterBank::new(FilterBankConfig {
        fragile_thresholds: None,
        ..Default::default()
    })
    .unwrap();
    let image = |width, height| NormalizedIris {
        image: GrayImage::new(width, height, vec![127; width * height]).unwrap(),
        mask: Mask::new(width, height, vec![true; width * height]).unwrap(),
    };
    let a = bank.encode(&image(64, 32), EyeSide::Left).unwrap();
    let b = bank.encode(&image(128, 64), EyeSide::Left).unwrap();
    assert_ne!(a.encoding_profile, b.encoding_profile);
    assert!(
        Matcher::new(MatcherConfig::default())
            .unwrap()
            .compare(&a, &b)
            .is_err()
    );
}
