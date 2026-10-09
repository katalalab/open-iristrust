use iristrust::aggregation::{AggregationConfig, aggregate, aggregate_with_report};
use iristrust::matcher::{Matcher, MatcherConfig, rotate_template};
use iristrust::{CodePlane, Error, EyeSide, IrisTemplate, TEMPLATE_VERSION};

fn template(rows: usize, cols: usize, code: u64, mask: u64) -> IrisTemplate {
    let bits = rows * cols * 2;
    let words = bits.div_ceil(64);
    let mut plane = CodePlane {
        rows,
        cols,
        code: vec![0; words],
        mask: vec![0; words],
    };
    for i in 0..bits {
        if (code >> (i % 64)) & 1 != 0 {
            plane.code[i / 64] |= 1_u64 << (i % 64);
        }
        if (mask >> (i % 64)) & 1 != 0 {
            plane.mask[i / 64] |= 1_u64 << (i % 64);
        }
    }
    IrisTemplate {
        version: TEMPLATE_VERSION.into(),
        encoding_profile: "test-profile-v1".into(),
        eye_side: EyeSide::Left,
        planes: vec![plane],
    }
}

fn patterned_template(rows: usize, cols: usize, seed: u64) -> IrisTemplate {
    let bits = rows * cols * 2;
    let mut state = seed;
    let words = bits.div_ceil(64);
    let mut plane = CodePlane {
        rows,
        cols,
        code: vec![0; words],
        mask: vec![0; words],
    };
    for row in 0..rows {
        for col in 0..cols {
            for component in 0..2 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                plane
                    .put_bit(row, col, component, false, state >> 63 != 0)
                    .unwrap();
                plane
                    .put_bit(row, col, component, true, (state >> 8) & 7 != 0)
                    .unwrap();
            }
        }
    }
    IrisTemplate {
        version: TEMPLATE_VERSION.into(),
        encoding_profile: "test-profile-v1".into(),
        eye_side: EyeSide::Left,
        planes: vec![plane],
    }
}

fn naive_score(a: &IrisTemplate, b: &IrisTemplate, rotation: i32) -> (usize, usize) {
    let mut differing = 0;
    let mut valid = 0;
    for (a_plane, b_plane) in a.planes.iter().zip(&b.planes) {
        for row in 0..a_plane.rows {
            for col in 0..a_plane.cols {
                let b_col = (col as i32 - rotation).rem_euclid(a_plane.cols as i32) as usize;
                for component in 0..2 {
                    if a_plane.get_bit(row, col, component, true).unwrap()
                        && b_plane.get_bit(row, b_col, component, true).unwrap()
                    {
                        valid += 1;
                        differing += (a_plane.get_bit(row, col, component, false).unwrap()
                            != b_plane.get_bit(row, b_col, component, false).unwrap())
                            as usize;
                    }
                }
            }
        }
    }
    (differing, valid)
}

#[test]
fn packed_matching_matches_naive_oracle_across_words_and_rotations() {
    let a = patterned_template(3, 37, 7);
    let b = patterned_template(3, 37, 99);
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: 9,
        min_overlap_bits: 1,
        normalized: false,
    })
    .unwrap();
    let result = matcher.compare(&a, &b).unwrap();

    let mut expected = None;
    for rotation in [
        0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9,
    ] {
        let (differing, valid) = naive_score(&a, &b, rotation);
        let candidate = (differing as f64 / valid as f64, rotation, differing, valid);
        if expected.is_none_or(|prior: (f64, i32, usize, usize)| candidate.0 < prior.0) {
            expected = Some(candidate);
        }
    }
    let (distance, rotation, differing, valid) = expected.unwrap();
    assert_eq!(result.rotation, rotation);
    assert_eq!(result.differing_bits, differing);
    assert_eq!(result.valid_bits, valid);
    assert_eq!(result.distance, distance);
}

#[test]
fn masked_mismatches_do_not_contribute_to_distance() {
    let a = template(1, 32, 0, u64::MAX);
    let b = template(1, 32, u64::MAX, 1);
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: 0,
        min_overlap_bits: 1,
        normalized: false,
    })
    .unwrap();
    let result = matcher.compare(&a, &b).unwrap();
    assert_eq!(result.valid_bits, 1);
    assert_eq!(result.differing_bits, 1);
    assert_eq!(result.distance, 1.0);
}

#[test]
fn excludes_rotations_without_required_overlap() {
    let a = template(1, 4, 0, 1);
    let b = rotate_template(&a, 1).unwrap();
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: 1,
        min_overlap_bits: 2,
        normalized: false,
    })
    .unwrap();
    assert!(matches!(matcher.compare(&a, &b), Err(Error::NoOverlap)));
}

#[test]
fn rejects_incompatible_templates() {
    let a = template(1, 4, 0, u64::MAX);
    let mut other_eye = a.clone();
    other_eye.eye_side = EyeSide::Right;
    let mut other_profile = a.clone();
    other_profile.encoding_profile = "other-profile-v1".into();
    let mut different_layout = a.clone();
    different_layout.planes[0].cols = 2;
    different_layout.planes[0].code = vec![0];
    different_layout.planes[0].mask = vec![0b1111];
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: 0,
        min_overlap_bits: 1,
        normalized: false,
    })
    .unwrap();
    assert!(matches!(
        matcher.compare(&a, &other_eye),
        Err(Error::IncompatibleTemplate(_))
    ));
    assert!(matches!(
        matcher.compare(&a, &other_profile),
        Err(Error::IncompatibleTemplate(_))
    ));
    assert!(matches!(
        matcher.compare(&a, &different_layout),
        Err(Error::IncompatibleTemplate(_))
    ));
}

#[test]
fn rejects_uncalibrated_normalized_matching() {
    assert!(matches!(
        Matcher::new(MatcherConfig {
            max_rotation: 0,
            min_overlap_bits: 1,
            normalized: true,
        }),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn aggregation_validates_matcher_and_rotation_bounds_before_single_frame_return() {
    let source = template(1, 4, 0, u64::MAX);
    for config in [
        AggregationConfig {
            min_overlap_bits: 0,
            ..AggregationConfig::default()
        },
        AggregationConfig {
            max_rotation: 4097,
            min_overlap_bits: 1,
            ..AggregationConfig::default()
        },
        AggregationConfig {
            max_rotation: 3,
            min_overlap_bits: 1,
            ..AggregationConfig::default()
        },
    ] {
        assert!(matches!(
            aggregate(std::slice::from_ref(&source), &config),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
fn aggregation_rejects_single_frames_without_required_quality_or_vote_count() {
    let valid = template(1, 4, 0, u64::MAX);
    let empty = template(1, 4, 0, 0);
    let usable = AggregationConfig {
        max_rotation: 0,
        min_overlap_bits: 1,
        min_valid_frames: 1,
        max_distance: 0.35,
    };
    assert_eq!(
        aggregate(std::slice::from_ref(&valid), &usable).unwrap(),
        valid
    );
    assert!(matches!(
        aggregate(&[empty], &usable),
        Err(Error::NoOverlap)
    ));
    assert!(matches!(
        aggregate(
            &[valid],
            &AggregationConfig {
                min_valid_frames: 2,
                ..usable
            }
        ),
        Err(Error::Quality(_))
    ));
}

#[test]
fn rotation_is_invertible_and_matching_reports_the_alignment_sign() {
    let a = patterned_template(2, 35, 44);
    let shifted = rotate_template(&a, 7).unwrap();
    assert_eq!(rotate_template(&shifted, -7).unwrap(), a);
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: 8,
        min_overlap_bits: 1,
        normalized: false,
    })
    .unwrap();
    let result = matcher.compare(&a, &shifted).unwrap();
    assert_eq!(result.distance, 0.0);
    assert_eq!(result.rotation, -7);
}

#[test]
fn packed_rotation_matches_per_bit_oracle_across_word_boundaries() {
    let original = patterned_template(3, 37, 0x5eeda11c);
    for rotation in [-18, -9, -1, 0, 1, 9, 18] {
        let rotated = rotate_template(&original, rotation).unwrap();
        for row in 0..3 {
            for source_col in 0..37 {
                let destination = (source_col as i32 + rotation).rem_euclid(37) as usize;
                for component in 0..2 {
                    assert_eq!(
                        rotated.planes[0]
                            .get_bit(row, destination, component, false)
                            .unwrap(),
                        original.planes[0]
                            .get_bit(row, source_col, component, false)
                            .unwrap()
                    );
                    assert_eq!(
                        rotated.planes[0]
                            .get_bit(row, destination, component, true)
                            .unwrap(),
                        original.planes[0]
                            .get_bit(row, source_col, component, true)
                            .unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn aggregation_aligns_frames_masks_ties_and_rejects_outliers() {
    let base = patterned_template(1, 35, 123);
    let shifted = rotate_template(&base, 3).unwrap();
    let mut opposite = base.clone();
    for word in &mut opposite.planes[0].code {
        *word = !*word;
    }
    // Restore valid packed padding after flipping code words.
    let bits = opposite.planes[0].bit_len().unwrap();
    let tail = bits % 64;
    if tail != 0 {
        *opposite.planes[0].code.last_mut().unwrap() &= (1_u64 << tail) - 1;
    }
    let config = AggregationConfig {
        max_rotation: 4,
        min_overlap_bits: 1,
        min_valid_frames: 2,
        max_distance: 0.1,
    };
    let report = aggregate_with_report(&[base.clone(), shifted, opposite], &config).unwrap();
    assert_eq!(report.accepted_frames, vec![0, 1]);
    assert_eq!(report.rejected_frames, vec![2]);
    assert_eq!(report.rotations, vec![0, -3]);
    assert_eq!(report.template.planes[0].mask, base.planes[0].mask);
    for row in 0..1 {
        for col in 0..35 {
            for component in 0..2 {
                if base.planes[0].get_bit(row, col, component, true).unwrap() {
                    assert_eq!(
                        report.template.planes[0]
                            .get_bit(row, col, component, false)
                            .unwrap(),
                        base.planes[0].get_bit(row, col, component, false).unwrap()
                    );
                }
            }
        }
    }

    let mut inverse = base.clone();
    for word in &mut inverse.planes[0].code {
        *word = !*word;
    }
    if tail != 0 {
        *inverse.planes[0].code.last_mut().unwrap() &= (1_u64 << tail) - 1;
    }
    assert!(matches!(
        aggregate(
            &[base, inverse],
            &AggregationConfig {
                max_rotation: 0,
                min_valid_frames: 1,
                max_distance: 1.0,
                ..config
            },
        ),
        Err(Error::NoOverlap)
    ));
}
