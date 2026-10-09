//! Masked Hamming matching for packed iris templates.

use serde::{Deserialize, Serialize};

use crate::{CodePlane, Error, IrisTemplate, Result};

const MAX_ROTATION: usize = 4096;

/// Options for rotational masked Hamming matching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MatcherConfig {
    pub max_rotation: usize,
    pub min_overlap_bits: usize,
    /// Requires an explicit calibration policy, which is not yet implemented.
    pub normalized: bool,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            max_rotation: 15,
            min_overlap_bits: 1024,
            normalized: false,
        }
    }
}

/// The best masked Hamming comparison among the permitted rotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchResult {
    pub distance: f64,
    /// Columns applied to the second argument to align it with the first.
    pub rotation: i32,
    pub valid_bits: usize,
    pub differing_bits: usize,
}

#[derive(Debug, Clone)]
pub struct Matcher {
    config: MatcherConfig,
}

impl Matcher {
    pub fn new(config: MatcherConfig) -> Result<Self> {
        if config.normalized {
            return Err(Error::InvalidInput(
                "normalized matching requires an explicit calibration policy".into(),
            ));
        }
        if config.max_rotation > MAX_ROTATION {
            return Err(Error::InvalidInput(format!(
                "max_rotation must not exceed {MAX_ROTATION}"
            )));
        }
        if config.min_overlap_bits == 0 {
            return Err(Error::InvalidInput(
                "min_overlap_bits must be positive".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn compare(&self, a: &IrisTemplate, b: &IrisTemplate) -> Result<MatchResult> {
        ensure_compatible(a, b)?;
        validate_rotation_bounds(a, self.config.max_rotation)?;

        let mut best = None;
        for rotation in rotations(self.config.max_rotation) {
            let (differing_bits, valid_bits) = score_rotation(a, b, rotation);
            if valid_bits < self.config.min_overlap_bits {
                continue;
            }
            let candidate = MatchResult {
                distance: differing_bits as f64 / valid_bits as f64,
                rotation,
                valid_bits,
                differing_bits,
            };
            if best
                .as_ref()
                .is_none_or(|current: &MatchResult| is_better(&candidate, current))
            {
                best = Some(candidate);
            }
        }
        best.ok_or(Error::NoOverlap)
    }
}

/// Rotates every row by columns. A positive rotation moves a source column to
/// the right, matching `numpy.roll(..., rotation, axis=1)`.
pub fn rotate_template(template: &IrisTemplate, rotation: i32) -> Result<IrisTemplate> {
    template.validate()?;
    let mut rotated = template.clone();
    for plane in &mut rotated.planes {
        let source = plane.clone();
        plane.code.fill(0);
        plane.mask.fill(0);
        for row in 0..plane.rows {
            for col in 0..plane.cols {
                let destination =
                    (col as i64 + rotation as i64).rem_euclid(plane.cols as i64) as usize;
                for component in 0..2 {
                    let code = source.bit(row, col, component, false);
                    let mask = source.bit(row, col, component, true);
                    plane.set_bit(row, destination, component, false, code);
                    plane.set_bit(row, destination, component, true, mask);
                }
            }
        }
    }
    Ok(rotated)
}

pub(crate) fn ensure_compatible(a: &IrisTemplate, b: &IrisTemplate) -> Result<()> {
    if a.version != b.version {
        return Err(Error::IncompatibleTemplate(
            "template versions differ".into(),
        ));
    }
    if a.eye_side != b.eye_side {
        return Err(Error::IncompatibleTemplate("eye sides differ".into()));
    }
    if a.encoding_profile != b.encoding_profile {
        return Err(Error::IncompatibleTemplate(
            "encoding profiles differ".into(),
        ));
    }
    a.validate()?;
    b.validate()?;
    if a.planes.len() != b.planes.len()
        || a.planes
            .iter()
            .zip(&b.planes)
            .any(|(left, right)| left.rows != right.rows || left.cols != right.cols)
    {
        return Err(Error::IncompatibleTemplate(
            "code plane layouts differ".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_rotation_bounds(template: &IrisTemplate, max_rotation: usize) -> Result<()> {
    let smallest_columns = template
        .planes
        .iter()
        .map(|plane| plane.cols)
        .min()
        .unwrap_or(0);
    if max_rotation > smallest_columns / 2 {
        return Err(Error::InvalidInput(
            "max_rotation must not exceed half of every plane width".into(),
        ));
    }
    Ok(())
}

fn rotations(max_rotation: usize) -> impl Iterator<Item = i32> {
    std::iter::once(0)
        .chain((1..=max_rotation).flat_map(|amount| [-(amount as i32), amount as i32]))
}

fn is_better(candidate: &MatchResult, current: &MatchResult) -> bool {
    let left = candidate.differing_bits as u128 * current.valid_bits as u128;
    let right = current.differing_bits as u128 * candidate.valid_bits as u128;
    left < right
}

fn score_rotation(a: &IrisTemplate, b: &IrisTemplate, rotation: i32) -> (usize, usize) {
    a.planes
        .iter()
        .zip(&b.planes)
        .fold((0, 0), |(different, valid), (left, right)| {
            let (plane_different, plane_valid) = score_plane(left, right, rotation);
            (different + plane_different, valid + plane_valid)
        })
}

fn score_plane(left: &CodePlane, right: &CodePlane, rotation: i32) -> (usize, usize) {
    let row_bits = left.cols * 2;
    let mut different = 0;
    let mut valid = 0;
    for row in 0..left.rows {
        let left_start = row * row_bits;
        for offset in (0..row_bits).step_by(64) {
            let width = (row_bits - offset).min(64);
            let left_code = read_bits(&left.code, left_start + offset);
            let left_mask = read_bits(&left.mask, left_start + offset);
            let right_code =
                read_rotated_row(&right.code, left_start, row_bits, offset, rotation * 2);
            let right_mask =
                read_rotated_row(&right.mask, left_start, row_bits, offset, rotation * 2);
            let keep = low_bits(width);
            let overlap = (left_mask & right_mask) & keep;
            valid += overlap.count_ones() as usize;
            different += ((left_code ^ right_code) & overlap).count_ones() as usize;
        }
    }
    (different, valid)
}

fn read_rotated_row(
    words: &[u64],
    row_start: usize,
    row_bits: usize,
    offset: usize,
    rotation_bits: i32,
) -> u64 {
    let source = (offset as i64 - rotation_bits as i64).rem_euclid(row_bits as i64) as usize;
    let first_width = (row_bits - source).min(64);
    let first = read_bits(words, row_start + source) & low_bits(first_width);
    if first_width == 64 {
        first
    } else {
        first | ((read_bits(words, row_start) & low_bits(64 - first_width)) << first_width)
    }
}

fn read_bits(words: &[u64], offset: usize) -> u64 {
    let word = offset / 64;
    let bit = offset % 64;
    let low = words.get(word).copied().unwrap_or(0) >> bit;
    if bit == 0 {
        low
    } else {
        low | (words.get(word + 1).copied().unwrap_or(0) << (64 - bit))
    }
}

fn low_bits(width: usize) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}
