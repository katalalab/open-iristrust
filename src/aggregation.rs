//! Alignment and majority-vote aggregation for iris templates.

use serde::{Deserialize, Serialize};

use crate::matcher::{
    Matcher, MatcherConfig, ensure_compatible, rotate_template, validate_rotation_bounds,
};
use crate::{Error, IrisTemplate, Result};

/// Controls frame acceptance and per-bit voting during aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AggregationConfig {
    pub max_rotation: usize,
    pub min_overlap_bits: usize,
    pub min_valid_frames: usize,
    pub max_distance: f64,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            max_rotation: 15,
            min_overlap_bits: 1024,
            min_valid_frames: 1,
            max_distance: 0.35,
        }
    }
}

/// Aggregation output, including the frame decisions that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregationResult {
    pub template: IrisTemplate,
    pub accepted_frames: Vec<usize>,
    pub rejected_frames: Vec<usize>,
    /// Rotation for each accepted non-reference frame, in accepted-frame order.
    pub rotations: Vec<i32>,
}

pub fn aggregate(templates: &[IrisTemplate], config: &AggregationConfig) -> Result<IrisTemplate> {
    Ok(aggregate_with_report(templates, config)?.template)
}

pub fn aggregate_with_report(
    templates: &[IrisTemplate],
    config: &AggregationConfig,
) -> Result<AggregationResult> {
    validate_config(config)?;
    let matcher = Matcher::new(MatcherConfig {
        max_rotation: config.max_rotation,
        min_overlap_bits: config.min_overlap_bits,
        normalized: false,
    })?;
    let reference = templates
        .first()
        .ok_or_else(|| Error::InvalidInput("at least one template is required".into()))?;
    reference.validate()?;
    validate_rotation_bounds(reference, config.max_rotation)?;
    if reference.valid_bits() < config.min_overlap_bits {
        return Err(Error::NoOverlap);
    }
    for template in &templates[1..] {
        ensure_compatible(reference, template)?;
    }
    if templates.len() == 1 {
        if config.min_valid_frames > 1 {
            return Err(Error::Quality(
                "fewer accepted frames than min_valid_frames".into(),
            ));
        }
        return Ok(AggregationResult {
            template: reference.clone(),
            accepted_frames: vec![0],
            rejected_frames: Vec::new(),
            rotations: vec![0],
        });
    }

    let mut accepted = vec![reference.clone()];
    let mut accepted_frames = vec![0];
    let mut rejected_frames = Vec::new();
    let mut rotations = vec![0];
    for (index, template) in templates.iter().enumerate().skip(1) {
        match matcher.compare(reference, template) {
            Ok(result) if result.distance <= config.max_distance => {
                accepted.push(rotate_template(template, result.rotation)?);
                accepted_frames.push(index);
                rotations.push(result.rotation);
            }
            Ok(_) | Err(Error::NoOverlap) => rejected_frames.push(index),
            Err(error) => return Err(error),
        }
    }

    if accepted.len() < config.min_valid_frames {
        return Err(Error::Quality(
            "fewer accepted frames than min_valid_frames".into(),
        ));
    }
    let template = majority_vote(reference, &accepted, config.min_valid_frames);
    if template.valid_bits() < config.min_overlap_bits {
        return Err(Error::NoOverlap);
    }
    Ok(AggregationResult {
        template,
        accepted_frames,
        rejected_frames,
        rotations,
    })
}

fn validate_config(config: &AggregationConfig) -> Result<()> {
    if config.min_valid_frames == 0 {
        return Err(Error::InvalidInput(
            "min_valid_frames must be positive".into(),
        ));
    }
    if !config.max_distance.is_finite() || !(0.0..=1.0).contains(&config.max_distance) {
        return Err(Error::InvalidInput(
            "max_distance must be finite and within zero through one".into(),
        ));
    }
    Ok(())
}

fn majority_vote(
    reference: &IrisTemplate,
    aligned: &[IrisTemplate],
    min_valid_frames: usize,
) -> IrisTemplate {
    let mut output = reference.clone();
    for plane in &mut output.planes {
        plane.code.fill(0);
        plane.mask.fill(0);
    }
    for plane_index in 0..output.planes.len() {
        let output_plane = &mut output.planes[plane_index];
        for row in 0..output_plane.rows {
            for col in 0..output_plane.cols {
                for component in 0..2 {
                    let mut valid_votes = 0;
                    let mut one_votes = 0;
                    for template in aligned {
                        let plane = &template.planes[plane_index];
                        if plane.bit(row, col, component, true) {
                            valid_votes += 1;
                            one_votes += plane.bit(row, col, component, false) as usize;
                        }
                    }
                    if valid_votes >= min_valid_frames && one_votes * 2 != valid_votes {
                        output_plane.set_bit(row, col, component, true, true);
                        output_plane.set_bit(
                            row,
                            col,
                            component,
                            false,
                            one_votes * 2 > valid_votes,
                        );
                    }
                }
            }
        }
    }
    output
}
