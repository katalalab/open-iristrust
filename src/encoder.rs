use crate::filters::FilterResponse;
use crate::{CodePlane, Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct IrisEncoder {
    mask_threshold: f64,
}

impl IrisEncoder {
    pub fn new(mask_threshold: f64) -> Result<Self> {
        if !mask_threshold.is_finite() || !(0.0..=1.0).contains(&mask_threshold) {
            return Err(Error::InvalidInput(
                "mask threshold must be finite and in [0, 1]".into(),
            ));
        }
        Ok(Self { mask_threshold })
    }

    /// Encode phase planes; the filter/normalization owner must assign a template profile.
    pub fn encode_planes(&self, responses: &[FilterResponse]) -> Result<Vec<CodePlane>> {
        if responses.is_empty() || responses.len() > 16 {
            return Err(Error::InvalidInput(
                "filter response count must be from 1 through 16".into(),
            ));
        }
        let mut planes = Vec::with_capacity(responses.len());
        for response in responses {
            response.validate()?;
            let bit_len = response
                .rows
                .checked_mul(response.cols)
                .and_then(|count| count.checked_mul(2))
                .ok_or_else(|| Error::InvalidInput("filter response dimensions overflow".into()))?;
            let mut plane = CodePlane {
                rows: response.rows,
                cols: response.cols,
                code: vec![0; bit_len.div_ceil(64)],
                mask: vec![0; bit_len.div_ceil(64)],
            };
            for index in 0..response.real.len() {
                set_packed_bit(&mut plane.code, index * 2, response.real[index] > 0.0);
                set_packed_bit(&mut plane.code, index * 2 + 1, response.imag[index] > 0.0);
                set_packed_bit(
                    &mut plane.mask,
                    index * 2,
                    response.mask_real[index] >= self.mask_threshold,
                );
                set_packed_bit(
                    &mut plane.mask,
                    index * 2 + 1,
                    response.mask_imag[index] >= self.mask_threshold,
                );
            }
            plane.validate()?;
            planes.push(plane);
        }
        Ok(planes)
    }
}

fn set_packed_bit(words: &mut [u64], bit: usize, value: bool) {
    if value {
        words[bit / 64] |= 1_u64 << (bit % 64);
    }
}
