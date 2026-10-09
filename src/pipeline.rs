//! Reusable pipeline; the segmentation session is supplied by the caller.
use crate::{
    Error, EyeSide, Geometry, GeometryMasks, GrayImage, IrisTemplate, Mask, NormalizedIris, Result,
    filters::{FilterBank, FilterBankConfig},
    geometry::{GeometryConfig, estimate_geometry},
    normalization::{NormalizationConfig, normalize},
    quality::{QualityConfig, QualityMetrics, gate_quality, measure_quality},
    segmentation::Segmenter,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PipelineConfig {
    pub geometry: GeometryConfig,
    pub normalization: NormalizationConfig,
    pub filters: FilterBankConfig,
    pub quality: QualityConfig,
    pub encoding: EncodingConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EncodingConfig {
    pub min_valid_bits: usize,
    pub specular_threshold: u8,
}
impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            min_valid_bits: 1024,
            specular_threshold: 254,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimings {
    pub geometry_ms: f64,
    pub normalization_ms: f64,
    pub encoding_ms: f64,
    pub total_ms: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionReport {
    pub quality: QualityMetrics,
    pub timings: StageTimings,
    pub valid_bits: usize,
    pub total_bits: usize,
}
pub struct Extraction {
    pub template: IrisTemplate,
    pub geometry: Geometry,
    pub normalized: NormalizedIris,
    pub report: ExtractionReport,
}
pub struct IrisPipeline {
    config: PipelineConfig,
    filters: FilterBank,
}
impl IrisPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        if config.encoding.min_valid_bits == 0 || config.encoding.specular_threshold == 0 {
            return Err(Error::InvalidInput(
                "positive encoding bit count and reflection threshold are required".into(),
            ));
        }
        let filters = FilterBank::new(config.filters.clone())?;
        Ok(Self { config, filters })
    }
    pub fn extract(
        &self,
        image: &GrayImage,
        side: EyeSide,
        segmenter: &mut impl Segmenter,
    ) -> Result<Extraction> {
        let start = Instant::now();
        let masks = segmenter.segment(image)?;
        let mut result = self.extract_from_masks(image, side, &masks)?;
        result.report.timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }
    pub fn extract_from_masks(
        &self,
        image: &GrayImage,
        side: EyeSide,
        masks: &GeometryMasks,
    ) -> Result<Extraction> {
        image.validate()?;
        masks.validate()?;
        if (image.width, image.height) != (masks.iris.width, masks.iris.height) {
            return Err(Error::InvalidInput(
                "image and segmentation dimensions differ".into(),
            ));
        }
        let start = Instant::now();
        let geometry = estimate_geometry(masks, &self.config.geometry)?;
        let geometry_ms = start.elapsed().as_secs_f64() * 1000.0;
        // Iris predictions can be annular; fill the pupil hole for visibility.
        let visible = Mask::new(
            image.width,
            image.height,
            masks
                .eyeball
                .data
                .iter()
                .zip(&masks.iris.data)
                .zip(&masks.pupil.data)
                .map(|((&eye, &iris), &pupil)| eye && (iris || pupil))
                .collect(),
        )?;
        let mut result =
            self.extract_with_geometry(image, side, &geometry, Some(&masks.noise), Some(&visible))?;
        result.report.timings.geometry_ms = geometry_ms;
        result.report.timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }
    pub fn extract_with_geometry(
        &self,
        image: &GrayImage,
        side: EyeSide,
        geometry: &Geometry,
        noise: Option<&Mask>,
        visible: Option<&Mask>,
    ) -> Result<Extraction> {
        image.validate()?;
        geometry.validate()?;
        let start = Instant::now();
        if let Some(mask) = noise {
            mask.validate()?;
            if (mask.width, mask.height) != (image.width, image.height) {
                return Err(Error::InvalidInput("noise dimensions differ".into()));
            }
        }
        let reflection_noise = Mask::new(
            image.width,
            image.height,
            image
                .data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    v >= self.config.encoding.specular_threshold || noise.is_some_and(|m| m.data[i])
                })
                .collect(),
        )?;
        let normalized = normalize(
            image,
            geometry,
            &reflection_noise,
            visible,
            &self.config.normalization,
        )?;
        let normalization_ms = start.elapsed().as_secs_f64() * 1000.0;
        let quality = measure_quality(&normalized, geometry)?;
        gate_quality(&quality, &self.config.quality)?;
        let encoding_start = Instant::now();
        let mut template = self.encode_normalized(&normalized, side)?;
        template.encoding_profile = crate::profile::fingerprint(
            "rubbersheet-v1",
            &(&self.config.normalization, &self.config.filters),
        )?;
        let encoding_ms = encoding_start.elapsed().as_secs_f64() * 1000.0;
        let total_bits = template
            .planes
            .iter()
            .map(|p| p.bit_len())
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum();
        let report = ExtractionReport {
            quality,
            valid_bits: template.valid_bits(),
            total_bits,
            timings: StageTimings {
                geometry_ms: 0.0,
                normalization_ms,
                encoding_ms,
                total_ms: start.elapsed().as_secs_f64() * 1000.0,
            },
        };
        Ok(Extraction {
            template,
            geometry: geometry.clone(),
            normalized,
            report,
        })
    }
    pub fn encode_normalized(&self, image: &NormalizedIris, side: EyeSide) -> Result<IrisTemplate> {
        image.validate()?;
        let mut template = self.filters.encode(image, side)?;
        template.encoding_profile = crate::profile::fingerprint(
            "external-normalized",
            &(image.image.width, image.image.height, &self.config.filters),
        )?;
        if template.valid_bits() < self.config.encoding.min_valid_bits {
            return Err(Error::Quality(format!(
                "valid template bits {} < {}",
                template.valid_bits(),
                self.config.encoding.min_valid_bits
            )));
        }
        Ok(template)
    }
}
