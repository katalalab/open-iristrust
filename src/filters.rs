use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Error, NormalizedIris, Result};

const MAX_FILTERS: usize = 16;
const MAX_PROBES: usize = 8_388_608;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GaborConfig {
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub sigma_phi: f64,
    pub sigma_rho: f64,
    pub theta_degrees: f64,
    pub wavelength: f64,
    pub dc_correction: bool,
    pub fixed_point: bool,
}

impl Default for GaborConfig {
    fn default() -> Self {
        Self {
            kernel_width: 41,
            kernel_height: 21,
            sigma_phi: 7.0,
            sigma_rho: 6.13,
            theta_degrees: 90.0,
            wavelength: 28.0,
            dc_correction: true,
            fixed_point: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaborKernel {
    pub width: usize,
    pub height: usize,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub norm_real: f64,
    pub norm_imag: f64,
}

impl GaborKernel {
    pub fn new(config: &GaborConfig) -> Result<Self> {
        validate_gabor_config(config)?;
        let len = config.kernel_width * config.kernel_height;
        let mut real = Vec::with_capacity(len);
        let mut imag = Vec::with_capacity(len);
        let half_width = (config.kernel_width / 2) as isize;
        let half_height = (config.kernel_height / 2) as isize;
        let theta = config.theta_degrees.to_radians();
        let (sin_theta, cos_theta) = theta.sin_cos();

        for row in -half_height..=half_height {
            for col in -half_width..=half_width {
                let x = row as f64;
                let y = col as f64;
                let rotated_x = x * cos_theta + y * sin_theta;
                let rotated_y = -x * sin_theta + y * cos_theta;
                let envelope = -(rotated_x.powi(2) / config.sigma_phi.powi(2)
                    + rotated_y.powi(2) / config.sigma_rho.powi(2))
                    / 2.0;
                let amplitude = envelope.exp()
                    / (2.0 * std::f64::consts::PI * config.sigma_phi * config.sigma_rho);
                let carrier = 2.0 * std::f64::consts::PI * rotated_x / config.wavelength;
                real.push(amplitude * carrier.cos());
                imag.push(amplitude * carrier.sin());
            }
        }

        if config.dc_correction {
            for row in 0..config.kernel_height {
                let start = row * config.kernel_width;
                let end = start + config.kernel_width;
                let g_mean = real[start..end].iter().sum::<f64>() / config.kernel_width as f64;
                let correction_mean = (-half_width..=half_width)
                    .map(|col| {
                        let x = row as isize - half_height;
                        let y = col as f64;
                        let x = x as f64;
                        let rotated_x = x * cos_theta + y * sin_theta;
                        let rotated_y = -x * sin_theta + y * cos_theta;
                        -(rotated_x.powi(2) / config.sigma_phi.powi(2)
                            + rotated_y.powi(2) / config.sigma_rho.powi(2))
                            / 2.0
                    })
                    .sum::<f64>()
                    / config.kernel_width as f64;
                if correction_mean == 0.0 || !correction_mean.is_finite() {
                    return Err(Error::InvalidInput(
                        "Gabor DC correction is undefined".into(),
                    ));
                }
                for (offset, value) in real[start..end].iter_mut().enumerate() {
                    let col = offset as isize - half_width;
                    let x = row as isize - half_height;
                    let y = col as f64;
                    let x = x as f64;
                    let rotated_x = x * cos_theta + y * sin_theta;
                    let rotated_y = -x * sin_theta + y * cos_theta;
                    let envelope = -(rotated_x.powi(2) / config.sigma_phi.powi(2)
                        + rotated_y.powi(2) / config.sigma_rho.powi(2))
                        / 2.0;
                    *value -= g_mean / correction_mean * envelope;
                }
            }
        }

        normalize(&mut real)?;
        normalize(&mut imag)?;
        if config.fixed_point {
            for value in real.iter_mut().chain(imag.iter_mut()) {
                *value = (*value * 32_768.0).round_ties_even();
            }
        }
        let norm_real = frobenius_norm(&real);
        let norm_imag = frobenius_norm(&imag);
        if norm_real == 0.0 || norm_imag == 0.0 || !norm_real.is_finite() || !norm_imag.is_finite()
        {
            return Err(Error::InvalidInput(
                "Gabor kernel has a zero or non-finite component".into(),
            ));
        }

        Ok(Self {
            width: config.kernel_width,
            height: config.kernel_height,
            real,
            imag,
            norm_real,
            norm_imag,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterBankConfig {
    pub filters: Vec<GaborConfig>,
    pub probe_rows: usize,
    pub probe_cols: usize,
    pub mask_threshold: f64,
    #[serde(default = "default_fragile_thresholds")]
    pub fragile_thresholds: Option<[f64; 3]>,
}

impl Default for FilterBankConfig {
    fn default() -> Self {
        Self {
            filters: vec![
                GaborConfig::default(),
                GaborConfig {
                    kernel_width: 17,
                    kernel_height: 21,
                    sigma_phi: 2.0,
                    sigma_rho: 5.86,
                    theta_degrees: 90.0,
                    wavelength: 8.0,
                    dc_correction: true,
                    fixed_point: true,
                },
            ],
            probe_rows: 16,
            probe_cols: 256,
            mask_threshold: 0.93,
            fragile_thresholds: default_fragile_thresholds(),
        }
    }
}

fn default_fragile_thresholds() -> Option<[f64; 3]> {
    Some([0.0001, 0.275, 0.087_266_462_599_716_47])
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterResponse {
    pub rows: usize,
    pub cols: usize,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub mask_real: Vec<f64>,
    pub mask_imag: Vec<f64>,
}

impl FilterResponse {
    pub fn validate(&self) -> Result<()> {
        let len = self
            .rows
            .checked_mul(self.cols)
            .filter(|&len| self.rows > 0 && self.cols > 0 && len <= MAX_PROBES)
            .ok_or_else(|| Error::InvalidInput("invalid filter response dimensions".into()))?;
        if [
            self.real.len(),
            self.imag.len(),
            self.mask_real.len(),
            self.mask_imag.len(),
        ]
        .iter()
        .any(|&actual| actual != len)
        {
            return Err(Error::InvalidInput(
                "filter response data does not match dimensions".into(),
            ));
        }
        if self
            .real
            .iter()
            .chain(&self.imag)
            .chain(&self.mask_real)
            .chain(&self.mask_imag)
            .any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "filter response contains non-finite values".into(),
            ));
        }
        if self
            .mask_real
            .iter()
            .chain(&self.mask_imag)
            .any(|v| !(0.0..=1.0).contains(v))
        {
            return Err(Error::InvalidInput("filter masks must be in [0,1]".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FilterBank {
    config: FilterBankConfig,
    kernels: Vec<GaborKernel>,
}

impl FilterBank {
    pub fn config(&self) -> &FilterBankConfig {
        &self.config
    }
    pub fn kernels(&self) -> &[GaborKernel] {
        &self.kernels
    }
    pub fn new(config: FilterBankConfig) -> Result<Self> {
        validate_filter_bank_config(&config)?;
        let kernels = config
            .filters
            .iter()
            .map(GaborKernel::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { config, kernels })
    }

    pub fn respond(&self, iris: &NormalizedIris) -> Result<Vec<FilterResponse>> {
        iris.validate()?;
        if iris.image.width < 2 || iris.image.height < 2 {
            return Err(Error::InvalidInput(
                "normalized iris dimensions are too small for filtering".into(),
            ));
        }
        let samples = self.config.probe_rows * self.config.probe_cols;
        self.kernels
            .iter()
            .map(|kernel| {
                let mut response = FilterResponse {
                    rows: self.config.probe_rows,
                    cols: self.config.probe_cols,
                    real: vec![0.0; samples],
                    imag: vec![0.0; samples],
                    mask_real: vec![0.0; samples],
                    mask_imag: vec![0.0; samples],
                };
                response
                    .real
                    .par_iter_mut()
                    .zip(response.imag.par_iter_mut())
                    .zip(response.mask_real.par_iter_mut())
                    .zip(response.mask_imag.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (((real, imag), mask_real), mask_imag))| {
                        let row = index / self.config.probe_cols;
                        let col = index % self.config.probe_cols;
                        (*real, *imag, *mask_real, *mask_imag) =
                            self.convolve_probe(kernel, iris, row, col);
                    });
                Ok(response)
            })
            .collect()
    }

    pub fn encode(
        &self,
        iris: &NormalizedIris,
        eye_side: crate::EyeSide,
    ) -> Result<crate::IrisTemplate> {
        let mut responses = self.respond(iris)?;
        if let Some(thresholds) = self.config.fragile_thresholds {
            for response in &mut responses {
                apply_fragile_bit_refinement(response, thresholds);
            }
        }
        let planes = crate::encoder::IrisEncoder::new(self.config.mask_threshold)?
            .encode_planes(&responses)?;
        let template = crate::IrisTemplate {
            version: crate::TEMPLATE_VERSION.into(),
            encoding_profile: crate::profile::fingerprint(
                "gabor",
                &(iris.image.width, iris.image.height, &self.config),
            )?,
            eye_side,
            planes,
        };
        template.validate()?;
        Ok(template)
    }

    fn convolve_probe(
        &self,
        kernel: &GaborKernel,
        iris: &NormalizedIris,
        probe_row: usize,
        probe_col: usize,
    ) -> (f64, f64, f64, f64) {
        let image_rows = iris.image.height;
        let image_cols = iris.image.width;
        let kernel_row_padding = kernel.height / 2;
        let kernel_col_padding = kernel.width / 2;
        let rho = probe_row as f64 * 0.9375 / (self.config.probe_rows - 1) as f64;
        let phi = probe_col as f64 / self.config.probe_cols as f64;
        let center_row = round_ties_even(rho * image_rows as f64).min(image_rows - 1);
        let center_col = round_ties_even(phi * image_cols as f64).min(image_cols - 1);
        let top = center_row.saturating_sub(kernel_row_padding);
        let bottom = (center_row + kernel_row_padding + 1).min(image_rows);
        let patch_rows = bottom - top;
        let kernel_top = kernel_row_padding - patch_rows / 2;
        let mut real = 0.0;
        let mut imag = 0.0;
        let mut abs_real_total = 0.0;
        let mut abs_imag_total = 0.0;
        let mut abs_real_valid = 0.0;
        let mut abs_imag_valid = 0.0;
        let mut all_valid = true;

        for patch_row in 0..patch_rows {
            let image_row = top + patch_row;
            let kernel_row = kernel_top + patch_row;
            for kernel_col in 0..kernel.width {
                let image_col =
                    (center_col + kernel_col + image_cols - kernel_col_padding) % image_cols;
                let pixel_index = image_row * image_cols + image_col;
                let kernel_index = kernel_row * kernel.width + kernel_col;
                let pixel = iris.image.data[pixel_index] as f64;
                let real_weight = kernel.real[kernel_index];
                let imag_weight = kernel.imag[kernel_index];
                real += pixel * real_weight;
                imag += pixel * imag_weight;
                let real_abs = real_weight.abs();
                let imag_abs = imag_weight.abs();
                abs_real_total += real_abs;
                abs_imag_total += imag_abs;
                if iris.mask.data[pixel_index] {
                    abs_real_valid += real_abs;
                    abs_imag_valid += imag_abs;
                } else {
                    all_valid = false;
                }
            }
        }
        let scale = (patch_rows * kernel.width) as f64;
        real = real / scale / kernel.norm_real;
        imag = imag / scale / kernel.norm_imag;
        if real == 0.0 && imag == 0.0 {
            return (real, imag, 0.0, 0.0);
        }
        let mask_real = if all_valid {
            1.0
        } else {
            abs_real_valid / abs_real_total
        };
        let mask_imag = if all_valid {
            1.0
        } else {
            abs_imag_valid / abs_imag_total
        };
        (real, imag, mask_real, mask_imag)
    }
}

pub(crate) fn apply_fragile_bit_refinement(response: &mut FilterResponse, thresholds: [f64; 3]) {
    let angle_limit = thresholds[2].cos().abs();
    for index in 0..response.real.len() {
        let real = response.real[index];
        let imag = response.imag[index];
        let radius = real.hypot(imag);
        if radius < thresholds[0] || radius > thresholds[1] {
            response.mask_real[index] = 0.0;
            response.mask_imag[index] = 0.0;
            continue;
        }
        let angle = imag.atan2(real);
        if angle.sin().abs() > angle_limit {
            response.mask_real[index] = 0.0;
        }
        if angle.cos().abs() > angle_limit {
            response.mask_imag[index] = 0.0;
        }
    }
}

fn validate_gabor_config(config: &GaborConfig) -> Result<()> {
    if !(5..=97).contains(&config.kernel_width)
        || !(5..=97).contains(&config.kernel_height)
        || config.kernel_width.is_multiple_of(2)
        || config.kernel_height.is_multiple_of(2)
    {
        return Err(Error::InvalidInput(
            "Gabor kernel dimensions must be odd values from 5 through 97".into(),
        ));
    }
    if !config.sigma_phi.is_finite()
        || !config.sigma_rho.is_finite()
        || !config.theta_degrees.is_finite()
        || !config.wavelength.is_finite()
        || config.sigma_phi < 1.0
        || config.sigma_rho < 1.0
        || config.sigma_phi >= config.kernel_width as f64
        || config.sigma_rho >= config.kernel_height as f64
        || !(0.0..360.0).contains(&config.theta_degrees)
        || config.wavelength < 2.0
        || config.wavelength >= config.kernel_width as f64
    {
        return Err(Error::InvalidInput("invalid Gabor parameters".into()));
    }
    Ok(())
}

fn validate_filter_bank_config(config: &FilterBankConfig) -> Result<()> {
    let probes = config
        .probe_rows
        .checked_mul(config.probe_cols)
        .filter(|&count| count <= MAX_PROBES)
        .ok_or_else(|| Error::InvalidInput("filter bank probe count is too large".into()))?;
    if config.filters.is_empty()
        || config.filters.len() > MAX_FILTERS
        || config.probe_rows < 2
        || config.probe_cols < 2
        || probes == 0
        || !config.mask_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.mask_threshold)
    {
        return Err(Error::InvalidInput(
            "invalid filter bank configuration".into(),
        ));
    }
    if let Some(thresholds) = config.fragile_thresholds
        && (thresholds
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || thresholds[0] > thresholds[1]
            || thresholds[2] > std::f64::consts::FRAC_PI_2)
    {
        return Err(Error::InvalidInput("invalid fragile-bit thresholds".into()));
    }
    Ok(())
}

fn normalize(values: &mut [f64]) -> Result<()> {
    let norm = frobenius_norm(values);
    if norm == 0.0 || !norm.is_finite() {
        return Err(Error::InvalidInput(
            "Gabor component cannot be normalized".into(),
        ));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn frobenius_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn round_ties_even(value: f64) -> usize {
    value.round_ties_even().max(0.0) as usize
}
