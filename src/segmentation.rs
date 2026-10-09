//! Four-label segmentation with explicit preprocessing and no network access.
use crate::{Error, GeometryMasks, GrayImage, MAX_PIXELS, Raster, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SegmentationConfig {
    pub input_width: usize,
    pub input_height: usize,
    pub thresholds: [f32; 4],
    /// Bilateral filtering of pixels >=75. Disabled by default for throughput.
    pub denoise: bool,
    pub threads: usize,
}
impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            input_width: 640,
            input_height: 480,
            thresholds: [0.5; 4],
            denoise: false,
            threads: 4,
        }
    }
}
impl SegmentationConfig {
    pub fn validate(&self) -> Result<()> {
        let n = self
            .input_width
            .checked_mul(self.input_height)
            .filter(|&n| n > 0 && n <= MAX_PIXELS)
            .ok_or_else(|| Error::InvalidInput("invalid model resolution".into()))?;
        if n > MAX_PIXELS / 3
            || self.threads == 0
            || self.threads > 256
            || self
                .thresholds
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(Error::InvalidInput(
                "invalid segmentation configuration".into(),
            ));
        }
        Ok(())
    }
}

pub type SegmentationMap = Raster<[f32; 4]>;
pub trait Segmenter {
    fn segment(&mut self, image: &GrayImage) -> Result<GeometryMasks>;
}

/// OpenCV-style half-pixel bilinear resize followed by ImageNet normalization.
pub fn preprocess(image: &GrayImage, config: &SegmentationConfig) -> Result<Vec<f32>> {
    image.validate()?;
    config.validate()?;
    let (w, h) = (config.input_width, config.input_height);
    let mut resized = vec![0_u8; w * h];
    for y in 0..h {
        let sy = ((y as f64 + 0.5) * image.height as f64 / h as f64 - 0.5)
            .clamp(0.0, (image.height - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(image.height - 1);
        let fy = sy - y0 as f64;
        for x in 0..w {
            let sx = ((x as f64 + 0.5) * image.width as f64 / w as f64 - 0.5)
                .clamp(0.0, (image.width - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(image.width - 1);
            let fx = sx - x0 as f64;
            let a = image.data[y0 * image.width + x0] as f64 * (1.0 - fx)
                + image.data[y0 * image.width + x1] as f64 * fx;
            let b = image.data[y1 * image.width + x0] as f64 * (1.0 - fx)
                + image.data[y1 * image.width + x1] as f64 * fx;
            resized[y * w + x] = (a * (1.0 - fy) + b * fy).round_ties_even() as u8;
        }
    }
    if config.denoise {
        resized = bilateral(&resized, w, h);
    }
    let mut input = vec![0_f32; w * h * 3];
    for (channel, (mean, std)) in [(0.485_f32, 0.229_f32), (0.456, 0.224), (0.406, 0.225)]
        .into_iter()
        .enumerate()
    {
        for (i, &v) in resized.iter().enumerate() {
            input[channel * w * h + i] = (v as f32 / 255.0 - mean) / std;
        }
    }
    Ok(input)
}

fn reflect101(mut i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let last = n as isize - 1;
    while i < 0 || i > last {
        i = if i < 0 { -i } else { 2 * last - i };
    }
    i as usize
}

fn bilateral(input: &[u8], w: usize, h: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let offsets: Vec<(isize, isize, f64)> = (-2_isize..=2)
        .flat_map(|dy| {
            (-2_isize..=2).filter_map(move |dx| {
                let d = dx * dx + dy * dy;
                (d <= 4).then(|| (dx, dy, (-(d as f64) / 200.0).exp()))
            })
        })
        .collect();
    input
        .par_iter()
        .enumerate()
        .map(|(i, &center)| {
            if center < 75 {
                return center;
            }
            let (x, y) = (i % w, i / w);
            let (mut sum, mut weights) = (0.0, 0.0);
            for &(dx, dy, spatial) in &offsets {
                let value =
                    input[reflect101(y as isize + dy, h) * w + reflect101(x as isize + dx, w)];
                let delta = value as f64 - center as f64;
                let weight = spatial * (-delta * delta / (2.0 * 75.0 * 75.0)).exp();
                sum += weight * value as f64;
                weights += weight;
            }
            (sum / weights).round_ties_even() as u8
        })
        .collect()
}

/// NCHW probabilities, channels: eyeball, iris, pupil, eyelashes.
pub fn postprocess(
    values: &[f32],
    shape: &[i64],
    width: usize,
    height: usize,
) -> Result<SegmentationMap> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 4 || shape[2] <= 0 || shape[3] <= 0 {
        return Err(Error::InvalidInput(
            "expected ONNX output [1,4,height,width]".into(),
        ));
    }
    let (sh, sw) = (shape[2] as usize, shape[3] as usize);
    let n = sw
        .checked_mul(sh)
        .filter(|&n| n <= MAX_PIXELS / 4)
        .ok_or_else(|| Error::InvalidInput("ONNX output dimensions exceed limit".into()))?;
    let out_n = width
        .checked_mul(height)
        .filter(|&n| n > 0 && n <= MAX_PIXELS)
        .ok_or_else(|| Error::InvalidInput("invalid output dimensions".into()))?;
    if width == 0
        || height == 0
        || values.len() != n * 4
        || values
            .iter()
            .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err(Error::InvalidInput(
            "invalid segmentation probabilities".into(),
        ));
    }
    let mut data = Vec::with_capacity(out_n);
    for y in 0..height {
        for x in 0..width {
            let i = (y * sh / height) * sw + x * sw / width;
            data.push([
                values[i],
                values[n + i],
                values[2 * n + i],
                values[3 * n + i],
            ]);
        }
    }
    Raster::new(width, height, data)
}

pub fn binarize(map: &SegmentationMap, thresholds: &[f32; 4]) -> Result<GeometryMasks> {
    map.validate()?;
    if thresholds
        .iter()
        .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        || map
            .data
            .iter()
            .flatten()
            .any(|&v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err(Error::InvalidInput(
            "invalid probability or threshold".into(),
        ));
    }
    let mask = |c: usize| {
        Raster::new(
            map.width,
            map.height,
            map.data.iter().map(|p| p[c] >= thresholds[c]).collect(),
        )
    };
    Ok(GeometryMasks {
        eyeball: mask(0)?,
        iris: mask(1)?,
        pupil: mask(2)?,
        noise: mask(3)?,
    })
}

#[cfg(feature = "onnx")]
pub struct OnnxSegmenter {
    session: ort::session::Session,
    config: SegmentationConfig,
}
#[cfg(feature = "onnx")]
impl OnnxSegmenter {
    pub fn new(path: impl AsRef<std::path::Path>, config: SegmentationConfig) -> Result<Self> {
        use ort::value::{TensorElementType, ValueType};
        config.validate()?;
        let session = ort::session::Session::builder()
            .map_err(onnx_error)?
            .with_intra_threads(config.threads)
            .map_err(onnx_error)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(onnx_error)?
            .commit_from_file(path)
            .map_err(onnx_error)?;
        if session.inputs().len() != 1 || session.outputs().is_empty() {
            return Err(Error::Onnx(
                "expected one image input and at least one output".into(),
            ));
        }
        let expected = [1, 3, config.input_height as i64, config.input_width as i64];
        match session.inputs()[0].dtype() {
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                ..
            } if shape.len() == 4
                && shape
                    .iter()
                    .zip(expected)
                    .all(|(&actual, expected)| actual < 0 || actual == expected) => {}
            _ => {
                return Err(Error::Onnx(
                    "model input must be float32 NCHW matching configured resolution".into(),
                ));
            }
        }
        Ok(Self { session, config })
    }
}
#[cfg(feature = "onnx")]
impl Segmenter for OnnxSegmenter {
    fn segment(&mut self, image: &GrayImage) -> Result<GeometryMasks> {
        let input = preprocess(image, &self.config)?;
        let tensor = ort::value::Tensor::from_array((
            [
                1_usize,
                3,
                self.config.input_height,
                self.config.input_width,
            ],
            input.into_boxed_slice(),
        ))
        .map_err(onnx_error)?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(onnx_error)?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(onnx_error)?;
        let map = postprocess(values, shape, image.width, image.height)?;
        binarize(&map, &self.config.thresholds)
    }
}
#[cfg(feature = "onnx")]
fn onnx_error<T>(error: ort::Error<T>) -> Error {
    Error::Onnx(error.to_string())
}
