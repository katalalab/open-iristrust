use crate::{Error, Result};
use serde::{Deserialize, Serialize};

pub const MAX_PIXELS: usize = 16_777_216;
pub const TEMPLATE_VERSION: &str = "iristrust-v1";

/// Row-major raster. Validate deserialized values at API boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Raster<T> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}
impl<T> Raster<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Result<Self> {
        let raster = Self {
            width,
            height,
            data,
        };
        raster.validate()?;
        Ok(raster)
    }
    pub fn validate(&self) -> Result<()> {
        let len = self
            .width
            .checked_mul(self.height)
            .filter(|&n| n > 0 && n <= MAX_PIXELS)
            .ok_or_else(|| {
                Error::InvalidInput("raster dimensions are zero, overflowed, or too large".into())
            })?;
        if self.data.len() != len {
            return Err(Error::InvalidInput(
                "raster data does not match dimensions".into(),
            ));
        }
        Ok(())
    }
    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x < self.width && y < self.height {
            self.data.get(y * self.width + x)
        } else {
            None
        }
    }
}
pub type GrayImage = Raster<u8>;
pub type Mask = Raster<bool>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum EyeSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
impl Point {
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMasks {
    pub pupil: Mask,
    pub iris: Mask,
    pub eyeball: Mask,
    pub noise: Mask,
}
impl GeometryMasks {
    pub fn validate(&self) -> Result<()> {
        for mask in [&self.pupil, &self.iris, &self.eyeball, &self.noise] {
            mask.validate()?;
            if (mask.width, mask.height) != (self.pupil.width, self.pupil.height) {
                return Err(Error::InvalidInput(
                    "segmentation mask dimensions differ".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Equal angular samples starting at positive x around each center, in image coordinates.
/// The outer boundary must be convex and contain the entire pupil.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry {
    pub pupil: Vec<Point>,
    pub iris: Vec<Point>,
    pub eyeball: Vec<Point>,
    pub pupil_center: Point,
    pub iris_center: Point,
    pub orientation_radians: f64,
}
impl Geometry {
    pub fn validate(&self) -> Result<()> {
        if self.pupil.len() < 8
            || self.pupil.len() > 4096
            || self.pupil.len() != self.iris.len()
            || !self.pupil_center.is_finite()
            || !self.iris_center.is_finite()
            || !self.orientation_radians.is_finite()
            || self
                .pupil
                .iter()
                .chain(&self.iris)
                .chain(&self.eyeball)
                .any(|p| !p.is_finite())
        {
            return Err(Error::InvalidInput("invalid boundary geometry".into()));
        }
        let n = self.iris.len();
        for (points, center) in [
            (&self.pupil, self.pupil_center),
            (&self.iris, self.iris_center),
        ] {
            for (i, p) in points.iter().enumerate() {
                let (dx, dy) = (p.x - center.x, p.y - center.y);
                let radius = dx.hypot(dy);
                let angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
                let expected = std::f64::consts::TAU * i as f64 / n as f64;
                let delta = (angle - expected + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                if !radius.is_finite() || radius <= 1e-9 || radius > 1e6 || delta.abs() > 1e-6 {
                    return Err(Error::InvalidInput(
                        "boundaries must use equal angular samples starting at positive x".into(),
                    ));
                }
            }
        }
        for i in 0..n {
            let (a, b, c) = (self.iris[i], self.iris[(i + 1) % n], self.iris[(i + 2) % n]);
            if (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0.0 {
                return Err(Error::InvalidInput(
                    "iris boundary must be strictly convex".into(),
                ));
            }
        }
        for p in self.pupil.iter().chain(std::iter::once(&self.pupil_center)) {
            let (dx, dy) = (p.x - self.iris_center.x, p.y - self.iris_center.y);
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
            let sector = ((angle / std::f64::consts::TAU * n as f64).floor() as usize).min(n - 1);
            let (a, b) = (self.iris[sector], self.iris[(sector + 1) % n]);
            if (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) <= 1e-9 {
                return Err(Error::InvalidInput(
                    "pupil boundary must lie strictly inside iris".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedIris {
    pub image: GrayImage,
    pub mask: Mask,
}
impl NormalizedIris {
    pub fn validate(&self) -> Result<()> {
        self.image.validate()?;
        self.mask.validate()?;
        if (self.image.width, self.image.height) != (self.mask.width, self.mask.height) {
            return Err(Error::InvalidInput(
                "normalized image/mask dimensions differ".into(),
            ));
        }
        Ok(())
    }
}

/// Two phase bits per sample; little-endian u64 words, bit index (row*cols+col)*2+component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodePlane {
    pub rows: usize,
    pub cols: usize,
    pub code: Vec<u64>,
    pub mask: Vec<u64>,
}
impl CodePlane {
    pub fn bit_len(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.cols)
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| self.rows > 0 && self.cols > 0 && n <= MAX_PIXELS)
            .ok_or_else(|| Error::InvalidInput("invalid code plane dimensions".into()))
    }
    pub fn validate(&self) -> Result<()> {
        let len = self.bit_len()?;
        let words = len.div_ceil(64);
        if self.code.len() != words || self.mask.len() != words {
            return Err(Error::InvalidInput(
                "packed code/mask length mismatch".into(),
            ));
        }
        let tail = len % 64;
        if tail != 0 && ((self.code[words - 1] | self.mask[words - 1]) >> tail) != 0 {
            return Err(Error::InvalidInput("nonzero padding bits".into()));
        }
        Ok(())
    }
    /// Checked phase access; component 0 is real, 1 imaginary.
    pub fn get_bit(&self, row: usize, col: usize, component: usize, mask: bool) -> Result<bool> {
        self.validate()?;
        if row >= self.rows || col >= self.cols || component >= 2 {
            return Err(Error::InvalidInput(
                "bit coordinates outside code plane".into(),
            ));
        }
        Ok(self.bit(row, col, component, mask))
    }
    pub fn put_bit(
        &mut self,
        row: usize,
        col: usize,
        component: usize,
        mask: bool,
        value: bool,
    ) -> Result<()> {
        self.validate()?;
        if row >= self.rows || col >= self.cols || component >= 2 {
            return Err(Error::InvalidInput(
                "bit coordinates outside code plane".into(),
            ));
        }
        self.set_bit(row, col, component, mask, value);
        Ok(())
    }
    pub(crate) fn bit(&self, row: usize, col: usize, component: usize, mask: bool) -> bool {
        let i = (row * self.cols + col) * 2 + component;
        let words = if mask { &self.mask } else { &self.code };
        (words[i / 64] >> (i % 64)) & 1 != 0
    }
    pub(crate) fn set_bit(
        &mut self,
        row: usize,
        col: usize,
        component: usize,
        mask: bool,
        value: bool,
    ) {
        let i = (row * self.cols + col) * 2 + component;
        let words = if mask { &mut self.mask } else { &mut self.code };
        let b = 1_u64 << (i % 64);
        if value {
            words[i / 64] |= b;
        } else {
            words[i / 64] &= !b;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrisTemplate {
    pub version: String,
    pub encoding_profile: String,
    pub eye_side: EyeSide,
    pub planes: Vec<CodePlane>,
}
impl IrisTemplate {
    pub fn validate(&self) -> Result<()> {
        if self.version != TEMPLATE_VERSION
            || self.planes.is_empty()
            || self.planes.len() > 16
            || self.encoding_profile.is_empty()
            || self.encoding_profile.len() > 128
            || !self
                .encoding_profile
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_:".contains(&c))
        {
            return Err(Error::InvalidInput(
                "unsupported version or plane count".into(),
            ));
        }
        for plane in &self.planes {
            plane.validate()?;
        }
        Ok(())
    }
    pub fn valid_bits(&self) -> usize {
        self.planes
            .iter()
            .flat_map(|p| &p.mask)
            .map(|w| w.count_ones() as usize)
            .sum()
    }
}
