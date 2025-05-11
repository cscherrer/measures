//! Normal (Gaussian) distribution implementation.
//!
//! This module provides the Normal distribution, which is a continuous probability
//! distribution characterized by its mean and standard deviation. The density is
//! computed with respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::{Normal, LebesgueMeasure, HasDensity};
//!
//! let normal = Normal::new(0.0, 1.0); // Standard normal distribution
//!
//! // Compute density at x = 0
//! let density: f64 = normal.density(&0.0).into();
//!
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(&0.0).into();
//! ```

use crate::measures::lebesgue::LebesgueMeasure;
use crate::traits::{Density, LogDensity, Measure, PrimitiveMeasure};
use num_traits::Float;

/// A normal (Gaussian) distribution.
///
/// The normal distribution is characterized by its mean and standard deviation.
/// The density is computed with respect to Lebesgue measure.
#[derive(Clone)]
pub struct Normal<T: Float> {
    /// The mean of the distribution
    pub mean: T,
    /// The standard deviation of the distribution
    pub std_dev: T,
}

impl<T: Float> Normal<T> {
    /// Create a new normal distribution with the given mean and standard deviation.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the distribution
    /// * `std_dev` - The standard deviation of the distribution (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `std_dev` is not positive.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }
}

const SQRT_2_PI: f64 = 2.506_628_274_631_000_7;
const LOG_SQRT_2_PI: f64 = 0.918_938_533_204_672_8;

impl<T: Float> PrimitiveMeasure<T> for Normal<T> {}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

impl<T: Float> From<Density<'_, T, Normal<T>>> for f64 {
    fn from(val: Density<'_, T, Normal<T>>) -> Self {
        let x = val.x.to_f64().unwrap();
        let mean = val.measure.mean.to_f64().unwrap();
        let std_dev = val.measure.std_dev.to_f64().unwrap();

        let z = (x - mean) / std_dev;
        (1.0 / (std_dev * SQRT_2_PI)) * (-0.5 * z * z).exp()
    }
}

impl<T: Float> From<Density<'_, T, Normal<T>, LebesgueMeasure<T>>> for f64 {
    fn from(val: Density<'_, T, Normal<T>, LebesgueMeasure<T>>) -> Self {
        let x = val.x.to_f64().unwrap();
        let mean = val.measure.mean.to_f64().unwrap();
        let std_dev = val.measure.std_dev.to_f64().unwrap();

        let z = (x - mean) / std_dev;
        (1.0 / (std_dev * SQRT_2_PI)) * (-0.5 * z * z).exp()
    }
}

impl<T: Float> From<LogDensity<'_, T, Normal<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Normal<T>>) -> Self {
        let x = val.x.to_f64().unwrap();
        let mean = val.measure.mean.to_f64().unwrap();
        let std_dev = val.measure.std_dev.to_f64().unwrap();

        let z = (x - mean) / std_dev;
        -0.5 * z * z - LOG_SQRT_2_PI - std_dev.ln()
    }
}

impl<T: Float> From<LogDensity<'_, T, Normal<T>, LebesgueMeasure<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Normal<T>, LebesgueMeasure<T>>) -> Self {
        let x = val.x.to_f64().unwrap();
        let mean = val.measure.mean.to_f64().unwrap();
        let std_dev = val.measure.std_dev.to_f64().unwrap();

        let z = (x - mean) / std_dev;
        -0.5 * z * z - LOG_SQRT_2_PI - std_dev.ln()
    }
}
