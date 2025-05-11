//! Normal (Gaussian) distribution implementation.
//! 
//! This module provides the Normal distribution, which is a continuous probability
//! distribution characterized by its mean and standard deviation. The density is
//! computed with respect to Lebesgue measure.
//! 
//! # Example
//! 
//! ```rust
//! use measures::{Normal, LebesgueMeasure};
//! 
//! let normal = Normal::new(0.0, 1.0); // Standard normal distribution
//! let lebesgue = LebesgueMeasure::new();
//! 
//! // Compute density at x = 0
//! let density: f64 = normal.density(0.0).wrt(&lebesgue).into();
//! 
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(0.0).wrt(&lebesgue).into();
//! ```

use crate::traits::{PrimitiveMeasure, Measure, Density, DensityWRT, LogDensity, LogDensityWRT, HasDensity};
use crate::measures::lebesgue::LebesgueMeasure;
use num_traits::Float;
use std::f64::consts::PI;

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

impl<'a, T: Float> Into<f64> for Density<'a, T, Normal<T>> {
    fn into(self) -> f64 {
        let x = self.x.to_f64().unwrap();
        let mean = self.measure.mean.to_f64().unwrap();
        let std_dev = self.measure.std_dev.to_f64().unwrap();
        
        let z = (x - mean) / std_dev;
        (1.0 / (std_dev * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
    }
}

impl<'a, T: Float> Into<f64> for DensityWRT<'a, T, Normal<T>, LebesgueMeasure<T>> {
    fn into(self) -> f64 {
        let x = self.x.to_f64().unwrap();
        let mean = self.measure.mean.to_f64().unwrap();
        let std_dev = self.measure.std_dev.to_f64().unwrap();
        
        let z = (x - mean) / std_dev;
        (1.0 / (std_dev * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
    }
}

impl<'a, T: Float> Into<f64> for LogDensity<'a, T, Normal<T>> {
    fn into(self) -> f64 {
        let x = self.x.to_f64().unwrap();
        let mean = self.measure.mean.to_f64().unwrap();
        let std_dev = self.measure.std_dev.to_f64().unwrap();
        
        let z = (x - mean) / std_dev;
        -0.5 * z * z - (std_dev * (2.0 * PI).sqrt()).ln()
    }
}

impl<'a, T: Float> Into<f64> for LogDensityWRT<'a, T, Normal<T>, LebesgueMeasure<T>> {
    fn into(self) -> f64 {
        let x = self.x.to_f64().unwrap();
        let mean = self.measure.mean.to_f64().unwrap();
        let std_dev = self.measure.std_dev.to_f64().unwrap();
        
        let z = (x - mean) / std_dev;
        -0.5 * z * z - (std_dev * (2.0 * PI).sqrt()).ln()
    }
} 