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
//! let density: f64 = normal.log_density(&0.0).into();
//!
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(&0.0).into();
//! ```

use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{
     ExponentialFamily, ExponentialFamilyMeasure, compute_normal_log_density,
};
use crate::measures::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

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

impl<T: Float> Default for Normal<T> {
    fn default() -> Self {
        Self {
            mean: T::zero(),
            std_dev: T::one(),
        }
    }
}

impl<T: Float> MeasureMarker for Normal<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}


impl<T: Float + FloatConst> ExponentialFamilyMeasure<T, T> for Normal<T> {}

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
    #[must_use]
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }
}

impl<T: Float + FloatConst> Normal<T> {
    /// Compute the log density directly
    pub fn compute_log_density(&self, x: &T) -> T {
        compute_normal_log_density(self.mean, self.std_dev, *x)
    }
}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> <Self as Measure<T>>::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Implement HasDensity using the ExponentialFamily structure
impl<T: Float + FloatConst> HasDensity<T> for Normal<T> {
    fn log_density<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        // Use the exponential family form directly
        crate::core::measure::HasDensity::log_density_ef(self, x)
    }
}

// Implement From for LogDensity to f64 - use a single implementation
// that works with any base measure
impl<T: Float + FloatConst, M: Measure<T>> From<LogDensity<'_, T, Normal<T>, M>> for f64 {
    fn from(val: LogDensity<'_, T, Normal<T>, M>) -> Self {
        let normal = val.measure;
        let x = val.x;

        normal.compute_log_density(x).to_f64().unwrap()
    }
}

impl<T: Float + FloatConst> ExponentialFamily<T, T> for Normal<T> {
    type NaturalParam = [T; 2]; // (η₁, η₂) = (μ/σ², -1/(2σ²))
    type SufficientStat = [T; 2]; // (x, x²)
    type BaseMeasure = LebesgueMeasure<T>;
    
    fn from_natural(param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -T::one() / (T::from(2.0).unwrap() * eta2);
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn to_natural(&self) -> <Self as ExponentialFamily<T, T>>::NaturalParam {
        let sigma2 = self.std_dev * self.std_dev;
        [
            self.mean / sigma2,
            -T::one() / (T::from(2.0).unwrap() * sigma2),
        ]
    }

    fn log_partition(&self) -> T {
        let sigma2 = self.std_dev * self.std_dev;
        let mu2 = self.mean * self.mean;
        (T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap()
            + mu2 / (T::from(2.0).unwrap() * sigma2)
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::HasDensity;
    use approx::assert_relative_eq;

    #[test]
    fn test_density_calculation_methods() {
        // Create a normal distribution
        let normal = Normal::new(1.5, 2.0);
        let x = 3.0;

        // Method 1: Using the exponential family form via log_density
        let log_density = normal.log_density(&x);
        let log_density_ef: f64 = log_density.into();

        // Method 2: Direct computation through From implementation for LogDensity
        let log_density_from = f64::from(LogDensity::new(&normal, &x));

        // Method 3: Manual calculation using the standard formula
        let x_f64 = x;
        let mu = normal.mean;
        let sigma = normal.std_dev;
        let sigma2 = sigma * sigma;
        let norm_constant = 1.0 / (2.0 * std::f64::consts::PI * sigma2).sqrt();
        let exponent = -((x_f64 - mu) * (x_f64 - mu)) / (2.0 * sigma2);
        let log_density_direct = exponent + norm_constant.ln();

        // Compare the results
        println!("Log-density via exponential family: {log_density_ef}");
        println!("Log-density from From impl: {log_density_from}");
        println!("Log-density via direct calculation: {log_density_direct}");

        assert_relative_eq!(log_density_ef, log_density_from, epsilon = 1e-10);
        assert_relative_eq!(log_density_from, log_density_direct, epsilon = 1e-10);
    }
}
