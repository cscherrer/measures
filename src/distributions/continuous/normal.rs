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

use crate::core::{Density, False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{DotProduct, ExpFamDensity, ExponentialFamily};
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

impl<T: Float + FloatConst> ExpFamDensity<T> for Normal<T> {}

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
        self.log_density_ef(x)
    }
}

// Implement DotProduct for natural parameters
impl<T: Float> DotProduct<(T, T), T> for (T, T) {
    fn dot(lhs: &Self, rhs: &(T, T)) -> T {
        lhs.0 * rhs.0 + lhs.1 * rhs.1
    }
}

impl<T: Float + FloatConst> ExponentialFamily<T> for Normal<T> {
    type NaturalParam = (T, T); // (η₁, η₂) = (μ/σ², -1/(2σ²))
    type SufficientStat = (T, T); // (x, x²)

    fn from_natural(param: <Self as ExponentialFamily<T>>::NaturalParam) -> Self {
        let (eta1, eta2) = param;
        let sigma2 = -T::one() / (T::from(2.0).unwrap() * eta2);
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn to_natural(&self) -> <Self as ExponentialFamily<T>>::NaturalParam {
        let sigma2 = self.std_dev * self.std_dev;
        (
            self.mean / sigma2,
            -T::one() / (T::from(2.0).unwrap() * sigma2),
        )
    }

    fn log_partition(&self) -> T {
        let sigma2 = self.std_dev * self.std_dev;
        let mu2 = self.mean * self.mean;
        (T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap()
            + mu2 / (T::from(2.0).unwrap() * sigma2)
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T>>::SufficientStat {
        (*x, *x * *x)
    }

    fn carrier_measure(&self, _x: &T) -> T {
        T::one()
    }
}

// Implement From for Density to f64
impl<T: Float + FloatConst> From<Density<'_, T, Normal<T>>> for f64 {
    fn from(val: Density<'_, T, Normal<T>>) -> Self {
        let log_density: f64 = LogDensity::new(val.measure, val.x).into();
        log_density.exp()
    }
}

impl<T: Float + FloatConst> From<Density<'_, T, Normal<T>, LebesgueMeasure<T>>> for f64 {
    fn from(val: Density<'_, T, Normal<T>, LebesgueMeasure<T>>) -> Self {
        let log_density: f64 = LogDensity::new(val.measure, val.x).into();
        log_density.exp()
    }
}

// Implement From for LogDensity to f64
impl<T: Float + FloatConst> From<LogDensity<'_, T, Normal<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Normal<T>>) -> Self {
        let x = *val.x;
        let mu = val.measure.mean;
        let sigma = val.measure.std_dev;
        let sigma2 = sigma * sigma;

        let diff = x - mu;
        let norm_constant =
            -(T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap();
        let exponent = -(diff * diff) / (T::from(2.0).unwrap() * sigma2);

        (norm_constant + exponent).to_f64().unwrap()
    }
}

impl<T: Float + FloatConst> From<LogDensity<'_, T, Normal<T>, LebesgueMeasure<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Normal<T>, LebesgueMeasure<T>>) -> Self {
        // For Normal distribution, the log-density with respect to Lebesgue measure
        // is the same as the log-density with respect to itself
        From::<LogDensity<'_, T, Normal<T>>>::from(LogDensity::new(val.measure, val.x))
    }
}
