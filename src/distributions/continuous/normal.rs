//! Normal (Gaussian) distribution implementation.
//!
//! This module provides the Normal distribution, which is a continuous probability
//! distribution characterized by its mean and standard deviation. The density is
//! computed with respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::{Normal, LogDensityBuilder};
//!
//! let normal = Normal::new(0.0, 1.0); // Standard normal distribution
//!
//! // Compute log-density at x = 0
//! let ld = normal.log_density();
//! let log_density_value: f64 = ld.at(&0.0);
//! ```

use crate::core::types::{False, True};
use crate::core::utils::float_constant;
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

/// Normal distribution N(μ, σ²).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [μ/σ², -1/(2σ²)]
/// - Sufficient statistics: T(x) = [x, x²]
/// - Log partition: A(η) = -η₁²/(4η₂) - ½log(-2η₂) - ½log(2π)
/// - Base measure: Lebesgue measure (dx)
#[derive(Clone, Debug)]
pub struct Normal<T> {
    pub mean: T,
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

impl<T: Float + FloatConst> Normal<T> {
    /// Create a new normal distribution with given mean and standard deviation.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }

    /// Get the variance σ²
    pub fn variance(&self) -> T {
        self.std_dev * self.std_dev
    }
}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Normal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -(float_constant::<T>(2.0) * eta2).recip();
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let sigma2 = self.variance();
        let mu2 = self.mean * self.mean;
        let inv_sigma2 = sigma2.recip();

        let natural_params = [
            self.mean * inv_sigma2,
            float_constant::<T>(-0.5) * inv_sigma2,
        ];

        let log_partition = (float_constant::<T>(2.0) * T::PI() * sigma2).ln()
            * float_constant::<T>(0.5)
            + float_constant::<T>(0.5) * mu2 * inv_sigma2;

        (natural_params, log_partition)
    }
}

// Automatic JIT compilation using the auto-derivation system
#[cfg(feature = "jit")]
crate::auto_jit_impl!(Normal<f64>);
