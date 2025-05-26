//! Exponential distribution implementation.
//!
//! This module provides the Exponential distribution, which is a continuous probability
//! distribution characterized by its rate parameter λ. The density is computed with
//! respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Exponential;
//! use measures::LogDensityBuilder;
//!
//! let exp_dist = Exponential::new(1.0); // Rate parameter λ = 1.0
//!
//! // Compute log-density at x = 2.0
//! let ld = exp_dist.log_density();
//! let log_density_value: f64 = ld.at(&2.0);
//! ```

use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use num_traits::Float;

/// Exponential distribution Exp(λ)
///
/// The exponential distribution is a continuous probability distribution that models
/// the time between events in a Poisson process. It's parameterized by rate λ.
#[derive(Debug, Clone, PartialEq)]
pub struct Exponential<T> {
    /// Rate parameter λ (must be positive)
    pub rate: T,
}

impl<T: Float> Default for Exponential<T> {
    fn default() -> Self {
        Self { rate: T::one() }
    }
}

impl<T: Float> MeasureMarker for Exponential<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Exponential<T> {
    /// Create a new exponential distribution with given rate parameter λ.
    pub fn new(rate: T) -> Self {
        assert!(rate > T::zero(), "Rate parameter must be positive");
        Self { rate }
    }

    /// Get the mean 1/λ
    pub fn mean(&self) -> T {
        self.rate.recip()
    }

    /// Get the variance 1/λ²
    pub fn variance(&self) -> T {
        let inv_rate = self.rate.recip();
        inv_rate * inv_rate
    }
}

impl<T: Float> Measure<T> for Exponential<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        x >= T::zero()
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Exponential<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta] = param;
        let rate = -eta;
        Self::new(rate)
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [*x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_param = [-self.rate];
        let log_partition = self.rate.ln();
        (natural_param, log_partition)
    }
}

// Implementation of HasLogDensity for Exponential distribution
impl<T: Float> measures_core::HasLogDensity<T, T> for Exponential<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        if *x >= T::zero() {
            // Exponential PDF: f(x|λ) = λ * exp(-λx)
            // log f(x|λ) = log(λ) - λx
            self.rate.ln() - self.rate * *x
        } else {
            // Outside support, return negative infinity
            T::neg_infinity()
        }
    }
}

#[cfg(feature = "jit")]
use measures_core::safe_convert;

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<T, T> for Exponential<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _rate_f64: f64 = safe_convert(self.rate);
        let _log_rate = _rate_f64.ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Exponential distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Automatic JIT compilation using the auto-derivation system
#[cfg(feature = "jit")]
crate::auto_jit_impl!(Exponential<f64>);
