//! Bernoulli distribution implementation.
//!
//! This module provides the Bernoulli distribution, which is a discrete probability
//! distribution that takes value 1 with probability p and value 0 with probability 1-p.
//! The density is computed with respect to counting measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Bernoulli;
//! use measures::LogDensityBuilder;
//!
//! let bernoulli = Bernoulli::new(0.7); // Success probability p = 0.7
//!
//! // Compute log-density at x = 1 (success)
//! let ld = bernoulli.log_density();
//! let log_density_value: f64 = ld.at(&1);
//! ```

use crate::core::types::{False, True};
use crate::core::utils::float_constant;
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::Float;

/// Bernoulli distribution Bernoulli(p).
///
/// This is a member of the exponential family with:
/// - Natural parameter: η = log(p/(1-p)) (log-odds)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = log(1 + exp(η))
/// - Base measure: Counting measure on {0, 1}
#[derive(Clone, Debug)]
pub struct Bernoulli<T> {
    pub prob: T, // p
}

impl<T: Float> Default for Bernoulli<T> {
    fn default() -> Self {
        Self {
            prob: float_constant::<T>(0.5),
        }
    }
}

impl<T: Float> MeasureMarker for Bernoulli<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Bernoulli<T> {
    /// Create a new Bernoulli distribution with given success probability.
    pub fn new(prob: T) -> Self {
        assert!(
            prob >= T::zero() && prob <= T::one(),
            "Probability must be in [0, 1]"
        );
        Self { prob }
    }

    /// Get the mean (which equals the probability p)
    pub fn mean(&self) -> T {
        self.prob
    }

    /// Get the variance p(1-p)
    pub fn variance(&self) -> T {
        self.prob * (T::one() - self.prob)
    }

    /// Get the log-odds log(p/(1-p))
    pub fn log_odds(&self) -> T {
        (self.prob / (T::one() - self.prob)).ln()
    }
}

impl<T: Float> Measure<u8> for Bernoulli<T> {
    type RootMeasure = CountingMeasure<u8>;

    fn in_support(&self, x: u8) -> bool {
        x == 0 || x == 1
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u8>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<u8, T> for Bernoulli<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = CountingMeasure<u8>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta] = param;
        // η = log(p/(1-p)) => p = exp(η)/(1 + exp(η)) = sigmoid(η)
        let exp_eta = eta.exp();
        let prob = exp_eta / (T::one() + exp_eta);
        Self::new(prob)
    }

    fn sufficient_statistic(&self, x: &u8) -> Self::SufficientStat {
        [float_constant::<T>(f64::from(*x))]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        CountingMeasure::<u8>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_param = [self.log_odds()];

        // Log partition: A(η) = log(1 + exp(η))
        let log_partition = (T::one() + natural_param[0].exp()).ln();

        (natural_param, log_partition)
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<u8, T> for Bernoulli<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        use crate::core::utils::safe_convert;

        let _prob_f64: f64 = safe_convert(self.prob);
        let _log_odds = (_prob_f64 / (1.0 - _prob_f64)).ln();
        let _log_partition = (1.0 + _log_odds.exp()).ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Bernoulli distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
