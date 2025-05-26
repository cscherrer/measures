//! Geometric distribution implementation.
//!
//! This module provides the Geometric distribution, which is a discrete probability
//! distribution that models the number of trials needed to get the first success
//! in a sequence of independent Bernoulli trials. The density is computed with
//! respect to counting measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Geometric;
//! use measures::LogDensityBuilder;
//!
//! let geometric = Geometric::new(0.3); // Success probability p = 0.3
//!
//! // Compute log-density at x = 2 (first success on trial 2)
//! let ld = geometric.log_density();
//! let log_density_value: f64 = ld.at(&2);
//! ```

use measures_core::primitive::counting::CountingMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
use num_traits::Float;

/// Geometric distribution Geometric(p).
///
/// This models the number of trials needed to get the first success.
/// This is a member of the exponential family with:
/// - Natural parameter: η = log(1-p)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = -log(-η/(1+η)) = -log(1-p) - log(p)
/// - Base measure: Counting measure on {1, 2, 3, ...}
#[derive(Clone, Debug)]
pub struct Geometric<T> {
    pub prob: T, // p (success probability)
}

impl<T: Float> Default for Geometric<T> {
    fn default() -> Self {
        Self {
            prob: T::from(0.5).unwrap(),
        }
    }
}

impl<T: Float> MeasureMarker for Geometric<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Geometric<T> {
    /// Create a new geometric distribution with given success probability.
    pub fn new(prob: T) -> Self {
        assert!(
            prob > T::zero() && prob <= T::one(),
            "Probability must be in (0, 1]"
        );
        Self { prob }
    }

    /// Get the mean 1/p
    pub fn mean(&self) -> T {
        self.prob.recip()
    }

    /// Get the variance (1-p)/p²
    pub fn variance(&self) -> T {
        let one_minus_p = T::one() - self.prob;
        one_minus_p / (self.prob * self.prob)
    }

    /// Get the failure probability (1-p)
    pub fn failure_prob(&self) -> T {
        T::one() - self.prob
    }
}

impl<T: Float> Measure<u64> for Geometric<T> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        x >= 1 // Geometric distribution starts from 1
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u64>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<u64, T> for Geometric<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = CountingMeasure<u64>;

    fn from_natural(param: <Self as ExponentialFamily<u64, T>>::NaturalParam) -> Self {
        let [eta] = param;
        // η = log(1-p) => 1-p = exp(η) => p = 1 - exp(η)
        let failure_prob = eta.exp();
        let prob = T::one() - failure_prob;
        Self::new(prob)
    }

    fn sufficient_statistic(&self, x: &u64) -> <Self as ExponentialFamily<u64, T>>::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<u64, T>>::BaseMeasure {
        CountingMeasure::<u64>::new()
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<u64, T>>::NaturalParam, T) {
        let natural_param = [self.failure_prob().ln()];

        // Log partition: A(η) = -log(1 - exp(η)) = -log(p)
        let log_partition = -self.prob.ln();

        (natural_param, log_partition)
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> measures_exponential_family::JITOptimizer<u64, T> for Geometric<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<measures_exponential_family::JITFunction, measures_exponential_family::JITError>
    {
        let _prob_f64 = self.prob.to_f64().unwrap();
        let _failure_prob = 1.0 - _prob_f64;
        let _log_failure_prob = _failure_prob.ln();
        let _log_prob = _prob_f64.ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            measures_exponential_family::JITError::UnsupportedExpression(
                "Geometric distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Implementation of HasLogDensity for Geometric distribution
impl<T: Float> measures_core::HasLogDensity<u64, T> for Geometric<T> {
    fn log_density_wrt_root(&self, x: &u64) -> T {
        // Geometric PMF: P(X = k) = (1-p)^(k-1) * p for k = 1, 2, 3, ...
        // log P(X = k) = (k-1)*log(1-p) + log(p)
        if *x >= 1 {
            let k = T::from(*x).unwrap();
            (k - T::one()) * (T::one() - self.prob).ln() + self.prob.ln()
        } else {
            T::neg_infinity() // Outside support (k must be >= 1)
        }
    }
}
