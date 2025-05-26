//! Negative Binomial distribution implementation.
//!
//! This module provides the Negative Binomial distribution, which is a discrete probability
//! distribution that models the number of successes in a sequence of independent Bernoulli
//! trials before a specified number of failures occurs. The density is computed with
//! respect to counting measure.
//!
//! Note: This is only an exponential family when r (number of failures) is fixed and known.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::NegativeBinomial;
//! use measures::LogDensityBuilder;
//!
//! let neg_binom = NegativeBinomial::new(5, 0.6); // r = 5 failures, success probability p = 0.6
//!
//! // Compute log-density at x = 3 (3 successes before 5 failures)
//! let ld = neg_binom.log_density();
//! let log_density_value: f64 = ld.at(&3);
//! ```

use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::derived::negative_binomial_coefficient::NegativeBinomialCoefficientMeasure;
use crate::measures::primitive::counting::CountingMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use num_traits::Float;

/// Negative Binomial distribution NB(r, p) with fixed r.
///
/// This models the number of successes before r failures occur.
/// This is a member of the exponential family when r is fixed with:
/// - Natural parameter: η = log(p/(1-p)) (log-odds)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = -r * log(1 - p) = -r * log(1 - sigmoid(η))
/// - Base measure: Negative binomial coefficient measure (negative binomial coefficient term)
#[derive(Clone, Debug)]
pub struct NegativeBinomial<T> {
    pub r: u64,  // Number of failures (fixed)
    pub prob: T, // Success probability p
}

impl<T: Float> MeasureMarker for NegativeBinomial<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> NegativeBinomial<T> {
    /// Create a new negative binomial distribution with given number of failures and success probability.
    pub fn new(r: u64, prob: T) -> Self {
        assert!(r > 0, "Number of failures must be positive");
        assert!(
            prob > T::zero() && prob < T::one(),
            "Probability must be in (0, 1)"
        );
        Self { r, prob }
    }

    /// Get the mean r*p/(1-p)
    pub fn mean(&self) -> T {
        let r_t = T::from(self.r).unwrap();
        r_t * self.prob / (T::one() - self.prob)
    }

    /// Get the variance r*p/(1-p)²
    pub fn variance(&self) -> T {
        let r_t = T::from(self.r).unwrap();
        let one_minus_p = T::one() - self.prob;
        r_t * self.prob / (one_minus_p * one_minus_p)
    }

    /// Get the log-odds log(p/(1-p))
    pub fn log_odds(&self) -> T {
        (self.prob / (T::one() - self.prob)).ln()
    }

    /// Get the failure probability (1-p)
    pub fn failure_prob(&self) -> T {
        T::one() - self.prob
    }
}

impl<T: Float> Measure<u64> for NegativeBinomial<T> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, _x: u64) -> bool {
        true // Support is {0, 1, 2, ...}
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u64>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<u64, T> for NegativeBinomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = NegativeBinomialCoefficientMeasure<T>;

    fn from_natural(_param: Self::NaturalParam) -> Self {
        // This requires knowing r, which we can't determine from just η
        // In practice, you'd need to specify r separately
        panic!("Cannot construct NegativeBinomial from natural parameter without knowing r");
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        NegativeBinomialCoefficientMeasure::<T>::new(self.r)
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_param = [self.log_odds()];

        // Log partition: A(η) = -r * log(1 - sigmoid(η))
        // where sigmoid(η) = exp(η)/(1 + exp(η))
        let exp_eta = natural_param[0].exp();
        let sigmoid_eta = exp_eta / (T::one() + exp_eta);
        let r_t = T::from(self.r).unwrap();
        let log_partition = -r_t * (T::one() - sigmoid_eta).ln();

        (natural_param, log_partition)
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<u64, T> for NegativeBinomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _prob_f64 = self.prob.to_f64().unwrap();
        let _r_f64 = self.r as f64;
        let _log_odds = (_prob_f64 / (1.0 - _prob_f64)).ln();
        let _log_failure_prob = (1.0 - _prob_f64).ln();
        let _r_log_failure_prob = _r_f64 * _log_failure_prob;

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "NegativeBinomial distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Implementation of HasLogDensity for NegativeBinomial distribution
impl<T: Float> measures_core::HasLogDensity<u64, T> for NegativeBinomial<T> {
    fn log_density_wrt_root(&self, x: &u64) -> T {
        // Negative Binomial PMF: P(X = k) = C(k+r-1, k) * p^r * (1-p)^k
        // log P(X = k) = log(C(k+r-1, k)) + r*log(p) + k*log(1-p)
        let k = T::from(*x).unwrap();
        let r = T::from(self.r).unwrap();

        // The log binomial coefficient is handled by the base measure
        r * self.prob.ln() + k * (T::one() - self.prob).ln()
    }
}
