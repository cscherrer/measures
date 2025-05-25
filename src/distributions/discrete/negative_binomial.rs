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

use crate::core::density::HasLogDensity;
use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::counting::CountingMeasure;
use crate::traits::DotProduct;
use num_traits::{Float, ToPrimitive};
use special::Gamma as GammaTrait;

/// Negative Binomial distribution NB(r, p) with fixed r.
///
/// This models the number of successes before r failures occur.
/// This is a member of the exponential family when r is fixed with:
/// - Natural parameter: η = log(p/(1-p)) (log-odds)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = -r * log(1 - p) = -r * log(1 - sigmoid(η))
/// - Base measure: Counting measure with negative binomial coefficient
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

    /// Compute log negative binomial coefficient log(C(x+r-1, x))
    fn log_neg_binomial_coeff(&self, x: u64) -> T {
        // C(x+r-1, x) = C(x+r-1, r-1) = Γ(x+r) / (Γ(x+1) * Γ(r))
        if let (Some(x_f64), Some(r_f64)) = (x.to_f64(), self.r.to_f64()) {
            // Compute negative binomial coefficient using log-gamma function
            let log_coeff = {
                let (ln_gamma_x_plus_r, _) = (x_f64 + r_f64).ln_gamma();
                let (ln_gamma_x_plus_1, _) = (x_f64 + 1.0).ln_gamma();
                let (ln_gamma_r, _) = r_f64.ln_gamma();
                ln_gamma_x_plus_r - ln_gamma_x_plus_1 - ln_gamma_r
            };
            T::from(log_coeff).unwrap()
        } else {
            // Fallback for types that don't convert to f64
            T::zero()
        }
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
    type BaseMeasure = CountingMeasure<u64>;

    fn from_natural(_param: Self::NaturalParam) -> Self {
        // This requires knowing r, which we can't determine from just η
        // In practice, you'd need to specify r separately
        panic!("Cannot construct NegativeBinomial from natural parameter without knowing r");
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        CountingMeasure::<u64>::new()
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

    // Override the exponential family log-density to include negative binomial coefficient
    fn exp_fam_log_density(&self, x: &u64) -> T
    where
        Self::NaturalParam: crate::traits::DotProduct<Self::SufficientStat, Output = T>,
        Self::BaseMeasure: crate::core::HasLogDensity<u64, T>,
    {
        let (natural_params, log_partition) = self.natural_and_log_partition();
        let sufficient_stats = self.sufficient_statistic(x);

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

        // Chain rule part: log-density of base measure with respect to root measure
        let chain_rule_part = self.base_measure().log_density_wrt_root(x);

        // Negative binomial coefficient part: log C(x+r-1, x)
        let neg_binomial_coeff_part = self.log_neg_binomial_coeff(*x);

        // Complete log-density: exponential family + chain rule + negative binomial coefficient
        exp_fam_part + chain_rule_part + neg_binomial_coeff_part
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
