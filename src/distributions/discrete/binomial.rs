//! Binomial distribution implementation.
//!
//! This module provides the Binomial distribution, which is a discrete probability
//! distribution that models the number of successes in n independent Bernoulli trials.
//! The density is computed with respect to counting measure.
//!
//! Note: This is only an exponential family when n is fixed and known.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Binomial;
//! use measures::LogDensityBuilder;
//!
//! let binomial = Binomial::new(10, 0.3); // n = 10 trials, success probability p = 0.3
//!
//! // Compute log-density at x = 3 (3 successes out of 10 trials)
//! let ld = binomial.log_density();
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

/// Binomial distribution Binomial(n, p) with fixed n.
///
/// This is a member of the exponential family when n is fixed with:
/// - Natural parameter: η = log(p/(1-p)) (log-odds)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = n * log(1 + exp(η))
/// - Base measure: Counting measure with binomial coefficient
#[derive(Clone, Debug)]
pub struct Binomial<T> {
    pub n: u64,  // Number of trials (fixed)
    pub prob: T, // Success probability p
}

impl<T: Float> MeasureMarker for Binomial<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Binomial<T> {
    /// Create a new binomial distribution with given number of trials and success probability.
    pub fn new(n: u64, prob: T) -> Self {
        assert!(
            prob >= T::zero() && prob <= T::one(),
            "Probability must be in [0, 1]"
        );
        Self { n, prob }
    }

    /// Get the mean n*p
    pub fn mean(&self) -> T {
        T::from(self.n).unwrap() * self.prob
    }

    /// Get the variance n*p*(1-p)
    pub fn variance(&self) -> T {
        let n_t = T::from(self.n).unwrap();
        n_t * self.prob * (T::one() - self.prob)
    }

    /// Get the log-odds log(p/(1-p))
    pub fn log_odds(&self) -> T {
        (self.prob / (T::one() - self.prob)).ln()
    }

    /// Compute log binomial coefficient log(C(n, k))
    fn log_binomial_coeff(&self, k: u64) -> T {
        if k > self.n {
            return T::neg_infinity();
        }

        // Use the more stable log-gamma approach for large values
        if let (Some(n_f64), Some(k_f64)) = (self.n.to_f64(), k.to_f64()) {
            let log_coeff = {
                let (ln_gamma_n_plus_1, _) = (n_f64 + 1.0).ln_gamma();
                let (ln_gamma_k_plus_1, _) = (k_f64 + 1.0).ln_gamma();
                let (ln_gamma_n_minus_k_plus_1, _) = (n_f64 - k_f64 + 1.0).ln_gamma();
                ln_gamma_n_plus_1 - ln_gamma_k_plus_1 - ln_gamma_n_minus_k_plus_1
            };
            T::from(log_coeff).unwrap()
        } else {
            // Fallback for types that don't convert to f64
            T::zero()
        }
    }
}

impl<T: Float> Measure<u64> for Binomial<T> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        x <= self.n
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u64>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<u64, T> for Binomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = CountingMeasure<u64>;

    fn from_natural(_param: Self::NaturalParam) -> Self {
        // This requires knowing n, which we can't determine from just η
        // In practice, you'd need to specify n separately
        panic!("Cannot construct Binomial from natural parameter without knowing n");
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        CountingMeasure::<u64>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_param = [self.log_odds()];

        // Log partition: A(η) = n * log(1 + exp(η))
        let n_t = T::from(self.n).unwrap();
        let log_partition = n_t * (T::one() + natural_param[0].exp()).ln();

        (natural_param, log_partition)
    }

    // Override the exponential family log-density to include binomial coefficient
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

        // Binomial coefficient part: log C(n, k)
        let binomial_coeff_part = self.log_binomial_coeff(*x);

        // Complete log-density: exponential family + chain rule + binomial coefficient
        exp_fam_part + chain_rule_part + binomial_coeff_part
    }
}

// Symbolic optimization implementation
#[cfg(feature = "symbolic")]
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<u64, T> for Binomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: log C(n,x) + x * log(p/(1-p)) - n * log(1 + exp(log(p/(1-p))))
        let prob_f64 = self.prob.to_f64().unwrap();
        let n_f64 = self.n as f64;
        let log_odds = (prob_f64 / (1.0 - prob_f64)).ln();
        let log_partition = n_f64 * (1.0 + log_odds.exp()).ln();

        // For symbolic computation, we'll represent the binomial coefficient as a constant
        // In practice, this would need to be computed for each specific x value
        let expr = x * symbolic_const(log_odds) - symbolic_const(log_partition);

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("n".to_string(), n_f64);
        parameters.insert("prob".to_string(), prob_f64);
        parameters.insert("log_odds".to_string(), log_odds);
        parameters.insert("log_partition".to_string(), log_partition);

        crate::exponential_family::symbolic::SymbolicLogDensity::new(
            expr,
            parameters,
            vec!["x".to_string()],
        )
    }

    fn generate_optimized_function(
        &self,
    ) -> crate::exponential_family::symbolic::OptimizedFunction<u64, T> {
        use std::collections::HashMap;

        // Pre-compute constants
        let prob_f64 = self.prob.to_f64().unwrap();
        let n_f64 = self.n as f64;
        let log_odds = (prob_f64 / (1.0 - prob_f64)).ln();
        let log_partition = n_f64 * (1.0 + log_odds.exp()).ln();

        // Convert back to T type
        let log_odds_t = T::from(log_odds).unwrap();
        let log_partition_t = T::from(log_partition).unwrap();

        // Create optimized function
        let n_copy = self.n;
        let function = Box::new(move |x: &u64| -> T {
            let x_t = T::from(*x).unwrap();
            let binomial_coeff = if let (Some(n_f64), Some(x_f64)) = (n_copy.to_f64(), x.to_f64()) {
                let log_coeff = {
                    let (ln_gamma_n_plus_1, _) = (n_f64 + 1.0).ln_gamma();
                    let (ln_gamma_k_plus_1, _) = (x_f64 + 1.0).ln_gamma();
                    let (ln_gamma_n_minus_k_plus_1, _) = (n_f64 - x_f64 + 1.0).ln_gamma();
                    ln_gamma_n_plus_1 - ln_gamma_k_plus_1 - ln_gamma_n_minus_k_plus_1
                };
                T::from(log_coeff).unwrap()
            } else {
                T::zero()
            };

            binomial_coeff + x_t * log_odds_t - log_partition_t
        });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("n".to_string(), n_f64);
        constants.insert("prob".to_string(), prob_f64);
        constants.insert("log_odds".to_string(), log_odds);
        constants.insert("log_partition".to_string(), log_partition);

        let source_expression = format!(
            "Binomial(n={}, p={prob_f64}): log C(n,x) + x * {log_odds} - {log_partition}",
            self.n
        );

        crate::exponential_family::symbolic::OptimizedFunction::new(
            function,
            constants,
            source_expression,
        )
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<u64, T> for Binomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _prob_f64 = self.prob.to_f64().unwrap();
        let _n_f64 = self.n as f64;
        let _log_odds = (_prob_f64 / (1.0 - _prob_f64)).ln();
        let _log_partition = _n_f64 * (1.0 + _log_odds.exp()).ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Binomial distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
