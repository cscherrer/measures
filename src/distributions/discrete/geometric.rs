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

use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::counting::CountingMeasure;
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

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta] = param;
        // η = log(1-p) => 1-p = exp(η) => p = 1 - exp(η)
        let failure_prob = eta.exp();
        let prob = T::one() - failure_prob;
        Self::new(prob)
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        CountingMeasure::<u64>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_param = [self.failure_prob().ln()];

        // Log partition: A(η) = -log(1 - exp(η)) = -log(p)
        let log_partition = -self.prob.ln();

        (natural_param, log_partition)
    }
}

// Symbolic optimization implementation
#[cfg(feature = "symbolic")]
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<u64, T> for Geometric<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: x * log(1-p) + log(p)
        let prob_f64 = self.prob.to_f64().unwrap();
        let failure_prob = 1.0 - prob_f64;
        let log_failure_prob = failure_prob.ln();
        let log_prob = prob_f64.ln();

        // Expression: x * log(1-p) + log(p)
        let expr = x * symbolic_const(log_failure_prob) + symbolic_const(log_prob);

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("prob".to_string(), prob_f64);
        parameters.insert("failure_prob".to_string(), failure_prob);
        parameters.insert("log_failure_prob".to_string(), log_failure_prob);
        parameters.insert("log_prob".to_string(), log_prob);

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
        let failure_prob = 1.0 - prob_f64;
        let log_failure_prob = failure_prob.ln();
        let log_prob = prob_f64.ln();

        // Convert back to T type
        let log_failure_prob_t = T::from(log_failure_prob).unwrap();
        let log_prob_t = T::from(log_prob).unwrap();

        // Create optimized function
        let function = Box::new(move |x: &u64| -> T {
            let x_t = T::from(*x).unwrap();
            x_t * log_failure_prob_t + log_prob_t
        });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("prob".to_string(), prob_f64);
        constants.insert("failure_prob".to_string(), failure_prob);
        constants.insert("log_failure_prob".to_string(), log_failure_prob);
        constants.insert("log_prob".to_string(), log_prob);

        let source_expression =
            format!("Geometric(p={prob_f64}): x * {log_failure_prob} + {log_prob}");

        crate::exponential_family::symbolic::OptimizedFunction::new(
            function,
            constants,
            source_expression,
        )
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<u64, T> for Geometric<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _prob_f64 = self.prob.to_f64().unwrap();
        let _failure_prob = 1.0 - _prob_f64;
        let _log_failure_prob = _failure_prob.ln();
        let _log_prob = _prob_f64.ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Geometric distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
