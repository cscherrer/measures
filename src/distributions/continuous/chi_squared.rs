//! Chi-squared distribution implementation.
//!
//! This module provides the Chi-squared distribution, which is a continuous probability
//! distribution that is a special case of the Gamma distribution with shape = k/2 and rate = 1/2,
//! where k is the degrees of freedom. The density is computed with respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::ChiSquared;
//! use measures::LogDensityBuilder;
//!
//! let chi_sq = ChiSquared::new(3.0); // 3 degrees of freedom
//!
//! // Compute log-density at x = 2.0
//! let ld = chi_sq.log_density();
//! let log_density_value: f64 = ld.at(&2.0);
//! ```

use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::Float;
use special::Gamma as GammaTrait;

/// Chi-squared distribution χ²(k).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [k/2 - 1, -1/2]
/// - Sufficient statistics: T(x) = [log(x), x]
/// - Log partition: A(η) = log Γ(η₁ + 1) - (η₁ + 1) log(-η₂)
/// - Base measure: Lebesgue measure on (0, ∞)
///
/// This is equivalent to Gamma(k/2, 1/2).
#[derive(Clone, Debug)]
pub struct ChiSquared<T> {
    pub degrees_of_freedom: T, // k
}

impl<T: Float> Default for ChiSquared<T> {
    fn default() -> Self {
        Self {
            degrees_of_freedom: T::one(),
        }
    }
}

impl<T: Float> MeasureMarker for ChiSquared<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> ChiSquared<T> {
    /// Create a new chi-squared distribution with given degrees of freedom.
    pub fn new(degrees_of_freedom: T) -> Self {
        assert!(
            degrees_of_freedom > T::zero(),
            "Degrees of freedom must be positive"
        );
        Self { degrees_of_freedom }
    }

    /// Get the shape parameter (k/2)
    pub fn shape(&self) -> T {
        self.degrees_of_freedom / T::from(2.0).unwrap()
    }

    /// Get the rate parameter (1/2)
    pub fn rate(&self) -> T {
        T::from(0.5).unwrap()
    }

    /// Get the mean (which equals k)
    pub fn mean(&self) -> T {
        self.degrees_of_freedom
    }

    /// Get the variance (2k)
    pub fn variance(&self) -> T {
        T::from(2.0).unwrap() * self.degrees_of_freedom
    }
}

impl<T: Float> Measure<T> for ChiSquared<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        x > T::zero()
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for ChiSquared<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, _eta2] = param;
        // For chi-squared: eta1 = k/2 - 1, eta2 = -1/2
        // So k = 2 * (eta1 + 1)
        let degrees_of_freedom = T::from(2.0).unwrap() * (eta1 + T::one());
        Self::new(degrees_of_freedom)
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [x.ln(), *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let shape = self.shape();
        let rate = self.rate();

        let natural_params = [shape - T::one(), -rate];

        // Log partition: log Γ(k/2) - (k/2) log(1/2) = log Γ(k/2) + (k/2) log(2)
        let log_partition = gamma_ln(shape) + shape * T::from(2.0).unwrap().ln();

        (natural_params, log_partition)
    }
}

// Helper function for log gamma function
fn gamma_ln<T: Float>(x: T) -> T {
    if let Some(x_f64) = x.to_f64() {
        let (ln_gamma_val, _sign) = x_f64.ln_gamma();
        T::from(ln_gamma_val).unwrap()
    } else {
        // Fallback for types that don't convert to f64
        x.ln()
    }
}

// Symbolic optimization implementation
#[cfg(feature = "symbolic")]
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<T, T> for ChiSquared<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: (k/2-1)log(x) - x/2 + (k/2)log(2) - log Γ(k/2)
        let k_f64 = self.degrees_of_freedom.to_f64().unwrap();
        let shape = k_f64 / 2.0;
        let (log_gamma_shape, _) = shape.ln_gamma();
        let constant_term = shape * 2.0_f64.ln() - log_gamma_shape;

        // Expression: (k/2-1)log(x) - x/2 + constant
        // NOTE: rusymbols doesn't support .ln() method, so we represent log(x)
        // symbolically as just x for now. The actual logarithm computation
        // happens in generate_optimized_function() where we use real math.
        let log_x_placeholder = x.clone();
        let expr = symbolic_const(shape - 1.0) * log_x_placeholder - symbolic_const(0.5) * x
            + symbolic_const(constant_term);

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("degrees_of_freedom".to_string(), k_f64);
        parameters.insert("shape".to_string(), shape);
        parameters.insert("log_gamma_shape".to_string(), log_gamma_shape);
        parameters.insert("constant_term".to_string(), constant_term);

        crate::exponential_family::symbolic::SymbolicLogDensity::new(
            expr,
            parameters,
            vec!["x".to_string()],
        )
    }

    fn generate_optimized_function(
        &self,
    ) -> crate::exponential_family::symbolic::OptimizedFunction<T, T> {
        use std::collections::HashMap;

        // Pre-compute constants
        let k_f64 = self.degrees_of_freedom.to_f64().unwrap();
        let shape = k_f64 / 2.0;
        let (log_gamma_shape, _) = shape.ln_gamma();
        let constant_term = shape * 2.0_f64.ln() - log_gamma_shape;
        let log_coeff = shape - 1.0;

        // Convert back to T type
        let log_coeff_t = T::from(log_coeff).unwrap();
        let constant_term_t = T::from(constant_term).unwrap();
        let half_t = T::from(0.5).unwrap();

        // Create optimized function
        let function =
            Box::new(move |x: &T| -> T { log_coeff_t * x.ln() - half_t * *x + constant_term_t });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("degrees_of_freedom".to_string(), k_f64);
        constants.insert("shape".to_string(), shape);
        constants.insert("log_gamma_shape".to_string(), log_gamma_shape);
        constants.insert("constant_term".to_string(), constant_term);

        let source_expression =
            format!("ChiSquared(k={k_f64}): {log_coeff} * log(x) - 0.5 * x + {constant_term}");

        crate::exponential_family::symbolic::OptimizedFunction::new(
            function,
            constants,
            source_expression,
        )
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<T, T> for ChiSquared<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _k_f64 = self.degrees_of_freedom.to_f64().unwrap();
        let _shape = _k_f64 / 2.0;
        let (_log_gamma_shape, _) = _shape.ln_gamma();
        let _constant_term = _shape * 2.0_f64.ln() - _log_gamma_shape;
        let _log_coeff = _shape - 1.0;

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "ChiSquared distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
