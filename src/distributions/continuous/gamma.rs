//! Gamma distribution implementation.
//!
//! This module provides the Gamma distribution, which is a continuous probability
//! distribution characterized by its shape (α) and rate (β) parameters. The density is
//! computed with respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Gamma;
//! use measures::LogDensityBuilder;
//!
//! let gamma = Gamma::new(2.0, 1.5); // Shape α = 2.0, rate β = 1.5
//!
//! // Compute log-density at x = 1.0
//! let ld = gamma.log_density();
//! let log_density_value: f64 = ld.at(&1.0);
//! ```

use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::Float;
use special::Gamma as GammaTrait;

/// Gamma distribution Gamma(α, β).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [α - 1, -β]
/// - Sufficient statistics: T(x) = [log(x), x]
/// - Log partition: A(η) = log Γ(η₁ + 1) - (η₁ + 1) log(-η₂)
/// - Base measure: Lebesgue measure on (0, ∞)
#[derive(Clone, Debug)]
pub struct Gamma<T> {
    pub shape: T, // α
    pub rate: T,  // β
}

impl<T: Float> Default for Gamma<T> {
    fn default() -> Self {
        Self {
            shape: T::one(),
            rate: T::one(),
        }
    }
}

impl<T: Float> MeasureMarker for Gamma<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Gamma<T> {
    /// Create a new gamma distribution with given shape and rate parameters.
    pub fn new(shape: T, rate: T) -> Self {
        assert!(shape > T::zero(), "Shape parameter must be positive");
        assert!(rate > T::zero(), "Rate parameter must be positive");
        Self { shape, rate }
    }

    /// Create a gamma distribution from shape and scale parameters.
    pub fn from_shape_scale(shape: T, scale: T) -> Self {
        Self::new(shape, scale.recip())
    }

    /// Get the scale parameter (1/β)
    pub fn scale(&self) -> T {
        self.rate.recip()
    }

    /// Get the mean α/β
    pub fn mean(&self) -> T {
        self.shape / self.rate
    }

    /// Get the variance α/β²
    pub fn variance(&self) -> T {
        self.shape / (self.rate * self.rate)
    }
}

impl<T: Float> Measure<T> for Gamma<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        x > T::zero()
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Gamma<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let shape = eta1 + T::one();
        let rate = -eta2;
        Self::new(shape, rate)
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [x.ln(), *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_params = [self.shape - T::one(), -self.rate];

        // Log partition: log Γ(α) - α log(β)
        let log_partition = gamma_ln(self.shape) - self.shape * self.rate.ln();

        (natural_params, log_partition)
    }
}

// Helper function for log gamma function
// Note: This is a simplified implementation. In practice, you'd want to use
// a more robust implementation from a math library.
fn gamma_ln<T: Float>(x: T) -> T {
    // For now, use the special crate for demonstration
    if let Some(x_f64) = x.to_f64() {
        let (ln_gamma_val, _sign) = x_f64.ln_gamma();
        T::from(ln_gamma_val).unwrap()
    } else {
        // Fallback for types that don't convert to f64
        // This is a very rough approximation
        x.ln()
    }
}

// Symbolic optimization implementation
#[cfg(feature = "symbolic")]
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<T, T> for Gamma<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: (α-1)log(x) - βx + α log(β) - log Γ(α)
        let shape_f64 = self.shape.to_f64().unwrap();
        let rate_f64 = self.rate.to_f64().unwrap();
        let (log_gamma_alpha, _) = shape_f64.ln_gamma();

        // Expression: (α-1)log(x) - βx + α log(β) - log Γ(α)
        let log_x = x.clone(); // Note: rusymbols may not have ln method, using placeholder
        let expr = symbolic_const(shape_f64 - 1.0) * log_x - symbolic_const(rate_f64) * x
            + symbolic_const(shape_f64 * rate_f64.ln() - log_gamma_alpha);

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("shape".to_string(), shape_f64);
        parameters.insert("rate".to_string(), rate_f64);
        parameters.insert("log_gamma_alpha".to_string(), log_gamma_alpha);

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
        let shape_f64 = self.shape.to_f64().unwrap();
        let rate_f64 = self.rate.to_f64().unwrap();
        let (log_gamma_alpha, _) = shape_f64.ln_gamma();
        let constant_term = shape_f64 * rate_f64.ln() - log_gamma_alpha;
        let log_coeff = shape_f64 - 1.0;

        // Convert back to T type
        let rate_t = T::from(rate_f64).unwrap();
        let constant_term_t = T::from(constant_term).unwrap();
        let log_coeff_t = T::from(log_coeff).unwrap();

        // Create optimized function
        let function =
            Box::new(move |x: &T| -> T { log_coeff_t * x.ln() - rate_t * *x + constant_term_t });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("shape".to_string(), shape_f64);
        constants.insert("rate".to_string(), rate_f64);
        constants.insert("log_gamma_alpha".to_string(), log_gamma_alpha);
        constants.insert("constant_term".to_string(), constant_term);

        let source_expression = format!(
            "Gamma(α={shape_f64}, β={rate_f64}): {log_coeff} * log(x) - {rate_f64} * x + {constant_term}"
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
impl<T> crate::exponential_family::jit::JITOptimizer<T, T> for Gamma<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let _shape_f64 = self.shape.to_f64().unwrap();
        let _rate_f64 = self.rate.to_f64().unwrap();
        let (_log_gamma_alpha, _) = _shape_f64.ln_gamma();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Gamma distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
