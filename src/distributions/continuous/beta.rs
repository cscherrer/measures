//! Beta distribution implementation.
//!
//! This module provides the Beta distribution, which is a continuous probability
//! distribution characterized by its shape parameters α and β. The density is
//! computed with respect to Lebesgue measure on [0,1].
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Beta;
//! use measures::LogDensityBuilder;
//!
//! let beta = Beta::new(2.0, 3.0); // Shape parameters α = 2.0, β = 3.0
//!
//! // Compute log-density at x = 0.3
//! let ld = beta.log_density();
//! let log_density_value: f64 = ld.at(&0.3);
//! ```

use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::Float;
use special::Gamma;

/// Beta distribution Beta(α, β).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [α - 1, β - 1]
/// - Sufficient statistics: T(x) = [log(x), log(1-x)]
/// - Log partition: A(η) = log B(η₁ + 1, η₂ + 1) = log Γ(η₁ + 1) + log Γ(η₂ + 1) - log Γ(η₁ + η₂ + 2)
/// - Base measure: Lebesgue measure on [0, 1]
#[derive(Clone, Debug)]
pub struct Beta<T> {
    pub alpha: T, // α
    pub beta: T,  // β
}

impl<T: Float> Default for Beta<T> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            beta: T::one(),
        }
    }
}

impl<T: Float> MeasureMarker for Beta<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Beta<T> {
    /// Create a new beta distribution with given shape parameters.
    pub fn new(alpha: T, beta: T) -> Self {
        assert!(alpha > T::zero(), "Alpha parameter must be positive");
        assert!(beta > T::zero(), "Beta parameter must be positive");
        Self { alpha, beta }
    }

    /// Get the mean α/(α+β)
    pub fn mean(&self) -> T {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get the variance αβ/((α+β)²(α+β+1))
    pub fn variance(&self) -> T {
        let sum = self.alpha + self.beta;
        let numerator = self.alpha * self.beta;
        let denominator = sum * sum * (sum + T::one());
        numerator / denominator
    }
}

impl<T: Float> Measure<T> for Beta<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        x > T::zero() && x < T::one()
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Beta<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let alpha = eta1 + T::one();
        let beta = eta2 + T::one();
        Self::new(alpha, beta)
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [x.ln(), (T::one() - *x).ln()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_params = [self.alpha - T::one(), self.beta - T::one()];

        // Log partition: log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α + β)
        let log_partition =
            gamma_ln(self.alpha) + gamma_ln(self.beta) - gamma_ln(self.alpha + self.beta);

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
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<T, T> for Beta<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: (α-1)log(x) + (β-1)log(1-x) - log B(α,β)
        let alpha_f64 = self.alpha.to_f64().unwrap();
        let beta_f64 = self.beta.to_f64().unwrap();
        let log_beta_fn_f64 = {
            let (ln_gamma_alpha, _) = alpha_f64.ln_gamma();
            let (ln_gamma_beta, _) = beta_f64.ln_gamma();
            let (ln_gamma_sum, _) = (alpha_f64 + beta_f64).ln_gamma();
            ln_gamma_alpha + ln_gamma_beta - ln_gamma_sum
        };

        // Expression: (α-1)log(x) + (β-1)log(1-x) - log B(α,β)
        let one = symbolic_const(1.0);
        let log_x = x.clone(); // Note: rusymbols may not have ln method, using placeholder
        let log_one_minus_x = one - x.clone(); // Note: rusymbols may not have ln method, using placeholder
        let expr = symbolic_const(alpha_f64 - 1.0) * log_x
            + symbolic_const(beta_f64 - 1.0) * log_one_minus_x
            - symbolic_const(log_beta_fn_f64);

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("alpha".to_string(), alpha_f64);
        parameters.insert("beta".to_string(), beta_f64);
        parameters.insert("log_beta_fn".to_string(), log_beta_fn_f64);

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
        let alpha_f64 = self.alpha.to_f64().unwrap();
        let beta_f64 = self.beta.to_f64().unwrap();
        let log_beta_fn_f64 = {
            let (ln_gamma_alpha, _) = alpha_f64.ln_gamma();
            let (ln_gamma_beta, _) = beta_f64.ln_gamma();
            let (ln_gamma_sum, _) = (alpha_f64 + beta_f64).ln_gamma();
            ln_gamma_alpha + ln_gamma_beta - ln_gamma_sum
        };
        let alpha_minus_1 = alpha_f64 - 1.0;
        let beta_minus_1 = beta_f64 - 1.0;

        // Convert back to T type
        let alpha_minus_1_t = T::from(alpha_minus_1).unwrap();
        let beta_minus_1_t = T::from(beta_minus_1).unwrap();
        let log_beta_fn_t = T::from(log_beta_fn_f64).unwrap();

        // Create optimized function
        let function = Box::new(move |x: &T| -> T {
            alpha_minus_1_t * x.ln() + beta_minus_1_t * (T::one() - *x).ln() - log_beta_fn_t
        });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("alpha".to_string(), alpha_f64);
        constants.insert("beta".to_string(), beta_f64);
        constants.insert("log_beta_fn".to_string(), log_beta_fn_f64);
        constants.insert("alpha_minus_1".to_string(), alpha_minus_1);
        constants.insert("beta_minus_1".to_string(), beta_minus_1);

        let source_expression = format!(
            "Beta(α={alpha_f64}, β={beta_f64}): {alpha_minus_1} * log(x) + {beta_minus_1} * log(1-x) - {log_beta_fn_f64}"
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
impl<T> crate::exponential_family::jit::JITOptimizer<T, T> for Beta<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        let alpha_f64 = self.alpha.to_f64().unwrap();
        let beta_f64 = self.beta.to_f64().unwrap();
        let log_beta_fn_f64 = {
            let (ln_gamma_alpha, _) = alpha_f64.ln_gamma();
            let (ln_gamma_beta, _) = beta_f64.ln_gamma();
            let (ln_gamma_sum, _) = (alpha_f64 + beta_f64).ln_gamma();
            ln_gamma_alpha + ln_gamma_beta - ln_gamma_sum
        };
        let alpha_minus_1 = alpha_f64 - 1.0;
        let beta_minus_1 = beta_f64 - 1.0;

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            crate::exponential_family::jit::JITError::UnsupportedExpression(
                "Beta distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
