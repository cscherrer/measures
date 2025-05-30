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

use measures_core::primitive::lebesgue::LebesgueMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
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

    fn from_natural(param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let alpha = eta1 + T::one();
        let beta = eta2 + T::one();
        Self::new(alpha, beta)
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [x.ln(), (T::one() - *x).ln()]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<T, T>>::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<T, T>>::NaturalParam, T) {
        let natural_params = [self.alpha - T::one(), self.beta - T::one()];

        // Log partition: log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α + β)
        let log_partition = {
            let alpha_f64 = self.alpha.to_f64().unwrap();
            let beta_f64 = self.beta.to_f64().unwrap();
            let (ln_gamma_alpha, _) = alpha_f64.ln_gamma();
            let (ln_gamma_beta, _) = beta_f64.ln_gamma();
            let (ln_gamma_sum, _) = (alpha_f64 + beta_f64).ln_gamma();
            let result_f64 = ln_gamma_alpha + ln_gamma_beta - ln_gamma_sum;
            T::from(result_f64).unwrap()
        };

        (natural_params, log_partition)
    }
}

// Implementation of HasLogDensity for Beta distribution
impl<T: Float> measures_core::HasLogDensity<T, T> for Beta<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        if *x > T::zero() && *x < T::one() {
            // Beta PDF: f(x|α,β) = (Γ(α+β) / (Γ(α)Γ(β))) * x^(α-1) * (1-x)^(β-1)
            // log f(x|α,β) = log(Γ(α+β)) - log(Γ(α)) - log(Γ(β)) + (α-1)*log(x) + (β-1)*log(1-x)
            let alpha = self.alpha;
            let beta = self.beta;

            beta_ln(alpha + beta) - beta_ln(alpha) - beta_ln(beta)
                + (alpha - T::one()) * x.ln()
                + (beta - T::one()) * (T::one() - *x).ln()
        } else {
            // Outside support, return negative infinity
            T::neg_infinity()
        }
    }
}

// Helper function for log gamma function (same as in gamma.rs)
fn beta_ln<T: Float>(x: T) -> T {
    if let Some(x_f64) = x.to_f64() {
        let (ln_gamma_val, _sign) = x_f64.ln_gamma();
        T::from(ln_gamma_val).unwrap()
    } else {
        // Fallback for types that don't convert to f64
        x.ln()
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> measures_exponential_family::JITOptimizer<T, T> for Beta<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<measures_exponential_family::JITFunction, measures_exponential_family::JITError>
    {
        let alpha_f64 = self.alpha.to_f64().unwrap();
        let beta_f64 = self.beta.to_f64().unwrap();
        let _log_beta_fn_f64 = {
            let (ln_gamma_alpha, _) = alpha_f64.ln_gamma();
            let (ln_gamma_beta, _) = beta_f64.ln_gamma();
            let (ln_gamma_sum, _) = (alpha_f64 + beta_f64).ln_gamma();
            ln_gamma_alpha + ln_gamma_beta - ln_gamma_sum
        };
        let _alpha_minus_1 = alpha_f64 - 1.0;
        let _beta_minus_1 = beta_f64 - 1.0;

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            measures_exponential_family::JITError::UnsupportedExpression(
                "Beta distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}
