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

use measures_core::primitive::lebesgue::LebesgueMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
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

    fn from_natural(param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let shape = eta1 + T::one();
        let rate = -eta2;
        Self::new(shape, rate)
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [x.ln(), *x]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<T, T>>::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<T, T>>::NaturalParam, T) {
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

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> measures_exponential_family::JITOptimizer<T, T> for Gamma<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<measures_exponential_family::JITFunction, measures_exponential_family::JITError>
    {
        let _shape_f64 = self.shape.to_f64().unwrap();
        let _rate_f64 = self.rate.to_f64().unwrap();
        let (_log_gamma_alpha, _) = _shape_f64.ln_gamma();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            measures_exponential_family::JITError::UnsupportedExpression(
                "Gamma distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Implementation of HasLogDensity for Gamma distribution
impl<T: Float> measures_core::HasLogDensity<T, T> for Gamma<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        if *x > T::zero() {
            // Gamma PDF: f(x|α,β) = (β^α / Γ(α)) * x^(α-1) * exp(-βx)
            // log f(x|α,β) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx
            let alpha = self.shape;
            let beta = self.rate;

            alpha * beta.ln() - gamma_ln(alpha) + (alpha - T::one()) * x.ln() - beta * *x
        } else {
            // Outside support, return negative infinity
            T::neg_infinity()
        }
    }
}
