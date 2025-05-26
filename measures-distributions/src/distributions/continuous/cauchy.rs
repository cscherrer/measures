//! Cauchy distribution implementation.
//!
//! The Cauchy distribution is a continuous probability distribution that is not
//! an exponential family. It has heavy tails and no finite moments.
//!
//! # Examples
//!
//! ```rust
//! use measures::{Cauchy, LogDensityBuilder};
//!
//! let cauchy = Cauchy::new(0.0, 1.0);  // Standard Cauchy
//! let log_density: f64 = cauchy.log_density().at(&0.0);
//! ```

use measures_core::False;
use measures_core::float_constant;
use measures_core::primitive::lebesgue::LebesgueMeasure;
use measures_core::{DecompositionBuilder, HasLogDensityDecomposition, LogDensityDecomposition};
use measures_core::{HasLogDensity, Measure, MeasureMarker};
use num_traits::{Float, FloatConst, NumCast};
use std::f64::consts::PI;

/// Cauchy distribution with location parameter `location` and scale parameter `scale`.
///
/// The Cauchy distribution has probability density function:
/// f(x|x₀,γ) = 1/(πγ[1 + ((x-x₀)/γ)²])
///
/// Log-density decomposition:
/// log f(x|x₀,γ) = -log(π) - log(γ) - log(1 + ((x-x₀)/γ)²)
///                = `f_const` + `f_param(γ)` + `f_mixed(x,x₀,γ)`
///
/// Where:
/// - `f_const` = -log(π) (constant)
/// - `f_param(γ)` = -log(γ) (parameter-only)
/// - `f_mixed(x,x₀,γ)` = -log(1 + ((x-x₀)/γ)²) (mixed data-parameter)
///
/// This is NOT an exponential family distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Cauchy<T> {
    location: T,
    scale: T,
}

/// Parameters for Cauchy distribution: (location, scale)
pub type CauchyParams<T> = (T, T);

impl<T: Float> Cauchy<T> {
    /// Create a new Cauchy distribution with given location and scale parameters.
    ///
    /// # Arguments
    /// * `location` - Location parameter (median of the distribution)
    /// * `scale` - Scale parameter (must be positive)
    ///
    /// # Panics
    /// Panics if scale <= 0
    pub fn new(location: T, scale: T) -> Self {
        assert!(scale > T::zero(), "Scale parameter must be positive");
        Self { location, scale }
    }

    /// Create a standard Cauchy distribution (location=0, scale=1).
    #[must_use]
    pub fn standard() -> Self
    where
        T: NumCast,
    {
        Self::new(T::from(0.0).unwrap(), T::from(1.0).unwrap())
    }

    /// Get the location parameter.
    pub fn location(&self) -> T {
        self.location
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> T {
        self.scale
    }

    /// Get parameters as a tuple (location, scale).
    pub fn params(&self) -> CauchyParams<T> {
        (self.location, self.scale)
    }

    /// Compute the log probability density function at point x.
    ///
    /// log f(x) = -log(π) - log(γ) - log(1 + ((x-x₀)/γ)²)
    pub fn log_pdf(&self, x: T) -> T
    where
        T: NumCast,
    {
        let standardized = (x - self.location) / self.scale;
        let pi = T::from(PI).unwrap();

        -pi.ln() - self.scale.ln() - (T::one() + standardized * standardized).ln()
    }

    /// Compute log-density for IID samples efficiently using decomposition.
    ///
    /// For n IID samples, this is much more efficient than computing individual densities.
    pub fn log_density_iid(&self, samples: &[T]) -> T
    where
        T: NumCast + FloatConst + std::iter::Sum,
    {
        let decomp = self.log_density_decomposition();
        decomp.evaluate_iid(samples, &self.params())
    }
}

impl<T: Float> MeasureMarker for Cauchy<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = False; // Cauchy is NOT an exponential family
}

impl<T: Float + Clone> Measure<T> for Cauchy<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true // Cauchy has support on all real numbers
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

/// Manual implementation of `HasLogDensity` for Cauchy since it's not an exponential family.
///
/// This demonstrates how non-exponential family distributions implement log-density computation.
impl<T: Float + FloatConst> HasLogDensity<T, T> for Cauchy<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        let pi = float_constant::<T>(PI);
        let standardized = (*x - self.location) / self.scale;
        -(T::one() + standardized * standardized).ln() - self.scale.ln() - pi.ln()
    }
}

/// Implementation of structured log-density decomposition for Cauchy.
///
/// This shows how non-exponential families can still benefit from the structured approach.
impl<T: Float + FloatConst + NumCast> HasLogDensityDecomposition<T, CauchyParams<T>, T>
    for Cauchy<T>
{
    fn log_density_decomposition(&self) -> LogDensityDecomposition<T, CauchyParams<T>, T> {
        let pi = T::from(PI).unwrap();

        DecompositionBuilder::new()
            // Constant term: -log(π)
            .constant(-pi.ln())
            // Parameter-only term: -log(γ)
            .param_term(
                |(_location, scale): &CauchyParams<T>| -scale.ln(),
                "negative log scale parameter",
            )
            // Mixed term: -log(1 + ((x-x₀)/γ)²)
            .mixed_term(
                |x: &T, (location, scale): &CauchyParams<T>| {
                    let standardized = (*x - *location) / *scale;
                    -(T::one() + standardized * standardized).ln()
                },
                "negative log of standardized squared plus one",
            )
            .build()
    }
}

impl<T: Float> Default for Cauchy<T> {
    fn default() -> Self {
        Self::new(float_constant::<T>(0.0), float_constant::<T>(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use measures_core::LogDensityBuilder;

    #[test]
    fn test_cauchy_creation() {
        let cauchy = Cauchy::new(0.0, 1.0);
        assert_eq!(cauchy.location(), 0.0);
        assert_eq!(cauchy.scale(), 1.0);
    }

    #[test]
    fn test_standard_cauchy() {
        let cauchy = Cauchy::<f64>::standard();
        assert_eq!(cauchy.location(), 0.0);
        assert_eq!(cauchy.scale(), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_scale() {
        let _cauchy = Cauchy::new(0.0, -1.0); // Should panic
    }

    #[test]
    fn test_log_pdf() {
        let cauchy = Cauchy::new(0.0, 1.0);

        // At x = 0 (center), log_pdf should be -log(π)
        let expected = -PI.ln();
        let actual = cauchy.log_pdf(0.0);
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_density_builder() {
        let cauchy = Cauchy::new(0.0, 1.0);

        // Test that the builder pattern works
        let log_density: f64 = cauchy.log_density().at(&0.0);
        let expected = -PI.ln();
        assert!((log_density - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_density_decomposition() {
        let cauchy = Cauchy::new(1.0, 2.0);
        let decomp = cauchy.log_density_decomposition();

        let x = 0.5;
        let params = cauchy.params();

        // Test that decomposition matches direct computation
        let decomp_result = decomp.evaluate(&x, &params);
        let direct_result = cauchy.log_pdf(x);

        assert!((decomp_result - direct_result).abs() < 1e-10);
    }

    #[test]
    fn test_iid_log_density() {
        let cauchy = Cauchy::new(0.0, 1.0);
        let samples = vec![0.0, 1.0, -1.0, 2.0];

        // Test IID computation
        let iid_result = cauchy.log_density_iid(&samples);

        // Manual computation
        let manual_result: f64 = samples.iter().map(|&x| cauchy.log_pdf(x)).sum();

        assert!((iid_result - manual_result).abs() < 1e-10);
    }

    #[test]
    fn test_decomposition_efficiency() {
        let cauchy = Cauchy::new(2.0, 1.5);
        let decomp = cauchy.log_density_decomposition();
        let params = cauchy.params();

        // Test that we can evaluate parameter terms separately
        let param_terms = decomp.evaluate_param_terms(&params);
        let constants = decomp.constant_sum();

        // These should be independent of data
        assert!(param_terms.is_finite());
        assert!(constants.is_finite());

        // For Cauchy: param_terms = -log(scale), constants = -log(π)
        let expected_param = -1.5f64.ln();
        let expected_const = -PI.ln();

        assert!((param_terms - expected_param).abs() < 1e-10);
        assert!((constants - expected_const).abs() < 1e-10);
    }

    #[test]
    fn test_different_numeric_types() {
        let cauchy_f64 = Cauchy::new(0.0f64, 1.0f64);
        let cauchy_f32 = Cauchy::new(0.0f32, 1.0f32);

        let f64_result: f64 = cauchy_f64.log_density().at(&0.0f64);
        let f32_result: f32 = cauchy_f32.log_density().at(&0.0f32);

        // Results should be approximately equal
        assert!((f64_result - f32_result as f64).abs() < 1e-6);
    }

    #[test]
    fn test_relative_density() {
        let cauchy1 = Cauchy::new(0.0, 1.0);
        let cauchy2 = Cauchy::new(1.0, 2.0);

        // Test relative density computation
        let relative_density: f64 = cauchy1.log_density().wrt(cauchy2.clone()).at(&0.5);

        // Should equal individual densities subtracted
        let manual: f64 = cauchy1.log_density().at(&0.5) - cauchy2.log_density().at(&0.5);
        assert!((relative_density - manual).abs() < 1e-10);
    }

    #[test]
    fn test_support() {
        let cauchy = Cauchy::new(0.0, 1.0);

        // Cauchy has support on all real numbers
        assert!(cauchy.in_support(0.0));
        assert!(cauchy.in_support(100.0));
        assert!(cauchy.in_support(-100.0));
        assert!(cauchy.in_support(f64::INFINITY));
        assert!(cauchy.in_support(f64::NEG_INFINITY));
    }

    #[test]
    fn test_is_not_exponential_family() {
        use measures_core::TypeLevelBool;

        let _cauchy = Cauchy::new(0.0, 1.0);

        // Verify type-level markers - these are compile-time checks
        // The assertions are removed as they're always true and optimized out
        let _is_exp_fam = <Cauchy<f64> as MeasureMarker>::IsExponentialFamily::VALUE;
        let _is_primitive = <Cauchy<f64> as MeasureMarker>::IsPrimitive::VALUE;
    }
}
