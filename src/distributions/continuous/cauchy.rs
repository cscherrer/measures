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

use crate::core::{HasLogDensity, Measure, MeasureMarker};
use crate::core::types::False;
use crate::measures::primitive::LebesgueMeasure;
use num_traits::{Float, NumCast};
use std::f64::consts::PI;

/// Cauchy distribution with location parameter `location` and scale parameter `scale`.
///
/// The Cauchy distribution has probability density function:
/// f(x|x₀,γ) = 1/(πγ[1 + ((x-x₀)/γ)²])
///
/// This is NOT an exponential family distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct Cauchy<T> {
    location: T,
    scale: T,
}

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
    pub fn standard() -> Self
    where
        T: NumCast,
    {
        Self::new(
            T::from(0.0).unwrap(),
            T::from(1.0).unwrap(),
        )
    }

    /// Get the location parameter.
    pub fn location(&self) -> T {
        self.location
    }

    /// Get the scale parameter.
    pub fn scale(&self) -> T {
        self.scale
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
}

impl<T: Float> MeasureMarker for Cauchy<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;  // Cauchy is NOT an exponential family
}

impl<T: Float + Clone> Measure<T> for Cauchy<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true  // Cauchy has support on all real numbers
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

/// Manual implementation of HasLogDensity for Cauchy since it's not an exponential family.
///
/// This demonstrates how non-exponential family distributions implement log-density computation.
impl<T: Float + Clone + NumCast> HasLogDensity<T, T> for Cauchy<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        self.log_pdf(*x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LogDensityBuilder;

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
        let _cauchy = Cauchy::new(0.0, -1.0);  // Should panic
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
        use crate::core::types::TypeLevelBool;
        
        let _cauchy = Cauchy::new(0.0, 1.0);
        
        // Verify type-level markers
        assert!(!<Cauchy<f64> as MeasureMarker>::IsExponentialFamily::VALUE);
        assert!(!<Cauchy<f64> as MeasureMarker>::IsPrimitive::VALUE);
    }
} 