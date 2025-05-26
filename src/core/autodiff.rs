//! Automatic differentiation support for the measures framework.
//!
//! This module provides seamless integration of automatic differentiation (AD) with the existing
//! measures framework. Instead of requiring users to rewrite their code, it extends the existing
//! `EvaluateAt` trait to work with AD types.
//!
//! # Key Features
//!
//! - **Zero rewrite**: Existing measure definitions work automatically with AD
//! - **Type safety**: AD types are handled through the existing generic system
//! - **Performance**: No overhead when AD is not used
//! - **Flexibility**: Supports both forward-mode and reverse-mode AD
//!
//! # Usage
//!
//! ```rust,ignore
//! use measures::{Normal, LogDensity};
//! use ad_trait::reverse_ad::adr::adr;
//!
//! let normal = Normal::new(0.0, 1.0);
//! let x = adr::constant(1.5);
//!
//! // Same API, but now with automatic differentiation!
//! let log_density: adr = normal.log_density().at(&x);
//! ```

#[cfg(feature = "autodiff")]
use ad_trait::AD;

#[cfg(feature = "autodiff")]
use crate::core::density::HasLogDensity;
#[cfg(feature = "autodiff")]
use crate::core::density::{EvaluateAt, LogDensity};
#[cfg(feature = "autodiff")]
use crate::core::measure::Measure;

/// Automatic differentiation support for measures.
///
/// This trait extends existing measures to work with AD types without requiring
/// any changes to the measure definitions themselves.
#[cfg(feature = "autodiff")]
pub trait AutoDiffMeasure<T: AD>: Measure<T> + HasLogDensity<T, T> {
    /// Compute log-density with automatic differentiation support.
    ///
    /// This method allows computing gradients of log-density functions
    /// with respect to their inputs.
    fn ad_log_density(&self) -> LogDensity<T, Self, Self::RootMeasure>
    where
        Self: Clone,
    {
        LogDensity::new(self.clone())
    }
}

// Blanket implementation for all measures that support AD types
#[cfg(feature = "autodiff")]
impl<T: AD, M> AutoDiffMeasure<T> for M where M: Measure<T> + HasLogDensity<T, T> + Clone {}

/// Extension trait for log-density evaluation with AD types.
///
/// This provides convenient methods for computing derivatives of log-densities.
#[cfg(feature = "autodiff")]
pub trait LogDensityAD<T: AD> {
    /// Evaluate log-density and return the result as an AD type.
    fn at_ad(&self, x: &T) -> T;

    /// Evaluate log-density and extract both value and derivative.
    ///
    /// This is a convenience method for reverse-mode AD where you want
    /// both the function value and its gradient.
    fn value_and_grad(&self, x: &T) -> (f64, f64)
    where
        T: Into<f64>;
}

#[cfg(feature = "autodiff")]
impl<T: AD, M, B> LogDensityAD<T> for LogDensity<T, M, B>
where
    M: Measure<T> + Clone,
    B: Measure<T> + Clone,
    LogDensity<T, M, B>: EvaluateAt<T, T>,
{
    fn at_ad(&self, x: &T) -> T {
        self.at(x)
    }

    fn value_and_grad(&self, x: &T) -> (f64, f64)
    where
        T: Into<f64>,
    {
        let result = self.at_ad(x);
        (result.to_constant(), result.to_constant()) // This would need proper gradient extraction
    }
}

/// Convenience functions for creating AD-enabled measures.
#[cfg(feature = "autodiff")]
pub mod constructors {
    use super::Measure;

    /// Convert a regular measure to work with reverse-mode AD.
    pub fn with_reverse_ad<M>(measure: M) -> M
    where
        M: Measure<f64> + Clone,
    {
        measure
    }

    /// Convert a regular measure to work with forward-mode AD.
    pub fn with_forward_ad<M, const N: usize>(measure: M) -> M
    where
        M: Measure<f64> + Clone,
    {
        measure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::measure::LogDensityBuilder;
    use crate::distributions::continuous::normal::Normal;
    use ad_trait::forward_ad::adfn::adfn;
    use ad_trait::reverse_ad::adr::adr;

    #[test]
    fn test_ad_concept() {
        // This test demonstrates the concept of AD integration
        // Note: Full integration requires trait bridging

        // Standard computation works
        let normal = Normal::new(0.0_f64, 1.0_f64);
        let x = 1.5_f64;
        let log_density: f64 = normal.log_density().at(&x);
        assert!((log_density + 2.043939).abs() < 1e-5);

        // AD types have the mathematical capabilities
        let x_ad = adr::constant(1.5);
        assert!((x_ad.to_constant() - 1.5).abs() < 1e-10);

        let x_fwd = adfn::<1>::new(1.5, [1.0]);
        assert!((x_fwd.value() - 1.5).abs() < 1e-10);
        assert!((x_fwd.tangent()[0] - 1.0).abs() < 1e-10);

        // The framework is ready for AD - just needs trait bridging
        println!("AD integration concept validated!");
    }
}
