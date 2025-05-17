//! A library for measure theory and probability distributions.
//!
//! This crate provides a type-safe framework for working with measure theory
//! and probability distributions, with a focus on:
//!
//! 1. A clear separation between measures and their densities
//! 2. Efficient computation of log-densities
//! 3. Type safety for density computations
//! 4. Support for exponential family distributions
//!
//! # Example
//!
//! ```rust
//! use measures::{Normal, HasDensity};
//!
//! let normal = Normal::new(0.0, 1.0);
//!
//! // Compute density
//! let density: f64 = normal.density(&0.0).into();
//!
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(&0.0).into();
//! ```

// Core abstractions
pub mod core;
pub mod distributions;
pub mod exponential_family;
pub mod measures;
pub mod statistics;

// Re-export key types for convenient access
pub use core::{Density, HasDensity, LogDensity, Measure, PrimitiveMeasure};
pub use distributions::Normal;
pub use measures::counting::CountingMeasure;
pub use measures::dirac::Dirac;
pub use measures::lebesgue::LebesgueMeasure;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirac_density() {
        let dirac = Dirac::new(5);
        let counting = CountingMeasure::new();

        // Compute densities
        let density1: f64 = dirac.density(&5).into();
        let density2 = Into::<f64>::into(dirac.density(&5).wrt(&counting));

        // Compute log density directly
        let log_density: f64 = dirac.log_density(&5).into();

        assert_eq!(density1, 1.0);
        assert_eq!(density2, 1.0);
        assert_eq!(log_density, 0.0);
    }

    #[test]
    fn test_working_with_log_density() {
        // Create a standard normal distribution
        let normal = Normal::new(0.0, 1.0);

        // Get log-density as a LogDensity object
        let log_density = normal.log_density(&0.0);

        // Only convert to f64 when needed for numeric computation
        let value: f64 = log_density.into();

        // For standard normal at x=0, log-density should be -log(sqrt(2π))
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((value - expected).abs() < 1e-10);
    }
}
