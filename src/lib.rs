//! A library for measure theory and probability distributions.
//!
//! This crate provides a type-safe framework for working with measure theory
//! and probability distributions, with a focus on:
//!
//! 1. **Type-safe measure theory**: Clear separation between measures and their densities
//! 2. **Efficient log-density computation**: Optimized for numerical stability and performance  
//! 3. **Generic numeric types**: Support for f64, f32, dual numbers (autodiff), etc.
//! 4. **Automatic differentiation ready**: Same log-density works with dual numbers
//! 5. **Exponential family support**: Specialized implementations for exponential families
//!
//! # Quick Start
//!
//! ```rust
//! use measures::{Normal, Measure};
//!
//! let normal = Normal::new(0.0, 1.0);
//! let x = 0.5;
//!
//! // Compute log-density with respect to root measure (Lebesgue)
//! let ld = normal.log_density();
//! let log_density_value: f64 = ld.at(&x);
//!
//! // Compute log-density with respect to different base measure
//! let other_normal = Normal::new(1.0, 2.0);
//! let ld_wrt = normal.log_density().wrt(other_normal);
//! let relative_log_density: f64 = ld_wrt.at(&x);
//! ```

// Core abstractions
pub mod core;
pub mod distributions;
pub mod exponential_family;
pub mod measures;
pub mod statistics;
pub mod traits;

// Re-export key types for convenient access
pub use core::{DotProduct, EvaluateAt, HasLogDensity, LogDensity, LogDensityTrait, Measure, PrimitiveMeasure};
pub use distributions::Normal;
pub use measures::{CountingMeasure, Dirac, LebesgueMeasure};
