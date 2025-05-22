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
//! let density: f64 = normal.log_density(&0.0).into();
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
pub mod traits;

// Re-export key types for convenient access
pub use core::{LogDensity, Measure, PrimitiveMeasure};
pub use distributions::Normal;
pub use measures::counting::CountingMeasure;
pub use measures::dirac::Dirac;
pub use measures::lebesgue::LebesgueMeasure;
pub use traits::DotProduct;
