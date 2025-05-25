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
//! use measures::{Normal, LogDensityBuilder};
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
//!
//! // Same log-density, different numeric types (autodiff ready!)
//! let normal_f32 = Normal::new(0.0_f32, 1.0_f32);
//! let f32_x = 0.5_f32;
//! let f32_result: f32 = normal_f32.log_density().at(&f32_x);
//! // let dual_result: Dual64 = normal.log_density().at(&dual_x);  // With autodiff library
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

// Core abstractions
pub mod core;
pub mod distributions;
pub mod exponential_family;
pub mod measures;
pub mod statistics;
pub mod traits;

// Re-export key types for convenient access
pub use core::{
    DotProduct, EvaluateAt, HasLogDensity, LogDensity, LogDensityTrait, Measure, PrimitiveMeasure,
};
pub use distributions::continuous::beta::Beta;
pub use distributions::continuous::cauchy::Cauchy;
pub use distributions::continuous::chi_squared::ChiSquared;
pub use distributions::continuous::exponential::Exponential;
pub use distributions::continuous::gamma::Gamma;
pub use distributions::continuous::normal::Normal;
pub use distributions::continuous::stdnormal::StdNormal;
pub use distributions::discrete::bernoulli::Bernoulli;
pub use distributions::discrete::binomial::Binomial;
pub use distributions::discrete::categorical::Categorical;
pub use distributions::discrete::geometric::Geometric;
pub use distributions::discrete::negative_binomial::NegativeBinomial;
pub use distributions::discrete::poisson::Poisson;
pub use exponential_family::IIDExtension;

// Re-export core traits that users need to import
pub use core::LogDensityBuilder;
