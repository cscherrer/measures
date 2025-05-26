//! Probability distributions for the measures library.
//!
//! This crate provides a comprehensive collection of probability distributions
//! built on top of the measures-core framework. It includes:
//!
//! - **Continuous distributions**: Normal, Gamma, Beta, Cauchy, etc.
//! - **Discrete distributions**: Poisson, Binomial, Categorical, etc.
//! - **Multivariate distributions**: Multivariate Normal, etc.
//!
//! All distributions implement the core measure theory traits from measures-core,
//! providing type-safe and efficient log-density computation.
//!
//! # Quick Start
//!
//! ```rust
//! use measures_distributions::{Normal, Poisson};
//! use measures_core::{LogDensityBuilder, HasLogDensity};
//!
//! let normal = Normal::new(0.0, 1.0);
//! let poisson = Poisson::new(2.0);
//!
//! let normal_density: f64 = normal.log_density().at(&0.5);
//! let poisson_density: f64 = poisson.log_density().at(&3u64);
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

pub mod distributions;

// Re-export all distributions for convenience
pub use distributions::continuous::beta::Beta;
pub use distributions::continuous::cauchy::Cauchy;
pub use distributions::continuous::chi_squared::ChiSquared;
pub use distributions::continuous::exponential::Exponential;
pub use distributions::continuous::gamma::Gamma;
pub use distributions::continuous::normal::Normal;
pub use distributions::continuous::stdnormal::StdNormal;
pub use distributions::continuous::student_t::StudentT;

pub use distributions::discrete::bernoulli::Bernoulli;
pub use distributions::discrete::binomial::Binomial;
pub use distributions::discrete::categorical::Categorical;
pub use distributions::discrete::geometric::Geometric;
pub use distributions::discrete::negative_binomial::NegativeBinomial;
pub use distributions::discrete::poisson::Poisson;

// Re-export multivariate distributions when available
// pub use distributions::multivariate::multinormal::MultivariateNormal;

// Re-export core traits for convenience
pub use measures_core::{
    HasLogDensity, LogDensityBuilder, LogDensityTrait, Measure, PrimitiveMeasure,
};
