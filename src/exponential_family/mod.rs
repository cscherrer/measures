//! Exponential family framework for probability distributions.
//!
//! This module defines traits and utilities for working with exponential family distributions,
//! which have the form:
//!
//! p(x|θ) = h(x) exp(η(θ)·T(x) - A(η(θ)))
//!
//! where:
//! - η(θ) are the natural parameters
//! - T(x) are the sufficient statistics
//! - A(η) is the log-partition function
//! - h(x) is the carrier measure

pub mod implementations;
pub mod traits;

// Re-export key types
pub use implementations::{
    ExponentialFamilyDensity, compute_exp_fam_log_density, exp_fam_log_density,
};
pub use traits::{DotProduct, ExpFamDensity, ExponentialFamily};
