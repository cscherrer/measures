//! Bayesian inference and modeling for the measures library.
//!
//! This crate provides tools for Bayesian statistical modeling and inference,
//! including:
//!
//! - **Expression building**: Construct Bayesian models using symbolic expressions
//! - **Posterior composition**: Combine likelihoods and priors
//! - **Hierarchical models**: Support for multi-level Bayesian models
//! - **JIT compilation**: High-performance Bayesian computations
//!
//! # Quick Start
//!
//! ```rust
//! # #[cfg(feature = "symbolic")]
//! # {
//! use measures_bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};
//!
//! // Build Bayesian model expressions
//! let likelihood = normal_likelihood("x", "mu", "sigma");
//! let prior = normal_prior("mu", 0.0, 1.0);
//! let posterior = posterior_log_density(likelihood, prior);
//! # }
//! ```
//!
//! # JIT Compilation
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use measures_bayesian::BayesianJITOptimizer;
//! use symbolic_math::Expr;
//!
//! // Build and compile Bayesian models for maximum performance
//! let optimizer = BayesianJITOptimizer::new();
//! # }
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

pub mod bayesian;

// Re-export Bayesian functionality
pub use bayesian::expressions;

// Re-export JIT functionality when available
#[cfg(feature = "jit")]
pub use bayesian::BayesianJITOptimizer;

// Re-export core traits for convenience
pub use measures_core::{
    HasLogDensity, LogDensityBuilder, LogDensityTrait, Measure, PrimitiveMeasure,
};
