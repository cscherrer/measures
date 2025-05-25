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
//! 6. **Symbolic computation**: General mathematical expression system with JIT compilation
//! 7. **Bayesian inference**: Specialized tools for Bayesian statistical modeling
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
//!
//! # Symbolic Computation and JIT
//!
//! ## JIT Compilation Example
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use measures::symbolic_ir::{Expr, GeneralJITCompiler};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a mathematical expression: x^2 + 2*x + 1
//! let expr = Expr::Add(
//!     Box::new(Expr::Add(
//!         Box::new(Expr::Pow(
//!             Box::new(Expr::Var("x".to_string())),
//!             Box::new(Expr::Const(2.0))
//!         )),
//!         Box::new(Expr::Mul(
//!             Box::new(Expr::Const(2.0)),
//!             Box::new(Expr::Var("x".to_string()))
//!         ))
//!     )),
//!     Box::new(Expr::Const(1.0))
//! );
//!
//! let compiler = GeneralJITCompiler::new()?;
//! let jit_function = compiler.compile_expression(
//!     &expr,
//!     &["x".to_string()], // data variables
//!     &[], // parameter variables  
//!     &HashMap::new(), // constants
//! )?;
//!
//! // Use the JIT-compiled function
//! let result = jit_function.call_single(3.0); // (3^2 + 2*3 + 1) = 16
//! assert!((result - 16.0).abs() < 1e-10);
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! # Bayesian Modeling
//!
//! ```rust
//! use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};
//!
//! // Build Bayesian model expressions
//! let likelihood = normal_likelihood("x", "mu", "sigma");
//! let prior = normal_prior("mu", 0.0, 1.0);
//! let posterior = posterior_log_density(likelihood, prior);
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

// General mathematical computation
pub mod symbolic_ir;

// Bayesian inference and modeling
pub mod bayesian;

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

// Re-export symbolic computation types
pub use symbolic_ir::Expr;

#[cfg(feature = "jit")]
pub use symbolic_ir::{GeneralJITCompiler, GeneralJITFunction, JITError, JITSignature};

// Re-export Bayesian functionality
#[cfg(feature = "jit")]
pub use bayesian::BayesianJITOptimizer;
