//! Exponential family distributions framework.
//!
//! This module provides a comprehensive framework for working with exponential family
//! distributions, which have the canonical form:
//!
//! p(x|θ) = h(x) exp(η(θ)·T(x) - A(η(θ)))
//!
//! where:
//! - η(θ) are the natural parameters  
//! - T(x) are the sufficient statistics
//! - A(η) is the log-partition function
//! - h(x) is the carrier measure
//!
//! ## Core Components
//!
//! - [`ExponentialFamily`]: The main trait defining exponential family structure
//! - [`compute_exp_fam_log_density`]: Efficient log-density computation
//! - [`compute_iid_exp_fam_log_density`]: Efficient IID log-density computation
//! - [`IID`]: Wrapper for independent and identically distributed samples
//!
//! ## Performance Optimization
//!
//! For high-performance scenarios, use symbolic optimization:
//! - [`symbolic`]: Symbolic optimization for runtime code generation
//! - Generate specialized functions with precomputed constants
//! - Achieve significant speedups over standard evaluation
//!
//! ## Mathematical Foundation
//!
//! The framework automatically handles the complete exponential family formula
//! including chain rule terms for non-trivial base measures. This eliminates
//! the need for manual overrides in most distributions.
//!
//! For IID samples, the framework uses the efficient formula:
//! log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)

pub mod iid;
pub mod traits;

#[cfg(feature = "symbolic")]
pub mod symbolic;

// Re-export core traits and functions
pub use traits::{
    ExponentialFamily, ExponentialFamilyMeasure, SumSufficientStats, compute_exp_fam_log_density,
    compute_iid_exp_fam_log_density,
};

pub use iid::{IID, IIDExtension};

#[cfg(feature = "symbolic")]
pub use symbolic::{SymbolicExtension, SymbolicOptimizer};
