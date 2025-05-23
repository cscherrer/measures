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
//!
//! The framework supports exponential families over different spaces:
//! - The space X where the random variable lives (could be ints, vectors, etc.)
//! - The field F for numerical computations (always some Float type)

pub mod array_utils;
pub mod cache_trait;
pub mod generic_cache;
pub mod iid;
pub mod implementations;
pub mod traits;

// Re-export key types
pub use cache_trait::ExponentialFamilyCache;
pub use generic_cache::GenericExpFamCache;
pub use iid::{IID, IIDExtension};
pub use implementations::{ExpFam, ExponentialFamilyDensity};
pub use traits::{
    ExponentialFamily, ExponentialFamilyMeasure, GenericExpFamImpl, PrecomputeCache,
    SumSufficientStats, compute_exp_fam_log_density, compute_iid_exp_fam_log_density,
};
