//! Exponential family distributions and optimization.
//!
//! This module provides the core infrastructure for working with exponential family
//! distributions, including:
//!
//! - Core traits for exponential families
//! - JIT compilation for runtime optimization
//! - Automatic derivation of optimized implementations
//! - IID (independent and identically distributed) extensions

pub mod iid;
pub mod implementations;
pub mod traits;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "jit")]
pub mod auto_jit;

// Re-export core traits
pub use traits::{
    ExponentialFamily, ExponentialFamilyMeasure, SumSufficientStats, compute_exp_fam_log_density,
    compute_iid_exp_fam_log_density,
};

// Re-export IID functionality
pub use iid::{IID, IIDExtension};

// Re-export implementations
pub use implementations::ExpFam;

// Re-export JIT compilation (if enabled) - now using symbolic-math crate
#[cfg(feature = "jit")]
pub use jit::{CompilationStats, CustomJITOptimizer, JITError, JITFunction, JITOptimizer};

#[cfg(feature = "jit")]
pub use auto_jit::{AutoJITExt, AutoJITOptimizer, AutoJITPattern, AutoJITRegistry};

// Re-export the auto_jit_impl macro
#[cfg(feature = "jit")]
pub use crate::auto_jit_impl;
