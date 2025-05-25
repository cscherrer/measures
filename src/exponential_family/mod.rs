//! Exponential family distributions and optimization.
//!
//! This module provides the core infrastructure for working with exponential family
//! distributions, including:
//!
//! - Core traits for exponential families
//! - JIT compilation for runtime optimization
//! - Automatic derivation of optimized implementations
//! - IID (independent and identically distributed) extensions

pub mod traits;
pub mod iid;
pub mod implementations;

#[cfg(feature = "jit")]
pub mod symbolic_ir;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "jit")]
pub mod auto_jit;

#[cfg(feature = "jit")]
pub mod egglog_optimizer;

// Re-export core traits
pub use traits::{
    compute_exp_fam_log_density, compute_iid_exp_fam_log_density, ExponentialFamily,
    ExponentialFamilyMeasure, SumSufficientStats,
};

// Re-export IID functionality
pub use iid::{IIDExtension, IID};

// Re-export implementations
pub use implementations::ExpFam;

// Re-export JIT compilation (if enabled)
#[cfg(feature = "jit")]
pub use symbolic_ir::{EvalError, Expr, SymbolicLogDensity as CustomSymbolicLogDensity};

#[cfg(feature = "jit")]
pub use jit::{CompilationStats, JITError, JITFunction, JITOptimizer, CustomJITOptimizer};

#[cfg(feature = "jit")]
pub use auto_jit::{AutoJITExt, AutoJITOptimizer, AutoJITPattern, AutoJITRegistry};

// Re-export the auto_jit_impl macro
#[cfg(feature = "jit")]
pub use crate::auto_jit_impl;
