//! Exponential family distributions and optimization.
//!
//! This module provides the core functionality for working with exponential family
//! distributions, including traits, implementations, and optimization features.

pub mod traits;
pub mod implementations;
pub mod iid;
pub mod array_utils;

// Final tagless approach for zero-cost symbolic computation
pub mod final_tagless;

// JIT compilation features
#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "jit")]
pub mod auto_jit;

// Re-export key items for convenience
pub use traits::*;
pub use implementations::*;
pub use iid::*;

// Re-export final tagless functionality
pub use final_tagless::{ExponentialFamilyExpr, ExpFamEval, patterns};

// Re-export JIT functionality when available
#[cfg(feature = "jit")]
pub use jit::*;

#[cfg(feature = "jit")]
pub use auto_jit::*;
