//! Exponential family distributions and optimization.
//!
//! This module provides the core functionality for working with exponential family
//! distributions, including traits, implementations, and optimization features.

pub mod array_utils;
pub mod iid;
pub mod implementations;
pub mod traits;

// Final tagless approach for zero-cost symbolic computation
pub mod final_tagless;

// JIT compilation features
#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "jit")]
pub mod auto_jit;

// Re-export key items for convenience
pub use iid::*;
pub use implementations::*;
pub use traits::*;

// Re-export final tagless functionality
pub use final_tagless::{ExpFamEval, ExponentialFamilyExpr, patterns};

// Re-export JIT functionality when available
#[cfg(feature = "jit")]
pub use jit::*;

#[cfg(feature = "jit")]
pub use auto_jit::*;
