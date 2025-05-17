//! Discrete probability distributions.
//!
//! This module provides discrete probability distributions
//! that use counting measure as their base measure.

pub mod poisson;

// Re-export common distributions
pub use poisson::Poisson;
