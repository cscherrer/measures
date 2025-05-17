//! Continuous probability distributions.
//!
//! This module provides various continuous probability distributions,
//! all of which use Lebesgue measure as their base measure.

pub mod normal;

// Re-export for convenience
pub use normal::Normal;
