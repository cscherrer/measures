//! Continuous probability distributions.
//!
//! This module provides various continuous probability distributions,
//! all of which use Lebesgue measure as their base measure.

pub mod normal;
pub mod stdnormal;

// Re-export for convenience
pub use normal::Normal;
pub use stdnormal::StdNormal;
