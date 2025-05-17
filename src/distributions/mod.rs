//! Probability distributions module.
//!
//! This module contains various probability distributions, organized by:
//! - Continuous distributions (using Lebesgue measure)
//! - Discrete distributions (using counting measure)

pub mod continuous;
pub mod discrete;

// Re-export common distributions
pub use continuous::Normal;
