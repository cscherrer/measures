//! Multivariate probability distributions.
//!
//! This module provides probability distributions over vector spaces,
//! typically using the Lebesgue measure on those spaces.

pub mod multinormal;

// Re-export common distributions
pub use multinormal::{Matrix, MultivariateNormal, Vector};
