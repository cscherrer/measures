//! Probability distributions module.
//!
//! This module contains various probability distributions, organized by:
//! - Continuous distributions (using Lebesgue measure)
//! - Discrete distributions (using counting measure)
//! - Multivariate distributions (using Lebesgue measure on vector spaces)

pub mod continuous;
pub mod discrete;
pub mod multivariate;

// Re-export common distributions
pub use continuous::Normal;
pub use continuous::StdNormal;
pub use discrete::poisson::Poisson;
pub use multivariate::multinormal::MultivariateNormal;
pub use nalgebra::{DMatrix, DVector};
