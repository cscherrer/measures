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

// Continuous distributions
pub use continuous::Beta;
pub use continuous::ChiSquared;
pub use continuous::Exponential;
pub use continuous::Gamma;
pub use continuous::Normal;
pub use continuous::StdNormal;

// Discrete distributions
pub use discrete::Bernoulli;
pub use discrete::Binomial;
pub use discrete::Categorical;
pub use discrete::Geometric;
pub use discrete::NegativeBinomial;
pub use discrete::Poisson;

// pub use multivariate::multinormal::MultivariateNormal;
// pub use nalgebra::{DMatrix, DVector};
