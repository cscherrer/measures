//! Probability distributions for the measures library.
//!
//! This module provides implementations of common probability distributions,
//! organized by type (continuous, discrete, multivariate).

pub mod continuous;
pub mod discrete;
pub mod multivariate;

// Final tagless approach for distributions
pub mod final_tagless;

// Re-export common distributions for convenience
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, Gamma, Normal, StdNormal,
};

pub use discrete::{
    Bernoulli, Binomial, Categorical, Geometric, NegativeBinomial, Poisson,
};

// Note: Multivariate distributions are not yet implemented
// pub use multivariate::{
//     MultivariateNormal, Dirichlet, Wishart,
// };

// Re-export final tagless functionality
pub use final_tagless::{
    DistributionExpr, DistributionEval, DistributionMathExpr, patterns,
};

// pub use multivariate::multinormal::MultivariateNormal;
// pub use nalgebra::{DMatrix, DVector};
