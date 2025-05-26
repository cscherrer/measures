//! Discrete probability distributions.
//!
//! This module provides discrete probability distributions
//! that use counting measure as their base measure.

pub mod bernoulli;
pub mod binomial;
pub mod categorical;
pub mod geometric;
pub mod negative_binomial;
pub mod poisson;

// Re-export common distributions
pub use bernoulli::Bernoulli;
pub use binomial::Binomial;
pub use categorical::Categorical;
pub use geometric::Geometric;
pub use negative_binomial::NegativeBinomial;
pub use poisson::Poisson;
