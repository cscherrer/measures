//! Continuous probability distributions.
//!
//! This module provides various continuous probability distributions,
//! all of which use Lebesgue measure as their base measure.

pub mod beta;
pub mod cauchy;
pub mod chi_squared;
pub mod exponential;
pub mod gamma;
pub mod normal;
pub mod stdnormal;
pub mod student_t;

// Re-export for convenience
pub use beta::Beta;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use normal::Normal;
pub use stdnormal::StdNormal;
