//! Derived measures constructed from primitive measures.
//!
//! These measures are built on top of primitive measures and provide
//! additional functionality like point masses, weighted measures, etc.

pub mod binomial_coefficient;
pub mod dirac;
pub mod factorial;
pub mod integral;
pub mod negative_binomial_coefficient;
pub mod weighted;

pub use binomial_coefficient::BinomialCoefficientMeasure;
pub use dirac::Dirac;
pub use factorial::FactorialMeasure;
pub use integral::IntegralMeasure;
pub use negative_binomial_coefficient::NegativeBinomialCoefficientMeasure;
pub use weighted::WeightedMeasure;
