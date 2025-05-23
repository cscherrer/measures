//! Derived measures constructed from primitive measures.
//!
//! These measures are built on top of primitive measures and provide
//! additional functionality like point masses, weighted measures, etc.

pub mod dirac;
pub mod factorial;
pub mod integral;
pub mod weighted;

pub use dirac::Dirac;
pub use factorial::FactorialMeasure;
pub use integral::IntegralMeasure;
pub use weighted::WeightedMeasure;
