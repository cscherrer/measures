//! Derived measures constructed from primitive measures.
//!
//! These measures are built on top of primitive measures and provide
//! additional functionality like point masses, weighted measures, etc.

pub mod dirac;
pub mod weighted;

pub use dirac::Dirac;
pub use weighted::WeightedMeasure; 