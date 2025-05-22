//! Fundamental measures module.
//!
//! This module contains the basic measures used as building blocks
//! for more complex probability distributions:
//! - Lebesgue measure for continuous spaces
//! - Counting measure for discrete spaces
//! - Dirac measure for point masses
//! - Weighted measure for measures with weight functions

pub mod counting;
pub mod dirac;
pub mod lebesgue;
pub mod weighted;

// Re-export for convenience
pub use counting::CountingMeasure;
pub use dirac::Dirac;
pub use lebesgue::LebesgueMeasure;
pub use weighted::WeightedMeasure;
