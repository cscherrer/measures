//! Fundamental measures module.
//!
//! This module contains the basic measures used as building blocks
//! for more complex probability distributions:
//! - Lebesgue measure for continuous spaces
//! - Counting measure for discrete spaces
//! - Dirac measure for point masses

pub mod counting;
pub mod dirac;
pub mod lebesgue;

// Re-export for convenience
pub use counting::CountingMeasure;
pub use dirac::Dirac;
pub use lebesgue::LebesgueMeasure;
