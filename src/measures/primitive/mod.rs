//! Primitive measures that serve as building blocks.
//!
//! Primitive measures are the fundamental measures that cannot be decomposed
//! into simpler measures. They serve as the foundation for constructing
//! more complex measures and probability distributions.

pub mod counting;
pub mod lebesgue;

pub use counting::CountingMeasure;
pub use lebesgue::LebesgueMeasure; 