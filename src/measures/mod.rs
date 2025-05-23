//! Fundamental measures module.
//!
//! This module contains measures organized by their mathematical nature:
//! - **Primitive measures**: Basic building blocks (Lebesgue, Counting)
//! - **Derived measures**: Constructed from primitives (Dirac, Weighted)
//!
//! Primitive measures serve as the foundation for probability distributions,
//! while derived measures provide specialized functionality.

pub mod derived;
pub mod primitive;

// Re-export all measures for convenience
pub use derived::{Dirac, WeightedMeasure};
pub use primitive::{CountingMeasure, LebesgueMeasure};
