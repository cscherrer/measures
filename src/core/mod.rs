//! Core abstractions for measures and densities.
//!
//! This module provides the fundamental types and traits for working with
//! measure theory concepts in a type-safe way.

pub mod density;
pub mod measure;
pub mod types;

// Re-export key types for convenient use
pub use density::{LogDensity, LogDensityWithMethod};
pub use measure::{HasDensity, Measure, MeasureMarker, PrimitiveMeasure};
pub use types::{
    Default, ExponentialFamily, False, LogDensityMethod, Specialized, True, TypeLevelBool,
};
