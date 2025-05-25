//! Core abstractions for measures and densities.
//!
//! This module provides the fundamental types and traits for working with
//! measure theory concepts in a type-safe way.

pub mod density;
pub mod measure;
pub mod types;
pub mod utils;

// Re-export key types for convenient use
pub use density::{
    CachedLogDensity, DensityMeasure, EvaluateAt, HasLogDensity, LogDensity, LogDensityCaching,
    LogDensityTrait, SharedRootMeasure,
};
pub use measure::{LogDensityBuilder, Measure, MeasureMarker, PrimitiveMeasure};
pub use types::{
    Default, ExponentialFamily, False, LogDensityMethod, Specialized, True, TypeLevelBool,
};

// Re-export DotProduct as it's a fundamental computational trait
pub use crate::traits::DotProduct;
