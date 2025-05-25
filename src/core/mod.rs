//! Core traits and types for the measures library.

/// Log-density computation traits and implementations
pub mod density;
/// Core measure trait and implementations
pub mod measure;
/// Type-level programming utilities
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
