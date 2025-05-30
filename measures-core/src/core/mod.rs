//! Core traits and types for the measures library.

/// Log-density computation traits and implementations
pub mod density;
/// Log-density decomposition framework for structured computation
pub mod log_density_decomposition;
/// Core measure trait and implementations
pub mod measure;
/// Type-level programming utilities
pub mod types;
pub mod utils;

/// Automatic differentiation support (when autodiff feature is enabled)
#[cfg(feature = "autodiff")]
pub mod autodiff;

// Re-export key types for convenient use
pub use density::{
    CachedLogDensity, DensityMeasure, EvaluateAt, HasLogDensity, LogDensity, LogDensityCaching,
    LogDensityTrait, SharedRootMeasure,
};
pub use log_density_decomposition::{
    DecompositionBuilder, HasLogDensityDecomposition, LogDensityDecomposition,
};
pub use measure::{LogDensityBuilder, Measure, MeasureMarker, PrimitiveMeasure};
pub use types::{
    Default, ExponentialFamily, False, LogDensityMethod, Specialized, True, TypeLevelBool,
};

// Re-export DotProduct as it's a fundamental computational trait
pub use crate::traits::DotProduct;

// Re-export autodiff support when available
#[cfg(feature = "autodiff")]
pub use autodiff::{AutoDiffMeasure, LogDensityAD};
