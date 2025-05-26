//! Core measure theory abstractions and traits.
//!
//! This crate provides the fundamental building blocks for measure theory and
//! probability distributions, including:
//!
//! 1. **Core measure abstractions**: `Measure`, `PrimitiveMeasure`, and related traits
//! 2. **Log-density computation**: `LogDensity`, `HasLogDensity`, and evaluation traits
//! 3. **Log-density decomposition**: Framework for structured log-density computation
//! 4. **Type-level programming**: Utilities for compile-time specialization
//! 5. **Computational traits**: Core traits like `DotProduct` for numerical operations
//! 6. **Automatic differentiation**: Optional support for autodiff with dual numbers
//!
//! This crate serves as the foundation for higher-level probability and statistics
//! libraries, providing type-safe and efficient abstractions for measure theory.
//!
//! # Quick Start
//!
//! ```rust
//! use measures_core::{Measure, LogDensityBuilder, HasLogDensity};
//!
//! // Define a custom measure (this would typically be done by distribution crates)
//! # struct MyMeasure;
//! # impl measures_core::PrimitiveMeasure for MyMeasure {
//! #     type Domain = f64;
//! # }
//! # impl measures_core::HasLogDensity<f64> for MyMeasure {
//! #     fn log_density(&self) -> measures_core::LogDensity<Self, f64> {
//! #         measures_core::LogDensity::new(self)
//! #     }
//! # }
//! # impl measures_core::LogDensityTrait<f64> for MyMeasure {
//! #     fn log_density_at(&self, _x: &f64) -> f64 { 0.0 }
//! # }
//!
//! let measure = MyMeasure;
//! let x = 1.0;
//!
//! // Use the log-density builder pattern
//! let density_value: f64 = measure.log_density().at(&x);
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

// Core abstractions
pub mod core;
pub mod primitive;
pub mod traits;

// Re-export commonly used submodules for easier access
pub use core::log_density_decomposition;
pub use core::types;
pub use core::utils;

// Re-export primitive measures
pub use primitive::{CountingMeasure, LebesgueMeasure};

// Re-export commonly used utility functions
pub use core::utils::{float_constant, safe_convert, safe_convert_or, safe_float_convert};

// Re-export density helper functions
pub use core::density::{log_density_at, log_density_batch};

// Re-export key types for convenient access
pub use core::{
    DecompositionBuilder, EvaluateAt, HasLogDensity, HasLogDensityDecomposition, LogDensity,
    LogDensityBuilder, LogDensityDecomposition, LogDensityTrait, Measure, MeasureMarker,
    PrimitiveMeasure,
};

// Re-export core computational traits
pub use traits::DotProduct;

// Re-export type-level programming utilities
pub use core::{
    Default, ExponentialFamily, False, LogDensityMethod, Specialized, True, TypeLevelBool,
};

// Re-export density-related types
pub use core::{CachedLogDensity, DensityMeasure, LogDensityCaching, SharedRootMeasure};

// Re-export autodiff support when available
#[cfg(feature = "autodiff")]
pub use core::{AutoDiffMeasure, LogDensityAD};
