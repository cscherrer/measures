//! Exponential family distributions and JIT optimization.
//!
//! This crate provides specialized support for exponential family distributions,
//! including:
//!
//! - **Core exponential family traits**: Type-safe exponential family abstractions
//! - **JIT compilation**: Runtime optimization for exponential family computations
//! - **IID extensions**: Efficient handling of independent and identically distributed data
//! - **Automatic optimization**: Automatic derivation of optimized implementations
//!
//! # Quick Start
//!
//! ```rust
//! use measures_exponential_family::{ExponentialFamily, IIDExtension};
//! use measures_core::{LogDensityBuilder, HasLogDensity};
//!
//! // Example with a distribution that implements ExponentialFamily
//! # struct MyExpFam;
//! # impl measures_core::PrimitiveMeasure for MyExpFam {
//! #     type Domain = f64;
//! # }
//! # impl measures_core::HasLogDensity<f64> for MyExpFam {
//! #     fn log_density(&self) -> measures_core::LogDensity<Self, f64> {
//! #         measures_core::LogDensity::new(self)
//! #     }
//! # }
//! # impl measures_core::LogDensityTrait<f64> for MyExpFam {
//! #     fn log_density_at(&self, _x: &f64) -> f64 { 0.0 }
//! # }
//! # impl ExponentialFamily<f64> for MyExpFam {
//! #     type SufficientStats = [f64; 2];
//! #     type NaturalParams = [f64; 2];
//! #     fn sufficient_stats(&self, _x: &f64) -> Self::SufficientStats { [0.0, 0.0] }
//! #     fn log_normalizer(&self, _params: &Self::NaturalParams) -> f64 { 0.0 }
//! #     fn natural_params(&self) -> Self::NaturalParams { [0.0, 0.0] }
//! # }
//!
//! let dist = MyExpFam;
//! let data = vec![1.0, 2.0, 3.0];
//!
//! // Use IID extension for efficient batch computation
//! let iid_dist = dist.iid(data.len());
//! let batch_density: f64 = iid_dist.log_density().at(&data);
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

pub mod exponential_family;

// Re-export core exponential family functionality
pub use exponential_family::traits::{
    ExponentialFamily, ExponentialFamilyMeasure, SumSufficientStats, compute_exp_fam_log_density,
    compute_iid_exp_fam_log_density,
};

// Re-export IID functionality
pub use exponential_family::iid::{IID, IIDExtension};

// Re-export implementations
pub use exponential_family::implementations::ExpFam;

// Re-export JIT functionality when available
#[cfg(feature = "jit")]
pub use exponential_family::jit::{
    CompilationStats, CustomJITOptimizer, JITError, JITFunction, JITOptimizer,
};

#[cfg(feature = "jit")]
pub use exponential_family::auto_jit::{
    AutoJITExt, AutoJITOptimizer, AutoJITPattern, AutoJITRegistry,
};

// Re-export core traits for convenience
pub use measures_core::{
    HasLogDensity, LogDensityBuilder, LogDensityTrait, Measure, PrimitiveMeasure,
};
