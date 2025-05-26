//! Measure combinators for building complex measures from simpler ones.
//!
//! This crate provides combinators that allow you to build sophisticated
//! probability measures by composing simpler measures. This follows the
//! compositional approach of measure theory, enabling:
//!
//! - **Product measures**: For modeling independence
//! - **Pushforward measures**: For transformations and change of variables
//! - **Mixture measures**: For superposition and weighted combinations
//! - **Primitive measures**: Basic building blocks (Lebesgue, Counting)
//! - **Derived measures**: Specialized constructions (Dirac, Weighted)
//!
//! # Quick Start
//!
//! ```rust
//! use measures_combinators::{ProductMeasureExt, PushforwardExt, MixtureExt};
//! use measures_core::{LogDensityBuilder, HasLogDensity};
//!
//! # struct Normal { mean: f64, std: f64 }
//! # impl Normal { fn new(mean: f64, std: f64) -> Self { Normal { mean, std } } }
//! # impl measures_core::PrimitiveMeasure for Normal { type Domain = f64; }
//! # impl measures_core::HasLogDensity<f64> for Normal {
//! #     fn log_density(&self) -> measures_core::LogDensity<Self, f64> {
//! #         measures_core::LogDensity::new(self)
//! #     }
//! # }
//! # impl measures_core::LogDensityTrait<f64> for Normal {
//! #     fn log_density_at(&self, _x: &f64) -> f64 { 0.0 }
//! # }
//! # impl Clone for Normal { fn clone(&self) -> Self { Normal { mean: self.mean, std: self.std } } }
//! # struct Poisson { rate: f64 }
//! # impl Poisson { fn new(rate: f64) -> Self { Poisson { rate } } }
//! # impl measures_core::PrimitiveMeasure for Poisson { type Domain = u64; }
//! # impl measures_core::HasLogDensity<u64> for Poisson {
//! #     fn log_density(&self) -> measures_core::LogDensity<Self, u64> {
//! #         measures_core::LogDensity::new(self)
//! #     }
//! # }
//! # impl measures_core::LogDensityTrait<u64> for Poisson {
//! #     fn log_density_at(&self, _x: &u64) -> f64 { 0.0 }
//! # }
//!
//! // Product measures for independence
//! let normal = Normal::new(0.0, 1.0);
//! let poisson = Poisson::new(2.0);
//! let joint = normal.clone().product(poisson);
//! let joint_density: f64 = joint.log_density().at(&(0.5, 3u64));
//!
//! // Mixture measures
//! let component1 = Normal::new(-1.0, 1.0);
//! let component2 = Normal::new(1.0, 1.0);
//! let weights = vec![0.3, 0.7];
//! let components = vec![component1, component2];
//! let mixture = components.mixture(weights);
//! let mixture_density: f64 = mixture.log_density().at(&0.0);
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

pub mod measures;

// Re-export all measure types for convenience
pub use measures::derived::{Dirac, FactorialMeasure, WeightedMeasure};
pub use measures::primitive::{CountingMeasure, LebesgueMeasure};

// Re-export measure combinators
pub use measures::combinators::product::{ProductMeasure, ProductMeasureExt};
pub use measures::combinators::pushforward::{PushforwardExt, PushforwardMeasure};
pub use measures::combinators::superposition::{MixtureExt, MixtureMeasure};

// Re-export core traits for convenience
pub use measures_core::{
    HasLogDensity, LogDensityBuilder, LogDensityTrait, Measure, PrimitiveMeasure,
};
