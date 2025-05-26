//! Probability distributions for the measures library.
//!
//! This crate provides implementations of common probability distributions,
//! including:
//!
//! - **Continuous distributions**: Normal, Exponential, Gamma, Beta, etc.
//! - **Discrete distributions**: Bernoulli, Binomial, Poisson, etc.
//! - **Multivariate distributions**: (planned)
//! - **Final tagless approach**: Zero-cost symbolic computation with compile-time type safety
//! - **JIT compilation**: Runtime optimization for distribution computations
//!
//! # Quick Start
//!
//! ```rust
//! use measures_distributions::{Normal, distributions::final_tagless::*};
//! use measures_core::{HasLogDensity};
//!
//! // Traditional approach
//! let normal = Normal::new(0.0, 1.0);
//! let log_density = normal.log_density_wrt_root(&1.0);
//!
//! // Final tagless approach for ultimate performance
//! use symbolic_math::final_tagless::DirectEval;
//! let x = DirectEval::constant(1.0);
//! let mu = DirectEval::constant(0.0);
//! let sigma = DirectEval::constant(1.0);
//! let result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
//! ```
//!
//! # Final Tagless Approach
//!
//! For ultimate performance, use the final tagless approach:
//!
//! ```rust
//! use measures_distributions::distributions::final_tagless::*;
//! use symbolic_math::final_tagless::{DirectEval, JITEval};
//!
//! // Define normal log-density using final tagless
//! let x = DirectEval::constant(1.0);
//! let mu = DirectEval::constant(0.0);
//! let sigma = DirectEval::constant(1.0);
//! let result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
//!
//! // Compile to native code for ultimate performance (with JIT feature)
//! # #[cfg(feature = "jit")]
//! # {
//! let jit_expr = patterns::normal_log_density::<JITEval>(
//!     JITEval::var("x"),
//!     JITEval::var("mu"),
//!     JITEval::var("sigma")
//! );
//! let compiled = JITEval::compile_data_params(jit_expr, "x", &["mu".to_string(), "sigma".to_string()])?;
//! let result = compiled.call_data_params(1.0, &[0.0, 1.0]);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

pub mod distributions;

// Re-export core distributions for convenience
pub use distributions::{
    // Discrete distributions
    Bernoulli,
    // Continuous distributions
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    ChiSquared,
    DistributionEval,
    // Final tagless functionality
    DistributionExpr,
    DistributionMathExpr,
    Exponential,
    Gamma,
    Geometric,
    NegativeBinomial,
    Normal,
    Poisson,
    StdNormal,
    StudentT,
    patterns,
};

// Re-export core traits for convenience
pub use measures_core::{HasLogDensity, Measure, PrimitiveMeasure};

// Re-export exponential family traits when available
#[cfg(feature = "jit")]
pub use measures_exponential_family::{ExponentialFamily, IIDExtension};

/// Convenience module for final tagless distribution operations
pub mod final_tagless {
    pub use super::distributions::final_tagless::*;
    pub use symbolic_math::final_tagless::{DirectEval, MathExpr, PrettyPrint};

    // JITEval is only available with the jit feature
    #[cfg(feature = "jit")]
    pub use symbolic_math::final_tagless::JITEval;

    // Exponential family operations when available
    #[cfg(feature = "jit")]
    pub use measures_exponential_family::final_tagless::*;
}
