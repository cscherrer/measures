//! A library for measure theory and probability distributions.
//!
//! This crate provides a type-safe framework for working with measure theory
//! and probability distributions, with a focus on:
//!
//! 1. **Type-safe measure theory**: Clear separation between measures and their densities
//! 2. **Efficient log-density computation**: Optimized for numerical stability and performance  
//! 3. **Generic numeric types**: Support for f64, f32, dual numbers (autodiff), etc.
//! 4. **Automatic differentiation ready**: Same log-density works with dual numbers
//! 5. **Exponential family support**: Specialized implementations for exponential families
//! 6. **Symbolic computation**: General mathematical expression system with JIT compilation
//! 7. **Bayesian inference**: Specialized tools for Bayesian statistical modeling
//! 8. **Measure combinators**: Compositional framework for building complex measures
//!
//! # Quick Start
//!
//! ```rust
//! use measures::{Normal, Cauchy, LogDensityBuilder};
//!
//! let normal = Normal::new(0.0, 1.0);
//! let cauchy = Cauchy::new(0.0, 1.0);
//! let x = 0.5;
//!
//! // Both exponential families and non-exponential families work seamlessly
//! let normal_density: f64 = normal.log_density().at(&x);
//! let cauchy_density: f64 = cauchy.log_density().at(&x);
//!
//! // Compute relative densities
//! let relative_density: f64 = normal.log_density().wrt(cauchy).at(&x);
//! ```
//!
//! # Measure Combinators
//!
//! Build complex measures from simpler ones using combinators:
//!
//! ```rust
//! use measures::{Normal, Poisson, LogDensityBuilder};
//! use measures::measures::combinators::{
//!     product::ProductMeasureExt,
//!     superposition::MixtureExt,
//! };
//! use measures::mixture;
//!
//! // Product measures for independence
//! let normal = Normal::new(0.0, 1.0);
//! let poisson = Poisson::new(2.0);
//! let joint = normal.clone().product(poisson);
//! let joint_density: f64 = joint.log_density().at(&(0.5, 3u64));
//!
//! // Pushforward measures for transformations (temporarily disabled due to trait complexity)
//! // let (forward, inverse, log_jacobian) = exp_transform();
//! // let log_normal = normal.pushforward(forward, inverse, log_jacobian);
//! // let log_normal_density: f64 = log_normal.log_density().at(&1.0);
//!
//! // Mixture measures
//! let component1 = Normal::new(-1.0, 1.0);
//! let component2 = Normal::new(1.0, 1.0);
//! let mixture = mixture![(0.3, component1), (0.7, component2)];
//! let mixture_density: f64 = mixture.log_density().at(&0.0);
//! ```
//!
//! # Symbolic Computation and JIT
//!
//! ## JIT Compilation Example
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use symbolic_math::{Expr, jit::GeneralJITCompiler};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a mathematical expression: x^2 + 2*x + 1
//! let expr = Expr::Add(
//!     Box::new(Expr::Add(
//!         Box::new(Expr::Pow(
//!             Box::new(Expr::Var("x".to_string())),
//!             Box::new(Expr::Const(2.0))
//!         )),
//!         Box::new(Expr::Mul(
//!             Box::new(Expr::Const(2.0)),
//!             Box::new(Expr::Var("x".to_string()))
//!         ))
//!     )),
//!     Box::new(Expr::Const(1.0))
//! );
//!
//! let compiler = GeneralJITCompiler::new()?;
//! let jit_function = compiler.compile_expression(
//!     &expr,
//!     &["x".to_string()], // data variables
//!     &[], // parameter variables  
//!     &HashMap::new(), // constants
//! )?;
//!
//! // Use the JIT-compiled function
//! let result = jit_function.call_single(3.0); // (3^2 + 2*3 + 1) = 16
//! assert!((result - 16.0).abs() < 1e-10);
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! # Bayesian Modeling
//!
//! ```rust
//! # #[cfg(feature = "symbolic")]
//! # {
//! use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};
//!
//! // Build Bayesian model expressions
//! let likelihood = normal_likelihood("x", "mu", "sigma");
//! let prior = normal_prior("mu", 0.0, 1.0);
//! let posterior = posterior_log_density(likelihood, prior);
//! # }
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

// Re-export subcrates
pub mod measures {
    pub use measures_combinators::*;

    pub mod primitive {
        pub use measures_combinators::{CountingMeasure, LebesgueMeasure};

        // Re-export individual modules for compatibility
        pub mod counting {
            pub use measures_combinators::CountingMeasure;
        }
        pub mod lebesgue {
            pub use measures_combinators::LebesgueMeasure;
        }
    }

    pub mod derived {
        pub use measures_combinators::{Dirac, FactorialMeasure, WeightedMeasure};

        // Re-export individual modules for compatibility
        pub mod factorial {
            pub use measures_combinators::FactorialMeasure;
        }
    }

    pub mod combinators {
        pub use measures_combinators::measures::combinators::*;
    }
}

// Re-export distributions with proper module structure
pub mod distributions {
    pub use measures_distributions::*;

    pub mod continuous {
        pub use measures_distributions::{
            Beta, Cauchy, ChiSquared, Exponential, Gamma, Normal, StdNormal, StudentT,
        };

        // Re-export individual modules for compatibility
        pub mod beta {
            pub use measures_distributions::Beta;
        }
        pub mod cauchy {
            pub use measures_distributions::Cauchy;
        }
        pub mod chi_squared {
            pub use measures_distributions::ChiSquared;
        }
        pub mod exponential {
            pub use measures_distributions::Exponential;
        }
        pub mod gamma {
            pub use measures_distributions::Gamma;
        }
        pub mod normal {
            pub use measures_distributions::Normal;
        }
        pub mod student_t {
            pub use measures_distributions::StudentT;
        }
    }

    pub mod discrete {
        pub use measures_distributions::{
            Bernoulli, Binomial, Categorical, Geometric, NegativeBinomial, Poisson,
        };

        // Re-export individual modules for compatibility
        pub mod bernoulli {
            pub use measures_distributions::Bernoulli;
        }
        pub mod binomial {
            pub use measures_distributions::Binomial;
        }
        pub mod categorical {
            pub use measures_distributions::Categorical;
        }
        pub mod geometric {
            pub use measures_distributions::Geometric;
        }
        pub mod negative_binomial {
            pub use measures_distributions::NegativeBinomial;
        }
        pub mod poisson {
            pub use measures_distributions::Poisson;
        }
    }
}

// Re-export exponential family with proper module structure
pub mod exponential_family {
    pub use measures_exponential_family::*;

    pub mod traits {
        pub use measures_exponential_family::ExponentialFamily;
    }

    #[cfg(feature = "jit")]
    pub mod jit {
        pub use measures_exponential_family::exponential_family::jit::*;
    }
}

// Bayesian inference and modeling (optional)
#[cfg(feature = "measures-bayesian")]
pub use measures_bayesian as bayesian;

// Re-export core traits and types
pub use measures_core::{
    DotProduct, False, HasLogDensity, HasLogDensityDecomposition, LogDensity, LogDensityBuilder,
    LogDensityTrait, Measure, MeasureMarker, PrimitiveMeasure, True, TypeLevelBool, float_constant,
    safe_convert,
};

// Re-export commonly used distributions
pub use measures_distributions::{
    Bernoulli, Beta, Binomial, Categorical, Cauchy, ChiSquared, Exponential, Gamma, Geometric,
    NegativeBinomial, Normal, Poisson, StdNormal, StudentT,
};

// Re-export exponential family functionality
pub use measures_exponential_family::{ExponentialFamily, IIDExtension};

// Re-export JIT functionality
#[cfg(feature = "jit")]
pub use measures_exponential_family::{CustomJITOptimizer, JITError, JITFunction, JITOptimizer};

#[cfg(feature = "jit")]
pub use measures_exponential_family::exponential_family::jit::ZeroOverheadOptimizer;

// Re-export JIT module for direct access
#[cfg(feature = "jit")]
pub use measures_exponential_family::exponential_family::jit;

// Re-export measure combinators
pub use measures_combinators::{
    MixtureExt, MixtureMeasure, ProductMeasure, ProductMeasureExt, PushforwardExt,
    PushforwardMeasure,
};

// Re-export symbolic computation types
#[cfg(feature = "symbolic")]
pub use symbolic_math::Expr;

#[cfg(feature = "jit")]
pub use symbolic_math::jit::{GeneralJITCompiler, GeneralJITFunction};

// Statistics module (local)
pub mod statistics;

// Re-export mixture macro
pub use measures_combinators::mixture;

// Re-export utility functions
pub use measures_core::{log_density_at, log_density_batch, safe_convert_or, safe_float_convert};

/// Convenience module for final tagless exponential family operations
pub mod final_tagless {
    pub use super::exponential_family::final_tagless::*;
    pub use symbolic_math::final_tagless::{DirectEval, MathExpr, PrettyPrint};

    // JITEval is only available with the jit feature
    #[cfg(feature = "jit")]
    pub use symbolic_math::final_tagless::JITEval;
}
