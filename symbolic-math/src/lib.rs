//! Symbolic Mathematics and JIT Compilation
//!
//! This crate provides a general-purpose symbolic representation system for mathematical
//! expressions and Just-In-Time compilation. It's designed to be domain-agnostic and can
//! be used for any mathematical computation, not just probability distributions.
//!
//! ## Core Components
//!
//! - **`expr`**: Symbolic expression representation (`Expr` enum)
//! - **`jit`**: JIT compilation using Cranelift for native performance
//!
//! ## Features
//!
//! - **General Mathematical Expressions**: Support for arithmetic, transcendental functions,
//!   and complex mathematical operations
//! - **Variable Substitution**: Flexible variable mapping for different use cases
//! - **JIT Compilation**: Convert expressions to native machine code
//! - **Multiple Signatures**: Support for functions with different input/output types
//! - **Performance Optimization**: Native speed execution with zero-overhead abstractions
//!
//! ## Usage Examples
//!
//! ### Basic Expression Building
//! ```rust
//! use symbolic_math::Expr;
//!
//! // Build expression: x² + 2x + 1
//! let x = Expr::Var("x".to_string());
//! let x_squared = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
//! let two_x = Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x));
//! let expr = Expr::Add(
//!     Box::new(Expr::Add(Box::new(x_squared), Box::new(two_x))),
//!     Box::new(Expr::Const(1.0))
//! );
//! ```
//!
//! ### JIT Compilation Example
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use symbolic_math::{Expr, GeneralJITCompiler};
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build a mathematical expression: sin(x^2) + cos(x)
//! let expr = Expr::Add(
//!     Box::new(Expr::Sin(Box::new(Expr::Pow(
//!         Box::new(Expr::Var("x".to_string())),
//!         Box::new(Expr::Const(2.0))
//!     )))),
//!     Box::new(Expr::Cos(Box::new(Expr::Var("x".to_string()))))
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
//! let result = jit_function.call_single(1.0);
//! # Ok(())
//! # }
//! # }
//! ```

#![warn(missing_docs)]

pub mod expr;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "optimization")]
pub mod optimization;

pub mod final_tagless;

// Re-export commonly used types
pub use expr::{
    CacheStats, ConstantPool, EvalError, Expr, SymbolicLogDensity, clear_caches, get_cache_stats,
};

#[cfg(feature = "jit")]
pub use jit::{
    CompilationStats, CustomSymbolicLogDensity, GeneralJITCompiler, GeneralJITFunction, JITError,
    JITSignature,
};

#[cfg(feature = "optimization")]
pub use optimization::EgglogOptimize;

// Re-export final tagless types
pub use final_tagless::{
    ContextualEval, DirectEval, ExprBuilder, FinalTaglessConversion, MathExpr, PrettyPrint,
    StatisticalExpr, dsl,
};

/// Ergonomic macros for building expressions with natural Rust syntax
pub mod macros {
    /// Create expressions using natural mathematical notation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use symbolic_math::expr;
    /// use symbolic_math::Expr;
    ///
    /// // Variables
    /// let x = expr!(x);
    /// let y = expr!(y);
    ///
    /// // Constants
    /// let two = expr!(2.0);
    /// let pi = expr!(3.14159);
    ///
    /// // Simple arithmetic operations
    /// let sum = expr!(x + y);
    /// let product = expr!(2.0 * x);
    ///
    /// // Functions
    /// let log_expr = expr!(ln(x));
    /// let exp_expr = expr!(exp(x));
    /// let sqrt_expr = expr!(sqrt(x));
    /// let sin_expr = expr!(sin(x));
    /// let cos_expr = expr!(cos(y));
    /// ```
    #[macro_export]
    macro_rules! expr {
        // Variables (identifiers)
        ($var:ident) => {
            $crate::Expr::variable(stringify!($var))
        };

        // Constants (literals)
        ($val:literal) => {
            $crate::Expr::constant($val as f64)
        };

        // Parentheses
        (($e:tt)) => {
            expr!($e)
        };

        // Addition
        ($left:tt + $right:tt) => {
            expr!($left) + expr!($right)
        };

        // Subtraction
        ($left:tt - $right:tt) => {
            expr!($left) - expr!($right)
        };

        // Multiplication
        ($left:tt * $right:tt) => {
            expr!($left) * expr!($right)
        };

        // Division
        ($left:tt / $right:tt) => {
            expr!($left) / expr!($right)
        };

        // Power (using ^ operator)
        ($base:tt ^ $exp:tt) => {
            $crate::Expr::pow(expr!($base), expr!($exp))
        };

        // Natural logarithm
        (ln($e:tt)) => {
            $crate::Expr::ln(expr!($e))
        };

        // Exponential
        (exp($e:tt)) => {
            $crate::Expr::exp(expr!($e))
        };

        // Square root
        (sqrt($e:tt)) => {
            $crate::Expr::sqrt(expr!($e))
        };

        // Sine
        (sin($e:tt)) => {
            $crate::Expr::sin(expr!($e))
        };

        // Cosine
        (cos($e:tt)) => {
            $crate::Expr::cos(expr!($e))
        };

        // Negation
        (-$e:tt) => {
            -expr!($e)
        };
    }

    /// Create a variable expression
    #[macro_export]
    macro_rules! var {
        ($name:expr) => {
            $crate::Expr::variable($name)
        };
    }

    /// Create a constant expression
    #[macro_export]
    macro_rules! const_expr {
        ($val:expr) => {
            $crate::Expr::constant($val)
        };
    }
}

/// Enhanced builder functions for common mathematical patterns
pub mod builders {
    use crate::Expr;

    /// Build a normal log-PDF expression: -0.5 * ln(2π) - ln(σ) - 0.5 * (x - μ)² / σ²
    pub fn normal_log_pdf(x: impl Into<Expr>, mu: impl Into<Expr>, sigma: impl Into<Expr>) -> Expr {
        let x = x.into();
        let mu = mu.into();
        let sigma = sigma.into();

        let two_pi = Expr::constant(2.0 * std::f64::consts::PI);
        let half = Expr::constant(0.5);

        let normalization = -(half.clone() * Expr::ln(two_pi) + Expr::ln(sigma.clone()));
        let quadratic =
            -half * Expr::pow(x - mu, Expr::constant(2.0)) / Expr::pow(sigma, Expr::constant(2.0));

        normalization + quadratic
    }

    /// Build a polynomial expression: a₀ + a₁x + a₂x² + ... + aₙxⁿ
    pub fn polynomial(x: impl Into<Expr>, coefficients: &[f64]) -> Expr {
        let x = x.into();
        let mut result = Expr::constant(0.0);

        for (i, &coeff) in coefficients.iter().enumerate() {
            let term = if i == 0 {
                Expr::constant(coeff)
            } else if i == 1 {
                Expr::constant(coeff) * x.clone()
            } else {
                Expr::constant(coeff) * Expr::pow(x.clone(), Expr::constant(i as f64))
            };
            result = result + term;
        }

        result
    }

    /// Build a Gaussian kernel: exp(-0.5 * (x - μ)² / σ²)
    pub fn gaussian_kernel(
        x: impl Into<Expr>,
        mu: impl Into<Expr>,
        sigma: impl Into<Expr>,
    ) -> Expr {
        let x = x.into();
        let mu = mu.into();
        let sigma = sigma.into();

        let half = Expr::constant(0.5);
        let exponent =
            -half * Expr::pow(x - mu, Expr::constant(2.0)) / Expr::pow(sigma, Expr::constant(2.0));
        Expr::exp(exponent)
    }

    /// Build a logistic function: 1 / (1 + exp(-x))
    pub fn logistic(x: impl Into<Expr>) -> Expr {
        let x = x.into();
        Expr::constant(1.0) / (Expr::constant(1.0) + Expr::exp(-x))
    }

    /// Build a softplus function: ln(1 + exp(x))
    pub fn softplus(x: impl Into<Expr>) -> Expr {
        let x = x.into();
        Expr::ln(Expr::constant(1.0) + Expr::exp(x))
    }
}

/// Display utilities for expressions
pub mod display {
    use crate::Expr;
    use std::fmt;

    /// Pretty-print wrapper for expressions
    pub struct PrettyExpr<'a>(pub &'a Expr);

    impl fmt::Display for PrettyExpr<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    /// Generate LaTeX representation of an expression
    #[must_use]
    pub fn latex(expr: &Expr) -> String {
        format!("${expr}$")
    }

    /// Generate Python code representation of an expression
    #[must_use]
    pub fn python(expr: &Expr) -> String {
        format!("lambda vars: {expr}")
    }

    /// Generate a mathematical equation string
    #[must_use]
    pub fn equation(lhs: &str, expr: &Expr) -> String {
        format!("{lhs} = {expr}")
    }
}
