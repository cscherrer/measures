//! Symbolic Intermediate Representation for Mathematical Expressions
//!
//! This module provides a general-purpose symbolic representation system for mathematical
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
//! use measures::symbolic_ir::expr::Expr;
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
//! use measures::symbolic_ir::{Expr, GeneralJITCompiler};
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

pub mod expr;

#[cfg(feature = "jit")]
pub mod jit;

// Re-export commonly used types
pub use expr::Expr;

#[cfg(feature = "jit")]
pub use jit::{
    CompilationStats, GeneralJITCompiler, GeneralJITFunction, JITError, JITSignature, JITType,
};

/// Ergonomic macros for building expressions with natural Rust syntax
pub mod macros {
    /// Create expressions using natural mathematical notation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use measures::symbolic_ir::macros::expr;
    /// use measures::symbolic_ir::Expr;
    ///
    /// // Variables
    /// let x = expr!(x);
    /// let y = expr!(y);
    ///
    /// // Constants
    /// let two = expr!(2.0);
    /// let pi = expr!(3.14159);
    ///
    /// // Arithmetic operations
    /// let sum = expr!(x + y);
    /// let product = expr!(2.0 * x);
    /// let quadratic = expr!(x * x + 2.0 * x + 1.0);
    ///
    /// // Functions
    /// let log_expr = expr!(ln(x));
    /// let exp_expr = expr!(exp(x));
    /// let sqrt_expr = expr!(sqrt(x));
    /// let trig = expr!(sin(x) + cos(y));
    ///
    /// // Complex expressions
    /// let normal_pdf = expr!(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma) - ln(sigma) - 0.5 * ln(2.0 * 3.14159));
    /// ```
    #[macro_export]
    macro_rules! expr {
        // Variables (identifiers)
        ($var:ident) => {
            $crate::symbolic_ir::Expr::variable(stringify!($var))
        };

        // Constants (literals)
        ($val:literal) => {
            $crate::symbolic_ir::Expr::constant($val as f64)
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
            $crate::symbolic_ir::Expr::pow(expr!($base), expr!($exp))
        };

        // Natural logarithm
        (ln($e:tt)) => {
            $crate::symbolic_ir::Expr::ln(expr!($e))
        };

        // Exponential
        (exp($e:tt)) => {
            $crate::symbolic_ir::Expr::exp(expr!($e))
        };

        // Square root
        (sqrt($e:tt)) => {
            $crate::symbolic_ir::Expr::sqrt(expr!($e))
        };

        // Sine
        (sin($e:tt)) => {
            $crate::symbolic_ir::Expr::sin(expr!($e))
        };

        // Cosine
        (cos($e:tt)) => {
            $crate::symbolic_ir::Expr::cos(expr!($e))
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
            $crate::symbolic_ir::Expr::variable($name)
        };
    }

    /// Create a constant expression
    #[macro_export]
    macro_rules! const_expr {
        ($val:expr) => {
            $crate::symbolic_ir::Expr::constant($val)
        };
    }
}

/// Enhanced builder functions for common mathematical patterns
pub mod builders {
    use super::Expr;

    /// Build a normal distribution log-density: -0.5 * ((x - μ) / σ)² - ln(σ√(2π))
    pub fn normal_log_pdf(x: impl Into<Expr>, mu: impl Into<Expr>, sigma: impl Into<Expr>) -> Expr {
        let x = x.into();
        let mu = mu.into();
        let sigma = sigma.into();

        let diff = x - mu;
        let standardized = diff.clone() / sigma.clone();
        let quadratic = standardized.clone() * standardized;
        let log_norm = Expr::ln(sigma) + Expr::constant(0.5 * (2.0 * std::f64::consts::PI).ln());

        Expr::constant(-0.5) * quadratic - log_norm
    }

    /// Build a polynomial: a₀ + a₁x + a₂x² + ... + aₙxⁿ
    pub fn polynomial(x: impl Into<Expr>, coefficients: &[f64]) -> Expr {
        let x = x.into();
        let mut result = Expr::constant(0.0);

        for (i, &coeff) in coefficients.iter().enumerate() {
            if coeff != 0.0 {
                let term = if i == 0 {
                    Expr::constant(coeff)
                } else if i == 1 {
                    Expr::constant(coeff) * x.clone()
                } else {
                    Expr::constant(coeff) * Expr::pow(x.clone(), Expr::constant(i as f64))
                };
                result = result + term;
            }
        }

        result
    }

    /// Build a Gaussian kernel: exp(-0.5 * ((x - μ) / σ)²)
    pub fn gaussian_kernel(
        x: impl Into<Expr>,
        mu: impl Into<Expr>,
        sigma: impl Into<Expr>,
    ) -> Expr {
        let x = x.into();
        let mu = mu.into();
        let sigma = sigma.into();

        let diff = x - mu;
        let standardized = diff.clone() / sigma;
        let quadratic = standardized.clone() * standardized;

        Expr::exp(Expr::constant(-0.5) * quadratic)
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

/// Pretty printing utilities for expressions
pub mod display {
    use super::Expr;
    use std::fmt;

    /// A wrapper for pretty-printing expressions with mathematical notation
    pub struct PrettyExpr<'a>(pub &'a Expr);

    impl fmt::Display for PrettyExpr<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    /// Format an expression as LaTeX
    #[must_use]
    pub fn latex(expr: &Expr) -> String {
        expr.to_latex()
    }

    /// Format an expression as Python code
    #[must_use]
    pub fn python(expr: &Expr) -> String {
        format!(
            "import math\n\ndef f({}):\n    return {}",
            expr.variables().join(", "),
            expr.to_python()
        )
    }

    /// Format an expression as a mathematical equation
    #[must_use]
    pub fn equation(lhs: &str, expr: &Expr) -> String {
        format!("{} = {}", lhs, PrettyExpr(expr))
    }
}
