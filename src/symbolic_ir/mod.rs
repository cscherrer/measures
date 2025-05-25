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
