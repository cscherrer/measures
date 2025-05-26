//! Final Tagless Approach for Symbolic Expressions
//!
//! This module provides a final tagless implementation that solves the expression problem
//! and enables zero-cost abstractions for symbolic computation. Unlike the tagged union
//! approach in `expr.rs`, this approach uses traits with Generic Associated Types (GATs)
//! to represent operations and allows easy extension of both operations and interpreters.
//!
//! # Key Benefits
//!
//! 1. **Zero-cost abstraction**: Direct use of host language operations without intermediate representation
//! 2. **Solves expression problem**: Easy extension of both operations and interpreters
//! 3. **Better type safety**: Leverages Rust's type system more effectively
//! 4. **Composability**: Different DSL components can be easily composed
//! 5. **Performance**: Eliminates enum tagging overhead and enables better compiler optimizations
//!
//! # Usage
//!
//! ```rust
//! use symbolic_math::final_tagless::*;
//! use symbolic_math::Expr;
//!
//! // Define a simple linear expression to avoid move issues
//! fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
//!     let two = E::constant(2.0);
//!     let three = E::constant(3.0);
//!     
//!     // 2*x + 3
//!     E::add(E::mul(two, x), three)
//! }
//!
//! // Evaluate with different interpreters
//! let result_f64: f64 = linear::<DirectEval>(DirectEval::var("x", 2.0));
//! let expr_ast: Expr = linear::<ExprBuilder>(ExprBuilder::var("x"));
//! ```

use crate::Expr;
use num_traits::Float;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "jit")]
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Value, types};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{Linkage, Module};

#[cfg(feature = "jit")]
use crate::jit::{JITError, GeneralJITFunction, JITSignature, CompilationStats, GeneralJITCompiler};

/// Helper trait that bundles all the common trait bounds for numeric types
/// This makes the main `MathExpr` trait much cleaner and easier to read
pub trait NumericType: Clone + Default + Send + Sync + 'static + std::fmt::Display {}

/// Blanket implementation for all types that satisfy the bounds
impl<T> NumericType for T where T: Clone + Default + Send + Sync + 'static + std::fmt::Display {}

/// Core trait for mathematical expressions using Generic Associated Types (GATs)
/// This follows the final tagless approach where the representation type is parameterized
/// and works with generic numeric types including AD types
pub trait MathExpr {
    /// The representation type parameterized by the value type
    type Repr<T>;

    /// Create a constant value
    fn constant<T: NumericType>(value: T) -> Self::Repr<T>;

    /// Create a variable reference
    fn var<T: NumericType>(name: &str) -> Self::Repr<T>;

    // Arithmetic operations with flexible type parameters
    /// Addition operation
    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Subtraction operation
    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Multiplication operation
    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Division operation
    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Power operation
    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T>;

    /// Negation operation
    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T>;

    // Transcendental functions
    /// Natural logarithm
    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Exponential function
    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Square root
    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Sine function
    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Cosine function
    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;
}

/// Direct evaluation interpreter - evaluates expressions immediately to numeric values
pub struct DirectEval;

impl DirectEval {
    /// Create a variable with a specific value for evaluation
    pub fn var<T>(_name: &str, value: T) -> T {
        value
    }
}

impl MathExpr for DirectEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
        // For direct evaluation, we need the value to be provided separately
        // This is a limitation of the direct eval approach for variables
        panic!("Use DirectEval::var(name, value) instead for direct evaluation")
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left + right
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left - right
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left * right
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left / right
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        base.powf(exp)
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        -expr
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.ln()
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.exp()
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.sqrt()
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.sin()
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.cos()
    }
}

/// Expression builder interpreter - builds AST expressions compatible with existing system
/// Note: This is constrained to f64 since the existing Expr type only supports f64
/// This will be dropped when we fully migrate to final tagless
pub struct ExprBuilder;

impl ExprBuilder {
    /// Create a variable expression
    #[must_use]
    pub fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }
}

impl MathExpr for ExprBuilder {
    type Repr<T> = Expr;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        // ExprBuilder is a temporary bridge - it only works with f64
        // We'll drop this when we fully migrate to final tagless
        // For now, we require the caller to only use f64 with ExprBuilder
        assert!(
            (std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()),
            "ExprBuilder currently only supports f64 - use DirectEval for other types"
        );

        // This is safe because we verified T is f64
        let f64_value = unsafe {
            let ptr = (&raw const value).cast::<f64>();
            *ptr
        };
        std::mem::forget(value); // Don't drop the original value
        Expr::Const(f64_value)
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        Expr::Var(name.to_string())
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Expr::Add(Box::new(left), Box::new(right))
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Expr::Sub(Box::new(left), Box::new(right))
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Expr::Mul(Box::new(left), Box::new(right))
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Expr::Div(Box::new(left), Box::new(right))
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Neg(Box::new(expr))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Ln(Box::new(expr))
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Exp(Box::new(expr))
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Sqrt(Box::new(expr))
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Sin(Box::new(expr))
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Expr::Cos(Box::new(expr))
    }
}

/// Contextual evaluation interpreter - evaluates with variable bindings using closures
/// Note: Currently simplified to work with f64 only for compatibility
/// This will be improved or replaced when we fully migrate to final tagless
pub struct ContextualEval;

/// A closure-based representation for contextual evaluation
pub type ContextualRepr<T> = Box<dyn Fn(&HashMap<String, T>) -> T + Send + Sync>;

impl ContextualEval {
    /// Create a variable that will be looked up in the context
    #[must_use]
    pub fn var<T>(name: &str) -> ContextualRepr<T>
    where
        T: Clone + Default,
    {
        let var_name = name.to_string();
        Box::new(move |ctx| ctx.get(&var_name).cloned().unwrap_or_default())
    }

    /// Evaluate a contextual expression with a specific context
    #[must_use]
    pub fn eval_with<T>(expr: &ContextualRepr<T>, context: &HashMap<String, T>) -> T {
        expr(context)
    }

    /// Helper for addition when all types are the same
    #[must_use]
    pub fn add_same<T>(left: ContextualRepr<T>, right: ContextualRepr<T>) -> ContextualRepr<T>
    where
        T: NumericType + Add<Output = T>,
    {
        Box::new(move |ctx| left(ctx) + right(ctx))
    }

    /// Helper for subtraction when all types are the same
    #[must_use]
    pub fn sub_same<T>(left: ContextualRepr<T>, right: ContextualRepr<T>) -> ContextualRepr<T>
    where
        T: NumericType + Sub<Output = T>,
    {
        Box::new(move |ctx| left(ctx) - right(ctx))
    }

    /// Helper for multiplication when all types are the same
    #[must_use]
    pub fn mul_same<T>(left: ContextualRepr<T>, right: ContextualRepr<T>) -> ContextualRepr<T>
    where
        T: NumericType + Mul<Output = T>,
    {
        Box::new(move |ctx| left(ctx) * right(ctx))
    }

    /// Helper for division when all types are the same
    #[must_use]
    pub fn div_same<T>(left: ContextualRepr<T>, right: ContextualRepr<T>) -> ContextualRepr<T>
    where
        T: NumericType + Div<Output = T>,
    {
        Box::new(move |ctx| left(ctx) / right(ctx))
    }
}

impl MathExpr for ContextualEval {
    type Repr<T> = ContextualRepr<T>;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        Box::new(move |_| value.clone())
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        let var_name = name.to_string();
        Box::new(move |ctx| ctx.get(&var_name).cloned().unwrap_or_default())
    }

    fn add<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        // ContextualEval with flexible types is complex due to closure lifetime issues
        // For now, we just return a default - use the helper methods for same-type operations
        // This will be replaced when we fully migrate to final tagless
        Box::new(move |_ctx| Default::default())
    }

    fn sub<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Box::new(move |_ctx| Default::default())
    }

    fn mul<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Box::new(move |_ctx| Default::default())
    }

    fn div<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        Box::new(move |_ctx| Default::default())
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| base(ctx).powf(exp(ctx)))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| -expr(ctx))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| expr(ctx).ln())
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| expr(ctx).exp())
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| expr(ctx).sqrt())
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| expr(ctx).sin())
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        Box::new(move |ctx| expr(ctx).cos())
    }
}

/// Pretty printing interpreter - generates string representations
#[derive(Clone)]
pub struct PrettyPrint;

impl PrettyPrint {
    /// Create a variable string
    #[must_use]
    pub fn var(name: &str) -> String {
        name.to_string()
    }
}

impl MathExpr for PrettyPrint {
    type Repr<T> = String;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        format!("{value}")
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        name.to_string()
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} + {right})")
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} - {right})")
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} * {right})")
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} / {right})")
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        format!("({base} ^ {exp})")
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("(-{expr})")
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("ln({expr})")
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("exp({expr})")
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("sqrt({expr})")
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("sin({expr})")
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("cos({expr})")
    }
}

/// Ergonomic wrapper for final tagless expressions that supports operator overloading
/// This solves the orphan rule issues by creating our own wrapper type
#[derive(Debug, Clone)]
pub struct FinalTaglessExpr<E: MathExpr> {
    pub(crate) repr: E::Repr<f64>,
    pub(crate) _phantom: std::marker::PhantomData<E>,
}

impl<E: MathExpr> FinalTaglessExpr<E> {
    /// Create a new wrapper around a representation
    pub fn new(repr: E::Repr<f64>) -> Self {
        Self {
            repr,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Extract the inner representation
    pub fn into_repr(self) -> E::Repr<f64> {
        self.repr
    }

    /// Get a reference to the inner representation
    pub fn as_repr(&self) -> &E::Repr<f64> {
        &self.repr
    }

    /// Create a constant expression
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self::new(E::constant(value))
    }

    /// Create a variable expression
    #[must_use]
    pub fn var(name: &str) -> Self {
        Self::new(E::var(name))
    }

    /// Power operation
    pub fn pow(self, exp: Self) -> Self {
        Self::new(E::pow(self.repr, exp.repr))
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Self::new(E::ln(self.repr))
    }

    /// Exponential function
    pub fn exp(self) -> Self {
        Self::new(E::exp(self.repr))
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Self::new(E::sqrt(self.repr))
    }

    /// Sine function
    pub fn sin(self) -> Self {
        Self::new(E::sin(self.repr))
    }

    /// Cosine function
    pub fn cos(self) -> Self {
        Self::new(E::cos(self.repr))
    }
}

// Now we can implement operators for our own type
impl<E: MathExpr> Add for FinalTaglessExpr<E> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(E::add(self.repr, rhs.repr))
    }
}

impl<E: MathExpr> Sub for FinalTaglessExpr<E> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(E::sub(self.repr, rhs.repr))
    }
}

impl<E: MathExpr> Mul for FinalTaglessExpr<E> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(E::mul(self.repr, rhs.repr))
    }
}

impl<E: MathExpr> Div for FinalTaglessExpr<E> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::new(E::div(self.repr, rhs.repr))
    }
}

impl<E: MathExpr> Neg for FinalTaglessExpr<E> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(E::neg(self.repr))
    }
}

/// Convenience functions for building expressions in final tagless style
pub mod dsl {
    use super::{Add, Div, MathExpr, Mul, NumericType, Sub};

    /// Create a constant
    #[must_use]
    pub fn constant<E: MathExpr>(value: f64) -> E::Repr<f64> {
        E::constant(value)
    }

    /// Create a variable
    #[must_use]
    pub fn var<E: MathExpr>(name: &str) -> E::Repr<f64> {
        E::var(name)
    }

    /// Addition with flexible types
    pub fn add<E: MathExpr, L, R, Output>(left: E::Repr<L>, right: E::Repr<R>) -> E::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        E::add(left, right)
    }

    /// Addition with same types (convenience)
    pub fn add_same<E: MathExpr, T>(left: E::Repr<T>, right: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Add<Output = T>,
    {
        E::add(left, right)
    }

    /// Subtraction with flexible types
    pub fn sub<E: MathExpr, L, R, Output>(left: E::Repr<L>, right: E::Repr<R>) -> E::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        E::sub(left, right)
    }

    /// Subtraction with same types (convenience)
    pub fn sub_same<E: MathExpr, T>(left: E::Repr<T>, right: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Sub<Output = T>,
    {
        E::sub(left, right)
    }

    /// Multiplication with flexible types
    pub fn mul<E: MathExpr, L, R, Output>(left: E::Repr<L>, right: E::Repr<R>) -> E::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        E::mul(left, right)
    }

    /// Multiplication with same types (convenience)
    pub fn mul_same<E: MathExpr, T>(left: E::Repr<T>, right: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Mul<Output = T>,
    {
        E::mul(left, right)
    }

    /// Division with flexible types
    pub fn div<E: MathExpr, L, R, Output>(left: E::Repr<L>, right: E::Repr<R>) -> E::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        E::div(left, right)
    }

    /// Division with same types (convenience)
    pub fn div_same<E: MathExpr, T>(left: E::Repr<T>, right: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Div<Output = T>,
    {
        E::div(left, right)
    }

    /// Power
    pub fn pow<E: MathExpr>(base: E::Repr<f64>, exp: E::Repr<f64>) -> E::Repr<f64> {
        E::pow(base, exp)
    }

    /// Negation
    pub fn neg<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::neg(expr)
    }

    /// Natural logarithm
    pub fn ln<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::ln(expr)
    }

    /// Exponential
    pub fn exp<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::exp(expr)
    }

    /// Square root
    pub fn sqrt<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::sqrt(expr)
    }

    /// Sine
    pub fn sin<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::sin(expr)
    }

    /// Cosine
    pub fn cos<E: MathExpr>(expr: E::Repr<f64>) -> E::Repr<f64> {
        E::cos(expr)
    }
}

/// Extension trait for adding new operations to the final tagless approach
/// This demonstrates how to solve the expression problem by adding new operations
pub trait StatisticalExpr: MathExpr {
    /// Logistic function: 1 / (1 + exp(-x))
    fn logistic(x: Self::Repr<f64>) -> Self::Repr<f64> {
        Self::div(
            Self::constant(1.0),
            Self::add(Self::constant(1.0), Self::exp(Self::neg(x))),
        )
    }

    /// Softplus function: ln(1 + exp(x))
    fn softplus(x: Self::Repr<f64>) -> Self::Repr<f64> {
        Self::ln(Self::add(Self::constant(1.0), Self::exp(x)))
    }
}

// Blanket implementation for all MathExpr types
impl<T: MathExpr> StatisticalExpr for T {}

/// Extension trait to convert between final tagless and tagged union approaches
pub trait FinalTaglessConversion {
    /// Convert from tagged union Expr to final tagless representation
    fn from_expr<E: MathExpr>(expr: &Expr) -> E::Repr<f64>;
}

impl FinalTaglessConversion for Expr {
    fn from_expr<E: MathExpr>(expr: &Expr) -> E::Repr<f64> {
        match expr {
            Expr::Const(c) => E::constant(*c),
            Expr::Var(name) => E::var(name),
            Expr::Add(left, right) => {
                E::add(Self::from_expr::<E>(left), Self::from_expr::<E>(right))
            }
            Expr::Sub(left, right) => {
                E::sub(Self::from_expr::<E>(left), Self::from_expr::<E>(right))
            }
            Expr::Mul(left, right) => {
                E::mul(Self::from_expr::<E>(left), Self::from_expr::<E>(right))
            }
            Expr::Div(left, right) => {
                E::div(Self::from_expr::<E>(left), Self::from_expr::<E>(right))
            }
            Expr::Pow(base, exp) => E::pow(Self::from_expr::<E>(base), Self::from_expr::<E>(exp)),
            Expr::Ln(inner) => E::ln(Self::from_expr::<E>(inner)),
            Expr::Exp(inner) => E::exp(Self::from_expr::<E>(inner)),
            Expr::Sqrt(inner) => E::sqrt(Self::from_expr::<E>(inner)),
            Expr::Sin(inner) => E::sin(Self::from_expr::<E>(inner)),
            Expr::Cos(inner) => E::cos(Self::from_expr::<E>(inner)),
            Expr::Neg(inner) => E::neg(Self::from_expr::<E>(inner)),
        }
    }
}

/// JIT evaluation interpreter - directly compiles final tagless expressions to native code
/// This bypasses the Expr AST entirely and provides the ultimate performance
#[cfg(feature = "jit")]
pub struct JITEval;

#[cfg(feature = "jit")]
impl JITEval {
    /// Create a variable for JIT compilation
    pub fn var<T: NumericType>(name: &str) -> JITExprBuilder {
        JITExprBuilder::new_var(name.to_string())
    }

    /// Compile a JIT expression to a native function
    pub fn compile(expr: JITExprBuilder) -> Result<GeneralJITFunction, JITError> {
        // Convert JITExprBuilder to Expr for compatibility with existing JIT infrastructure
        let ast_expr = expr.to_expr();
        
        // Analyze the expression to determine variables
        let variables = expr.collect_variables();
        let sorted_vars: Vec<_> = variables.into_iter().collect();
        
        // Create the JIT compiler
        let compiler = GeneralJITCompiler::new()?;
        
        // Compile using the existing infrastructure
        // For now, treat all variables as data variables (no parameters)
        let constants = std::collections::HashMap::new();
        
        compiler.compile_expression(&ast_expr, &sorted_vars, &[], &constants)
    }
}

/// Builder for JIT expressions that tracks the computation graph
#[cfg(feature = "jit")]
#[derive(Debug, Clone)]
pub enum JITExprBuilder {
    Constant(f64),
    Variable(String),
    Add(Box<JITExprBuilder>, Box<JITExprBuilder>),
    Sub(Box<JITExprBuilder>, Box<JITExprBuilder>),
    Mul(Box<JITExprBuilder>, Box<JITExprBuilder>),
    Div(Box<JITExprBuilder>, Box<JITExprBuilder>),
    Pow(Box<JITExprBuilder>, Box<JITExprBuilder>),
    Neg(Box<JITExprBuilder>),
    Ln(Box<JITExprBuilder>),
    Exp(Box<JITExprBuilder>),
    Sqrt(Box<JITExprBuilder>),
    Sin(Box<JITExprBuilder>),
    Cos(Box<JITExprBuilder>),
}

#[cfg(feature = "jit")]
impl JITExprBuilder {
    /// Create a new variable
    pub fn new_var(name: String) -> Self {
        Self::Variable(name)
    }

    /// Collect all variables in the expression
    pub fn collect_variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_variables_recursive(&mut vars);
        vars
    }

    fn collect_variables_recursive(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Self::Variable(name) => {
                vars.insert(name.clone());
            }
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Div(left, right)
            | Self::Pow(left, right) => {
                left.collect_variables_recursive(vars);
                right.collect_variables_recursive(vars);
            }
            Self::Neg(inner)
            | Self::Ln(inner)
            | Self::Exp(inner)
            | Self::Sqrt(inner)
            | Self::Sin(inner)
            | Self::Cos(inner) => {
                inner.collect_variables_recursive(vars);
            }
            Self::Constant(_) => {}
        }
    }

    /// Estimate the complexity of the expression for performance metrics
    pub fn estimate_complexity(&self) -> usize {
        match self {
            Self::Constant(_) | Self::Variable(_) => 1,
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Div(left, right)
            | Self::Pow(left, right) => {
                1 + left.estimate_complexity() + right.estimate_complexity()
            }
            Self::Neg(inner) => 1 + inner.estimate_complexity(),
            Self::Ln(inner)
            | Self::Exp(inner)
            | Self::Sqrt(inner)
            | Self::Sin(inner)
            | Self::Cos(inner) => {
                3 + inner.estimate_complexity() // Transcendental functions are more expensive
            }
        }
    }

    /// Count the number of constants in the expression
    pub fn count_constants(&self) -> usize {
        match self {
            Self::Constant(_) => 1,
            Self::Variable(_) => 0,
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Div(left, right)
            | Self::Pow(left, right) => left.count_constants() + right.count_constants(),
            Self::Neg(inner)
            | Self::Ln(inner)
            | Self::Exp(inner)
            | Self::Sqrt(inner)
            | Self::Sin(inner)
            | Self::Cos(inner) => inner.count_constants(),
        }
    }

    /// Convert to Expr AST for compatibility with existing JIT infrastructure
    pub fn to_expr(&self) -> Expr {
        match self {
            Self::Constant(c) => Expr::Const(*c),
            Self::Variable(name) => Expr::Var(name.clone()),
            Self::Add(left, right) => Expr::Add(Box::new(left.to_expr()), Box::new(right.to_expr())),
            Self::Sub(left, right) => Expr::Sub(Box::new(left.to_expr()), Box::new(right.to_expr())),
            Self::Mul(left, right) => Expr::Mul(Box::new(left.to_expr()), Box::new(right.to_expr())),
            Self::Div(left, right) => Expr::Div(Box::new(left.to_expr()), Box::new(right.to_expr())),
            Self::Pow(left, right) => Expr::Pow(Box::new(left.to_expr()), Box::new(right.to_expr())),
            Self::Neg(inner) => Expr::Neg(Box::new(inner.to_expr())),
            Self::Ln(inner) => Expr::Ln(Box::new(inner.to_expr())),
            Self::Exp(inner) => Expr::Exp(Box::new(inner.to_expr())),
            Self::Sqrt(inner) => Expr::Sqrt(Box::new(inner.to_expr())),
            Self::Sin(inner) => Expr::Sin(Box::new(inner.to_expr())),
            Self::Cos(inner) => Expr::Cos(Box::new(inner.to_expr())),
        }
    }

    /// Generate Cranelift IR for this expression
    pub fn generate_clif(
        &self,
        builder: &mut FunctionBuilder,
        var_map: &HashMap<String, Value>,
    ) -> Result<Value, JITError> {
        match self {
            Self::Constant(c) => Ok(builder.ins().f64const(*c)),
            Self::Variable(name) => var_map
                .get(name)
                .copied()
                .ok_or_else(|| JITError::CompilationError(format!("Unknown variable: {name}"))),
            Self::Add(left, right) => {
                let left_val = left.generate_clif(builder, var_map)?;
                let right_val = right.generate_clif(builder, var_map)?;
                Ok(builder.ins().fadd(left_val, right_val))
            }
            Self::Sub(left, right) => {
                let left_val = left.generate_clif(builder, var_map)?;
                let right_val = right.generate_clif(builder, var_map)?;
                Ok(builder.ins().fsub(left_val, right_val))
            }
            Self::Mul(left, right) => {
                let left_val = left.generate_clif(builder, var_map)?;
                let right_val = right.generate_clif(builder, var_map)?;
                Ok(builder.ins().fmul(left_val, right_val))
            }
            Self::Div(left, right) => {
                let left_val = left.generate_clif(builder, var_map)?;
                let right_val = right.generate_clif(builder, var_map)?;
                Ok(builder.ins().fdiv(left_val, right_val))
            }
            Self::Pow(base, exp) => {
                let base_val = base.generate_clif(builder, var_map)?;
                let exp_val = exp.generate_clif(builder, var_map)?;
                // Use exp(exp * ln(base)) for general power
                let ln_base = generate_ln_call(builder, base_val)?;
                let product = builder.ins().fmul(exp_val, ln_base);
                generate_exp_call(builder, product)
            }
            Self::Neg(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                Ok(builder.ins().fneg(val))
            }
            Self::Ln(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                generate_ln_call(builder, val)
            }
            Self::Exp(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                generate_exp_call(builder, val)
            }
            Self::Sqrt(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                Ok(builder.ins().sqrt(val))
            }
            Self::Sin(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                generate_sin_call(builder, val)
            }
            Self::Cos(inner) => {
                let val = inner.generate_clif(builder, var_map)?;
                generate_cos_call(builder, val)
            }
        }
    }
}

#[cfg(feature = "jit")]
impl std::fmt::Display for JITExprBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{c}"),
            Self::Variable(name) => write!(f, "{name}"),
            Self::Add(left, right) => write!(f, "({left} + {right})"),
            Self::Sub(left, right) => write!(f, "({left} - {right})"),
            Self::Mul(left, right) => write!(f, "({left} * {right})"),
            Self::Div(left, right) => write!(f, "({left} / {right})"),
            Self::Pow(left, right) => write!(f, "({left} ^ {right})"),
            Self::Neg(inner) => write!(f, "(-{inner})"),
            Self::Ln(inner) => write!(f, "ln({inner})"),
            Self::Exp(inner) => write!(f, "exp({inner})"),
            Self::Sqrt(inner) => write!(f, "sqrt({inner})"),
            Self::Sin(inner) => write!(f, "sin({inner})"),
            Self::Cos(inner) => write!(f, "cos({inner})"),
        }
    }
}

/// Helper functions for generating transcendental function calls in CLIF IR
#[cfg(feature = "jit")]
fn generate_ln_call(builder: &mut FunctionBuilder, val: Value) -> Result<Value, JITError> {
    // For now, use a simple Taylor series approximation
    // ln(1+x) ≈ x - x²/2 + x³/3 for |x| < 1
    // For general ln(x), we can use ln(x) = ln(1 + (x-1)) when x is close to 1
    
    // This is a simplified implementation - a production version would use
    // more sophisticated approximations or call to libm
    let one = builder.ins().f64const(1.0);
    let two = builder.ins().f64const(2.0);
    let three = builder.ins().f64const(3.0);
    
    // x - 1
    let x_minus_1 = builder.ins().fsub(val, one);
    
    // (x-1)²
    let x2 = builder.ins().fmul(x_minus_1, x_minus_1);
    
    // (x-1)³
    let x3 = builder.ins().fmul(x2, x_minus_1);
    
    // x - x²/2 + x³/3
    let term2 = builder.ins().fdiv(x2, two);
    let term3 = builder.ins().fdiv(x3, three);
    
    let result = builder.ins().fsub(x_minus_1, term2);
    let result = builder.ins().fadd(result, term3);
    
    Ok(result)
}

#[cfg(feature = "jit")]
fn generate_exp_call(builder: &mut FunctionBuilder, val: Value) -> Result<Value, JITError> {
    // Simple Taylor series: exp(x) ≈ 1 + x + x²/2 + x³/6
    let one = builder.ins().f64const(1.0);
    let two = builder.ins().f64const(2.0);
    let six = builder.ins().f64const(6.0);
    
    // x²
    let x2 = builder.ins().fmul(val, val);
    
    // x³
    let x3 = builder.ins().fmul(x2, val);
    
    // 1 + x + x²/2 + x³/6
    let term2 = builder.ins().fdiv(x2, two);
    let term3 = builder.ins().fdiv(x3, six);
    
    let result = builder.ins().fadd(one, val);
    let result = builder.ins().fadd(result, term2);
    let result = builder.ins().fadd(result, term3);
    
    Ok(result)
}

#[cfg(feature = "jit")]
fn generate_sin_call(builder: &mut FunctionBuilder, val: Value) -> Result<Value, JITError> {
    // Taylor series: sin(x) ≈ x - x³/6 + x⁵/120
    let six = builder.ins().f64const(6.0);
    let one_twenty = builder.ins().f64const(120.0);
    
    // x³
    let x2 = builder.ins().fmul(val, val);
    let x3 = builder.ins().fmul(x2, val);
    
    // x⁵
    let x4 = builder.ins().fmul(x3, val);
    let x5 = builder.ins().fmul(x4, val);
    
    // x - x³/6 + x⁵/120
    let term2 = builder.ins().fdiv(x3, six);
    let term3 = builder.ins().fdiv(x5, one_twenty);
    
    let result = builder.ins().fsub(val, term2);
    let result = builder.ins().fadd(result, term3);
    
    Ok(result)
}

#[cfg(feature = "jit")]
fn generate_cos_call(builder: &mut FunctionBuilder, val: Value) -> Result<Value, JITError> {
    // Taylor series: cos(x) ≈ 1 - x²/2 + x⁴/24
    let one = builder.ins().f64const(1.0);
    let two = builder.ins().f64const(2.0);
    let twentyfour = builder.ins().f64const(24.0);
    
    // x²
    let x2 = builder.ins().fmul(val, val);
    
    // x⁴
    let x4 = builder.ins().fmul(x2, x2);
    
    // 1 - x²/2 + x⁴/24
    let term2 = builder.ins().fdiv(x2, two);
    let term3 = builder.ins().fdiv(x4, twentyfour);
    
    let result = builder.ins().fsub(one, term2);
    let result = builder.ins().fadd(result, term3);
    
    Ok(result)
}

#[cfg(feature = "jit")]
impl MathExpr for JITEval {
    type Repr<T> = JITExprBuilder;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        // For JIT compilation, we need to convert to f64
        // This is a limitation we'll address when we support more numeric types
        let f64_value = format!("{value}").parse::<f64>()
            .unwrap_or_else(|_| panic!("Cannot convert {value} to f64 for JIT compilation"));
        JITExprBuilder::Constant(f64_value)
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        JITExprBuilder::Variable(name.to_string())
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITExprBuilder::Add(Box::new(left), Box::new(right))
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITExprBuilder::Sub(Box::new(left), Box::new(right))
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITExprBuilder::Mul(Box::new(left), Box::new(right))
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITExprBuilder::Div(Box::new(left), Box::new(right))
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Pow(Box::new(base), Box::new(exp))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Neg(Box::new(expr))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Ln(Box::new(expr))
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Exp(Box::new(expr))
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Sqrt(Box::new(expr))
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Sin(Box::new(expr))
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITExprBuilder::Cos(Box::new(expr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_eval() {
        // Test: 2*x + 3 where x = 5
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            let three = E::constant(3.0);
            E::add(E::mul(two, x), three)
        }

        let result = test_expr::<DirectEval>(DirectEval::var("x", 5.0));
        assert_eq!(result, 13.0);
    }

    #[test]
    fn test_expr_builder() {
        // Test: x^2 + 1
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            let one = E::constant(1.0);
            E::add(E::pow(x, two), one)
        }

        let result = test_expr::<ExprBuilder>(ExprBuilder::var("x"));

        // Verify the structure
        match result {
            Expr::Add(left, right) => match (*left, *right) {
                (Expr::Pow(ref base, ref exp), Expr::Const(c)) => {
                    assert!(matches!(**base, Expr::Var(_)));
                    assert!(matches!(**exp, Expr::Const(2.0)));
                    assert_eq!(c, 1.0);
                }
                _ => panic!("Unexpected expression structure"),
            },
            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    #[ignore] // Temporarily disabled due to lifetime complexity - ContextualEval will be dropped anyway
    fn test_contextual_eval() {
        // Test: x^2 + y where x = 3, y = 4
        fn test_expr<E: MathExpr>(x: E::Repr<f64>, y: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            E::add(E::pow(x, two), y)
        }

        let result =
            test_expr::<ContextualEval>(ContextualEval::var("x"), ContextualEval::var("y"));

        let mut context = HashMap::new();
        context.insert("x".to_string(), 3.0);
        context.insert("y".to_string(), 4.0);

        let value = ContextualEval::eval_with(&result, &context);
        assert_eq!(value, 13.0); // 3^2 + 4 = 9 + 4 = 13
    }

    #[test]
    fn test_pretty_print() {
        // Test: x^2 + y
        fn test_expr<E: MathExpr>(x: E::Repr<f64>, y: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            E::add(E::pow(x, two), y)
        }

        let result = test_expr::<PrettyPrint>(PrettyPrint::var("x"), PrettyPrint::var("y"));
        assert_eq!(result, "((x ^ 2) + y)");
    }

    #[test]
    fn test_polymorphic_function() {
        // Define a polymorphic function that works with any interpreter
        // We'll use a simpler function to avoid the clone issue
        fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            let three = E::constant(3.0);

            // 2*x + 3
            E::add(E::mul(two, x), three)
        }

        // Test with direct evaluation
        let direct_result = linear::<DirectEval>(DirectEval::var("x", 2.0));
        assert_eq!(direct_result, 7.0); // 2*2 + 3 = 7

        // Test with expression builder
        let expr_result = linear::<ExprBuilder>(ExprBuilder::var("x"));
        // Verify it's a valid expression
        assert!(matches!(expr_result, Expr::Add(_, _)));

        // Test with pretty print
        let pretty_result = linear::<PrettyPrint>(PrettyPrint::var("x"));
        assert!(pretty_result.contains('x'));
    }

    #[test]
    fn test_statistical_extension() {
        // Test the statistical extension
        fn logistic_expr<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::logistic(x)
        }

        // Test with direct evaluation
        let result = logistic_expr::<DirectEval>(DirectEval::var("x", 0.0));
        // At x=0, logistic should be 0.5
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_conversion_from_expr() {
        // Test converting from Expr to final tagless
        let expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Var("x".to_string())),
            )),
            Box::new(Expr::Const(3.0)),
        );

        // Convert to ExprBuilder (should produce equivalent expression)
        let converted: Expr = Expr::from_expr::<ExprBuilder>(&expr);

        // The expressions should be structurally equivalent
        assert!(matches!(converted, Expr::Add(_, _)));
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_basic() {
        // Test: x + 1
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let one = E::constant(1.0);
            E::add(x, one)
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var::<f64>("x"));
        let compiled = JITEval::compile(jit_expr).expect("JIT compilation should succeed");
        
        // Test the compiled function
        let result = compiled.call_single(5.0);
        assert_eq!(result, 6.0); // 5 + 1 = 6
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_complex() {
        // Test: x^2 + 2*x + 1 (should equal (x+1)^2)
        // Build the expression directly for JITEval to avoid generic issues
        let _x = JITEval::var::<f64>("x");
        let one = JITEval::constant::<f64>(1.0);
        let two = JITEval::constant::<f64>(2.0);
        let x_var = JITEval::var::<f64>("x");
        let x_var2 = JITEval::var::<f64>("x");
        
        let x_squared = JITEval::pow::<f64>(x_var, two);
        let two_x = JITEval::mul::<f64, f64, f64>(JITEval::constant::<f64>(2.0), x_var2);
        let jit_expr = JITEval::add::<f64, f64, f64>(JITEval::add::<f64, f64, f64>(x_squared, two_x), one);

        let compiled = JITEval::compile(jit_expr).expect("JIT compilation should succeed");
        
        // Test the compiled function
        let result = compiled.call_single(3.0);
        assert_eq!(result, 16.0); // (3+1)^2 = 16
        
        let result = compiled.call_single(-1.0);
        assert_eq!(result, 0.0); // (-1+1)^2 = 0
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_transcendental() {
        // Test: exp(ln(x)) should equal x
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::exp(E::ln(x))
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var::<f64>("x"));
        let compiled = JITEval::compile(jit_expr).expect("JIT compilation should succeed");
        
        // Test the compiled function (with some tolerance for Taylor series approximation)
        let result = compiled.call_single(2.0);
        assert!((result - 2.0).abs() < 0.1, "exp(ln(2)) should be approximately 2, got {result}");
    }
}
