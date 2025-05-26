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

/// Helper trait that bundles all the common trait bounds for numeric types
/// This makes the main `MathExpr` trait much cleaner and easier to read
pub trait NumericType:
    Clone + Default + Send + Sync + 'static + std::fmt::Display + Into<f64>
{
}

/// Blanket implementation for all types that satisfy the bounds
impl<T> NumericType for T where
    T: Clone + Default + Send + Sync + 'static + std::fmt::Display + Into<f64>
{
}

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

    // Arithmetic operations
    /// Addition operation
    fn add<T: NumericType + Add<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T>;

    /// Subtraction operation
    fn sub<T: NumericType + Sub<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T>;

    /// Multiplication operation
    fn mul<T: NumericType + Mul<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T>;

    /// Division operation
    fn div<T: NumericType + Div<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T>;

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

    fn add<T: NumericType + Add<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        left + right
    }

    fn sub<T: NumericType + Sub<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        left - right
    }

    fn mul<T: NumericType + Mul<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        left * right
    }

    fn div<T: NumericType + Div<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
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
        // For ExprBuilder, we need to convert to f64 since Expr only supports f64
        // This is a limitation of the existing Expr type
        Expr::Const(value.into())
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        Expr::Var(name.to_string())
    }

    fn add<T: NumericType + Add<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Expr::Add(Box::new(left), Box::new(right))
    }

    fn sub<T: NumericType + Sub<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Expr::Sub(Box::new(left), Box::new(right))
    }

    fn mul<T: NumericType + Mul<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Expr::Mul(Box::new(left), Box::new(right))
    }

    fn div<T: NumericType + Div<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
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

    fn add<T: NumericType + Add<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Box::new(move |ctx| left(ctx) + right(ctx))
    }

    fn sub<T: NumericType + Sub<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Box::new(move |ctx| left(ctx) - right(ctx))
    }

    fn mul<T: NumericType + Mul<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Box::new(move |ctx| left(ctx) * right(ctx))
    }

    fn div<T: NumericType + Div<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        Box::new(move |ctx| left(ctx) / right(ctx))
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

    fn add<T: NumericType + Add<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        format!("({left} + {right})")
    }

    fn sub<T: NumericType + Sub<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        format!("({left} - {right})")
    }

    fn mul<T: NumericType + Mul<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
        format!("({left} * {right})")
    }

    fn div<T: NumericType + Div<Output = T>>(
        left: Self::Repr<T>,
        right: Self::Repr<T>,
    ) -> Self::Repr<T> {
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
    use super::{Add, Div, Float, MathExpr, Mul, Neg, Sub};

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

    /// Addition
    pub fn add<E: MathExpr>(left: E::Repr<f64>, right: E::Repr<f64>) -> E::Repr<f64> {
        E::add(left, right)
    }

    /// Subtraction
    pub fn sub<E: MathExpr>(left: E::Repr<f64>, right: E::Repr<f64>) -> E::Repr<f64> {
        E::sub(left, right)
    }

    /// Multiplication
    pub fn mul<E: MathExpr>(left: E::Repr<f64>, right: E::Repr<f64>) -> E::Repr<f64> {
        E::mul(left, right)
    }

    /// Division
    pub fn div<E: MathExpr>(left: E::Repr<f64>, right: E::Repr<f64>) -> E::Repr<f64> {
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
        // Test: x + 2
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            E::add(x, two)
        }

        let result = test_expr::<PrettyPrint>(PrettyPrint::var("x"));
        assert_eq!(result, "(x + 2)");
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
}
