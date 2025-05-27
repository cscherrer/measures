//! Final Tagless Approach for Symbolic Mathematical Expressions
//!
//! This module implements the final tagless approach to solve the expression problem in symbolic
//! mathematics. The final tagless approach uses traits with Generic Associated Types (GATs) to
//! represent mathematical operations, enabling both easy extension of operations and interpreters
//! without modifying existing code.
//!
//! # Technical Motivation
//!
//! Traditional approaches to symbolic mathematics face the expression problem: adding new operations
//! requires modifying existing interpreter code, while adding new interpreters requires modifying
//! existing operation definitions. The final tagless approach solves this by:
//!
//! 1. **Parameterizing representation types**: Operations are defined over abstract representation
//!    types `Repr<T>`, allowing different interpreters to use different concrete representations
//! 2. **Trait-based extensibility**: New operations can be added via trait extension without
//!    modifying existing code
//! 3. **Zero intermediate representation**: Expressions compile directly to target representations
//!    without building intermediate ASTs
//!
//! # Architecture
//!
//! ## Core Traits
//!
//! - **`MathExpr`**: Defines basic mathematical operations (arithmetic, transcendental functions)
//! - **`StatisticalExpr`**: Extends `MathExpr` with statistical functions (logistic, softplus)
//! - **`NumericType`**: Helper trait bundling common numeric type requirements
//!
//! ## Interpreters
//!
//! - **`DirectEval`**: Immediate evaluation using native Rust operations (`type Repr<T> = T`)
//! - **`PrettyPrint`**: String representation generation (`type Repr<T> = String`)
//!
//! # Usage Patterns
//!
//! ## Polymorphic Expression Definition
//!
//! Define mathematical expressions that work with any interpreter:
//!
//! ```rust
//! use mathjit::final_tagless::*;
//!
//! // Define a quadratic function: 2x² + 3x + 1
//! fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! where
//!     E::Repr<f64>: Clone,
//! {
//!     let a = E::constant(2.0);
//!     let b = E::constant(3.0);
//!     let c = E::constant(1.0);
//!     
//!     E::add(
//!         E::add(
//!             E::mul(a, E::pow(x.clone(), E::constant(2.0))),
//!             E::mul(b, x)
//!         ),
//!         c
//!     )
//! }
//! ```
//!
//! ## Direct Evaluation
//!
//! Evaluate expressions immediately using native Rust operations:
//!
//! ```rust
//! # use mathjit::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
//! assert_eq!(result, 15.0); // 2(4) + 3(2) + 1 = 15
//! ```
//!
//! ## Pretty Printing
//!
//! Generate human-readable mathematical notation:
//!
//! ```rust
//! # use mathjit::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
//! println!("Expression: {}", pretty);
//! // Output: "((2 * (x ^ 2)) + (3 * x)) + 1"
//! ```
//!
//! # Extension Example
//!
//! Adding new operations requires only trait extension:
//!
//! ```rust
//! use mathjit::final_tagless::*;
//! use num_traits::Float;
//!
//! // Extend with hyperbolic functions
//! trait HyperbolicExpr: MathExpr {
//!     fn tanh<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>
//!     where
//!         Self::Repr<T>: Clone,
//!     {
//!         let exp_x = Self::exp(x.clone());
//!         let exp_neg_x = Self::exp(Self::neg(x));
//!         let numerator = Self::sub(exp_x.clone(), exp_neg_x.clone());
//!         let denominator = Self::add(exp_x, exp_neg_x);
//!         Self::div(numerator, denominator)
//!     }
//! }
//!
//! // Automatically works with all existing interpreters
//! impl HyperbolicExpr for DirectEval {}
//! impl HyperbolicExpr for PrettyPrint {}
//! ```

use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

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

/// Polynomial evaluation utilities using Horner's method
///
/// This module provides efficient polynomial evaluation using the final tagless approach.
/// Horner's method reduces the number of multiplications and provides better numerical
/// stability compared to naive polynomial evaluation.
pub mod polynomial {
    use super::{MathExpr, NumericType};
    use std::ops::{Add, Mul, Sub};

    /// Evaluate a polynomial using Horner's method
    ///
    /// Given coefficients [a₀, a₁, a₂, ..., aₙ] representing the polynomial:
    /// a₀ + a₁x + a₂x² + ... + aₙxⁿ
    ///
    /// Horner's method evaluates this as:
    /// a₀ + x(a₁ + x(a₂ + x(...)))
    ///
    /// This reduces the number of multiplications from O(n²) to O(n) and
    /// provides better numerical stability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::final_tagless::{DirectEval, polynomial::horner};
    ///
    /// // Evaluate 1 + 3x + 2x² at x = 2
    /// let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
    /// let x = DirectEval::var("x", 2.0);
    /// let result = horner::<DirectEval, f64>(&coeffs, x);
    /// assert_eq!(result, 15.0); // 1 + 3(2) + 2(4) = 15
    /// ```
    ///
    /// # Type Parameters
    ///
    /// - `E`: The expression interpreter (`DirectEval`, `PrettyPrint`, etc.)
    /// - `T`: The numeric type (f64, f32, etc.)
    pub fn horner<E: MathExpr, T>(coeffs: &[T], x: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Clone + Add<Output = T> + Mul<Output = T>,
        E::Repr<T>: Clone,
    {
        if coeffs.is_empty() {
            return E::constant(T::default());
        }

        if coeffs.len() == 1 {
            return E::constant(coeffs[0].clone());
        }

        // Start with the highest degree coefficient (last in ascending order)
        let mut result = E::constant(coeffs[coeffs.len() - 1].clone());

        // Work backwards through the coefficients (from highest to lowest degree)
        for coeff in coeffs.iter().rev().skip(1) {
            result = E::add(E::mul(result, x.clone()), E::constant(coeff.clone()));
        }

        result
    }

    /// Evaluate a polynomial with explicit coefficients using Horner's method
    ///
    /// This is a convenience function for when you want to specify coefficients
    /// as expression representations rather than raw values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::final_tagless::{DirectEval, MathExpr, polynomial::horner_expr};
    ///
    /// // Evaluate 1 + 3x + 2x² at x = 2
    /// let coeffs = [
    ///     DirectEval::constant(1.0), // constant term
    ///     DirectEval::constant(3.0), // x coefficient  
    ///     DirectEval::constant(2.0), // x² coefficient
    /// ];
    /// let x = DirectEval::var("x", 2.0);
    /// let result = horner_expr::<DirectEval, f64>(&coeffs, x);
    /// assert_eq!(result, 15.0);
    /// ```
    pub fn horner_expr<E: MathExpr, T>(coeffs: &[E::Repr<T>], x: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Add<Output = T> + Mul<Output = T>,
        E::Repr<T>: Clone,
    {
        if coeffs.is_empty() {
            return E::constant(T::default());
        }

        if coeffs.len() == 1 {
            return coeffs[0].clone();
        }

        // Start with the highest degree coefficient
        let mut result = coeffs[coeffs.len() - 1].clone();

        // Work backwards through the coefficients
        for coeff in coeffs.iter().rev().skip(1) {
            result = E::add(E::mul(result, x.clone()), coeff.clone());
        }

        result
    }

    /// Create a polynomial from its roots using the final tagless approach
    ///
    /// Given roots [r₁, r₂, ..., rₙ], constructs the polynomial:
    /// (x - r₁)(x - r₂)...(x - rₙ)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::final_tagless::{DirectEval, polynomial::from_roots};
    ///
    /// // Create polynomial with roots at 1 and 2: (x-1)(x-2) = x² - 3x + 2
    /// let roots = [1.0, 2.0];
    /// let x = DirectEval::var("x", 0.0);
    /// let poly = from_roots::<DirectEval, f64>(&roots, x);
    /// // At x=0: (0-1)(0-2) = 2
    /// assert_eq!(poly, 2.0);
    /// ```
    pub fn from_roots<E: MathExpr, T>(roots: &[T], x: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Clone + Sub<Output = T> + num_traits::One,
        E::Repr<T>: Clone,
    {
        if roots.is_empty() {
            return E::constant(num_traits::One::one());
        }

        let mut result = E::sub(x.clone(), E::constant(roots[0].clone()));

        for root in roots.iter().skip(1) {
            let factor = E::sub(x.clone(), E::constant(root.clone()));
            result = E::mul(result, factor);
        }

        result
    }

    /// Evaluate the derivative of a polynomial using Horner's method
    ///
    /// Given coefficients [a₀, a₁, a₂, ..., aₙ] representing:
    /// a₀ + a₁x + a₂x² + ... + aₙxⁿ
    ///
    /// The derivative is: a₁ + 2a₂x + 3a₃x² + ... + naₙx^(n-1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::final_tagless::{DirectEval, polynomial::horner_derivative};
    ///
    /// // Derivative of 1 + 3x + 2x² is 3 + 4x
    /// let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
    /// let x = DirectEval::var("x", 2.0);
    /// let result = horner_derivative::<DirectEval, f64>(&coeffs, x);
    /// assert_eq!(result, 11.0); // 3 + 4(2) = 11
    /// ```
    pub fn horner_derivative<E: MathExpr, T>(coeffs: &[T], x: E::Repr<T>) -> E::Repr<T>
    where
        T: NumericType + Clone + Add<Output = T> + Mul<Output = T> + num_traits::FromPrimitive,
        E::Repr<T>: Clone,
    {
        if coeffs.len() <= 1 {
            return E::constant(T::default());
        }

        // Create derivative coefficients: [a₁, 2a₂, 3a₃, ...]
        let mut deriv_coeffs = Vec::with_capacity(coeffs.len() - 1);
        for (i, coeff) in coeffs.iter().enumerate().skip(1) {
            // Multiply coefficient by its power
            let power = num_traits::FromPrimitive::from_usize(i).unwrap_or_else(|| T::default());
            deriv_coeffs.push(coeff.clone() * power);
        }

        horner::<E, T>(&deriv_coeffs, x)
    }
}

/// Direct evaluation interpreter for immediate computation
///
/// This interpreter provides immediate evaluation of mathematical expressions using native Rust
/// operations. It represents expressions directly as their computed values (`type Repr<T> = T`),
/// making it the simplest and most straightforward interpreter implementation.
///
/// # Characteristics
///
/// - **Zero overhead**: Direct mapping to native Rust operations
/// - **Immediate evaluation**: No intermediate representation or compilation step
/// - **Type preservation**: Works with any numeric type that implements required traits
/// - **Reference implementation**: Serves as the canonical behavior for other interpreters
///
/// # Usage Patterns
///
/// ## Simple Expression Evaluation
///
/// ```rust
/// use mathjit::final_tagless::{DirectEval, MathExpr};
///
/// // Define a mathematical function
/// fn polynomial<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     // 3x² + 2x + 1
///     let x_squared = E::pow(x.clone(), E::constant(2.0));
///     let three_x_squared = E::mul(E::constant(3.0), x_squared);
///     let two_x = E::mul(E::constant(2.0), x);
///     E::add(E::add(three_x_squared, two_x), E::constant(1.0))
/// }
///
/// // Evaluate directly with a specific value
/// let result = polynomial::<DirectEval>(DirectEval::var("x", 2.0));
/// assert_eq!(result, 17.0); // 3(4) + 2(2) + 1 = 17
/// ```
///
/// ## Working with Different Numeric Types
///
/// ```rust
/// # use mathjit::final_tagless::{DirectEval, MathExpr, NumericType};
/// // Function that works with any numeric type
/// fn linear<E: MathExpr, T>(x: E::Repr<T>, slope: T, intercept: T) -> E::Repr<T>
/// where
///     T: Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + NumericType,
/// {
///     E::add(E::mul(E::constant(slope), x), E::constant(intercept))
/// }
///
/// // Works with f32
/// let result_f32 = linear::<DirectEval, f32>(
///     DirectEval::var("x", 3.0_f32),
///     2.0_f32,
///     1.0_f32
/// );
/// assert_eq!(result_f32, 7.0_f32);
///
/// // Works with f64
/// let result_f64 = linear::<DirectEval, f64>(
///     DirectEval::var("x", 3.0_f64),
///     2.0_f64,
///     1.0_f64
/// );
/// assert_eq!(result_f64, 7.0_f64);
/// ```
///
/// ## Testing and Validation
///
/// `DirectEval` is particularly useful for testing the correctness of expressions
/// before using them with other interpreters:
///
/// ```rust
/// # use mathjit::final_tagless::{DirectEval, MathExpr, StatisticalExpr};
/// // Test a statistical function
/// fn test_logistic<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
///     E::logistic(x)
/// }
///
/// // Verify known values
/// let result_zero = test_logistic::<DirectEval>(DirectEval::var("x", 0.0));
/// assert!((result_zero - 0.5).abs() < 1e-10); // logistic(0) = 0.5
///
/// let result_large = test_logistic::<DirectEval>(DirectEval::var("x", 10.0));
/// assert!(result_large > 0.99); // logistic(10) ≈ 1.0
/// ```
pub struct DirectEval;

impl DirectEval {
    /// Create a variable with a specific value for direct evaluation
    #[must_use]
    pub fn var<T: NumericType>(_name: &str, value: T) -> T {
        value
    }
}

impl MathExpr for DirectEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
        T::default()
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

/// Extension trait for statistical operations
pub trait StatisticalExpr: MathExpr {
    /// Logistic function: 1 / (1 + exp(-x))
    fn logistic<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        let one = Self::constant(T::one());
        let neg_x = Self::neg(x);
        let exp_neg_x = Self::exp(neg_x);
        let denominator = Self::add(one, exp_neg_x);
        Self::div(Self::constant(T::one()), denominator)
    }

    /// Softplus function: ln(1 + exp(x))
    fn softplus<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        let one = Self::constant(T::one());
        let exp_x = Self::exp(x);
        let one_plus_exp_x = Self::add(one, exp_x);
        Self::ln(one_plus_exp_x)
    }

    /// Sigmoid function (alias for logistic)
    fn sigmoid<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::logistic(x)
    }
}

// Implement StatisticalExpr for DirectEval
impl StatisticalExpr for DirectEval {}

/// String representation interpreter for mathematical expressions
///
/// This interpreter converts final tagless expressions into human-readable mathematical notation.
/// It generates parenthesized infix expressions that clearly show the structure and precedence
/// of operations. This is useful for debugging, documentation, and displaying expressions to users.
///
/// # Output Format
///
/// - **Arithmetic operations**: Infix notation with parentheses `(a + b)`, `(a * b)`
/// - **Functions**: Function call notation `ln(x)`, `exp(x)`, `sqrt(x)`
/// - **Variables**: Variable names as provided `x`, `theta`, `data`
/// - **Constants**: Numeric literals `2`, `3.14159`, `-1.5`
///
/// # Usage Examples
///
/// ## Basic Expression Formatting
///
/// ```rust
/// use mathjit::final_tagless::{PrettyPrint, MathExpr};
///
/// // Simple quadratic: x² + 2x + 1
/// fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     let x_squared = E::pow(x.clone(), E::constant(2.0));
///     let two_x = E::mul(E::constant(2.0), x);
///     E::add(E::add(x_squared, two_x), E::constant(1.0))
/// }
///
/// let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
/// println!("Quadratic: {}", pretty);
/// // Output: "((x ^ 2) + (2 * x)) + 1"
/// ```
///
/// ## Complex Mathematical Expressions
///
/// ```rust
/// # use mathjit::final_tagless::{PrettyPrint, MathExpr, StatisticalExpr};
/// // Logistic regression: 1 / (1 + exp(-θx))
/// fn logistic_regression<E: StatisticalExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
///     E::logistic(E::mul(theta, x))
/// }
///
/// let pretty = logistic_regression::<PrettyPrint>(
///     PrettyPrint::var("x"),
///     PrettyPrint::var("theta")
/// );
/// println!("Logistic: {}", pretty);
/// // Output shows the expanded logistic function structure
/// ```
///
/// ## Transcendental Functions
///
/// ```rust
/// # use mathjit::final_tagless::{PrettyPrint, MathExpr};
/// // Gaussian: exp(-x²/2) / sqrt(2π)
/// fn gaussian_kernel<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     let x_squared = E::pow(x, E::constant(2.0));
///     let neg_half_x_squared = E::div(E::neg(x_squared), E::constant(2.0));
///     let numerator = E::exp(neg_half_x_squared);
///     let denominator = E::sqrt(E::mul(E::constant(2.0), E::constant(3.14159)));
///     E::div(numerator, denominator)
/// }
///
/// let pretty = gaussian_kernel::<PrettyPrint>(PrettyPrint::var("x"));
/// println!("Gaussian: {}", pretty);
/// // Output: "(exp((-(x ^ 2)) / 2) / sqrt((2 * 3.14159)))"
/// ```
pub struct PrettyPrint;

impl PrettyPrint {
    /// Create a variable for pretty printing
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

// Implement StatisticalExpr for PrettyPrint
impl StatisticalExpr for PrettyPrint {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_eval() {
        // Test: 2*x + 3 where x = 5
        fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            let two = E::constant(2.0);
            let three = E::constant(3.0);
            E::add(E::mul(two, x), three)
        }

        let result = linear::<DirectEval>(DirectEval::var("x", 5.0));
        assert_eq!(result, 13.0); // 2*5 + 3 = 13
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
    fn test_pretty_print() {
        // Test pretty printing of expressions
        fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
        where
            E::Repr<f64>: Clone,
        {
            let a = E::constant(2.0);
            let b = E::constant(3.0);
            let c = E::constant(1.0);
            E::add(
                E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)),
                c,
            )
        }

        let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));

        // Should contain the key components
        assert!(pretty.contains('x'));
        assert!(pretty.contains('2'));
        assert!(pretty.contains('3'));
        assert!(pretty.contains('1'));
        assert!(pretty.contains('^'));
        assert!(pretty.contains('*'));
        assert!(pretty.contains('+'));
    }

    #[test]
    fn test_horner_polynomial() {
        use crate::final_tagless::polynomial::horner;

        // Test: 1 + 3x + 2x² at x = 2
        // Expected: 1 + 3(2) + 2(4) = 1 + 6 + 8 = 15
        let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
        let x = DirectEval::var("x", 2.0);
        let result = horner::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 15.0);

        // Test edge cases
        let empty_coeffs: [f64; 0] = [];
        let result_empty = horner::<DirectEval, f64>(&empty_coeffs, DirectEval::var("x", 5.0));
        assert_eq!(result_empty, 0.0);

        let single_coeff = [42.0];
        let result_single = horner::<DirectEval, f64>(&single_coeff, DirectEval::var("x", 5.0));
        assert_eq!(result_single, 42.0);
    }

    #[test]
    fn test_horner_pretty_print() {
        use crate::final_tagless::polynomial::horner;

        // Test pretty printing of Horner polynomial
        let coeffs = [1.0, 3.0, 2.0]; // 1 + 3x + 2x²
        let x = PrettyPrint::var("x");
        let pretty = horner::<PrettyPrint, f64>(&coeffs, x);

        // Should contain the structure of Horner's method
        assert!(pretty.contains('x'));
        assert!(pretty.contains('1'));
        assert!(pretty.contains('3'));
        assert!(pretty.contains('2'));
    }

    #[test]
    fn test_polynomial_from_roots() {
        use crate::final_tagless::polynomial::from_roots;

        // Test: (x-1)(x-2) = x² - 3x + 2
        let roots = [1.0, 2.0];

        // At x=0: (0-1)(0-2) = 2
        let result_0 = from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 0.0));
        assert_eq!(result_0, 2.0);

        // At x=1: (1-1)(1-2) = 0
        let result_1 = from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 1.0));
        assert_eq!(result_1, 0.0);

        // At x=2: (2-1)(2-2) = 0
        let result_2 = from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 2.0));
        assert_eq!(result_2, 0.0);

        // At x=3: (3-1)(3-2) = 2
        let result_3 = from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 3.0));
        assert_eq!(result_3, 2.0);
    }
}
