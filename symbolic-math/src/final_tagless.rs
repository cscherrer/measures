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
//! - **`JITEval`**: Native code compilation via Cranelift IR (`type Repr<T> = JITRepr`)
//!
//! # Usage Patterns
//!
//! ## Polymorphic Expression Definition
//!
//! Define mathematical expressions that work with any interpreter:
//!
//! ```rust
//! use symbolic_math::final_tagless::*;
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
//! # use symbolic_math::final_tagless::*;
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
//! # use symbolic_math::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
//! println!("Expression: {}", pretty);
//! // Output: "((2 * (x ^ 2)) + (3 * x)) + 1"
//! ```
//!
//! ## JIT Compilation
//!
//! Compile expressions to native machine code for repeated evaluation:
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! # use symbolic_math::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let jit_expr = quadratic::<JITEval>(JITEval::var("x"));
//! let compiled = JITEval::compile_single_var(jit_expr, "x")?;
//! let result = compiled.call_single(2.0);
//! assert_eq!(result, 15.0);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Extension Example
//!
//! Adding new operations requires only trait extension:
//!
//! ```rust
//! use symbolic_math::final_tagless::*;
//! use num_traits::Float;
//!
//! // Extend with hyperbolic functions
//! trait HyperbolicExpr: MathExpr {
//!     fn tanh<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
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
//! # #[cfg(feature = "jit")]
//! impl HyperbolicExpr for JITEval {}
//! ```

use num_traits::Float;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "jit")]
use cranelift_codegen::ir::{InstBuilder, Value};
#[cfg(feature = "jit")]
use cranelift_frontend::FunctionBuilder;

#[cfg(feature = "jit")]
use crate::jit::{GeneralJITCompiler, GeneralJITFunction, JITError};

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
/// use symbolic_math::final_tagless::{DirectEval, MathExpr};
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
/// # use symbolic_math::final_tagless::{DirectEval, MathExpr};
/// // Function that works with any numeric type
/// fn linear<E: MathExpr, T>(x: E::Repr<T>, slope: T, intercept: T) -> E::Repr<T>
/// where
///     T: Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + crate::final_tagless::NumericType,
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
/// DirectEval is particularly useful for testing the correctness of expressions
/// before using them with other interpreters:
///
/// ```rust
/// # use symbolic_math::final_tagless::{DirectEval, MathExpr, StatisticalExpr};
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
/// use symbolic_math::final_tagless::{PrettyPrint, MathExpr};
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
/// # use symbolic_math::final_tagless::{PrettyPrint, MathExpr, StatisticalExpr};
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
/// # use symbolic_math::final_tagless::{PrettyPrint, MathExpr};
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

/// JIT compilation interpreter for final tagless expressions
///
/// This interpreter compiles final tagless expressions directly to native machine code using
/// the Cranelift code generator. It builds a computation graph (`JITRepr`) during expression
/// construction, then generates optimized native code for repeated evaluation.
///
/// # Compilation Process
///
/// 1. **Expression building**: Creates a `JITRepr` computation graph
/// 2. **Variable analysis**: Automatically detects variables and their usage patterns
/// 3. **IR generation**: Converts the graph to Cranelift intermediate representation
/// 4. **Native compilation**: Generates optimized machine code
/// 5. **Function creation**: Returns a callable native function
///
/// # Compilation Signatures
///
/// The JIT compiler supports multiple function signatures:
///
/// - **Single variable**: `f(x) -> f64`
/// - **Data + parameter**: `f(data, param) -> f64`
/// - **Data + multiple parameters**: `f(data, param1, param2, ...) -> f64`
/// - **Custom signatures**: Explicit variable classification
/// - **Embedded constants**: Compile-time constant folding
///
/// # Usage Examples
///
/// ## Basic Single Variable Function
///
/// ```rust
/// # #[cfg(feature = "jit")]
/// # {
/// use symbolic_math::final_tagless::{JITEval, MathExpr};
///
/// // Define expression: x² + 2x + 1
/// fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     let x_squared = E::pow(x.clone(), E::constant(2.0));
///     let two_x = E::mul(E::constant(2.0), x);
///     E::add(E::add(x_squared, two_x), E::constant(1.0))
/// }
///
/// // Compile to native code
/// let jit_expr = quadratic::<JITEval>(JITEval::var("x"));
/// let compiled = JITEval::compile_single_var(jit_expr, "x")?;
///
/// // Call compiled function
/// let result = compiled.call_single(3.0); // (3² + 2×3 + 1) = 16
/// assert_eq!(result, 16.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Data and Parameter Function
///
/// ```rust
/// # #[cfg(feature = "jit")]
/// # {
/// # use symbolic_math::final_tagless::{JITEval, MathExpr};
/// // Define expression: θ × data + 1
/// fn linear_model<E: MathExpr>(data: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
///     E::add(E::mul(theta, data), E::constant(1.0))
/// }
///
/// let jit_expr = linear_model::<JITEval>(JITEval::var("data"), JITEval::var("theta"));
/// let compiled = JITEval::compile_data_param(jit_expr, "data", "theta")?;
///
/// let result = compiled.call_data_param(5.0, 2.0); // 2×5 + 1 = 11
/// assert_eq!(result, 11.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Embedded Constants for Optimization
///
/// ```rust
/// # #[cfg(feature = "jit")]
/// # {
/// # use symbolic_math::final_tagless::{JITEval, MathExpr};
/// # use std::collections::HashMap;
/// // Expression with mathematical constants
/// fn circle_area<E: MathExpr>(radius: E::Repr<f64>, pi: E::Repr<f64>) -> E::Repr<f64> {
///     E::mul(pi, E::pow(radius, E::constant(2.0)))
/// }
///
/// let jit_expr = circle_area::<JITEval>(JITEval::var("radius"), JITEval::var("pi"));
///
/// // Embed π as a compile-time constant
/// let mut constants = HashMap::new();
/// constants.insert("pi".to_string(), std::f64::consts::PI);
///
/// let compiled = JITEval::compile_with_constants(
///     jit_expr,
///     &["radius".to_string()],
///     &[],
///     &constants
/// )?;
///
/// let result = compiled.call_single(2.0); // π × 2² ≈ 12.566
/// assert!((result - std::f64::consts::PI * 4.0).abs() < 1e-10);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "jit")]
pub struct JITEval;

#[cfg(feature = "jit")]
impl JITEval {
    /// Create a variable for JIT compilation
    #[must_use]
    pub fn var(name: &str) -> JITRepr {
        JITRepr::Variable(name.to_string())
    }
}

/// Internal representation for JIT compilation
/// This tracks the computation graph for direct Cranelift IR generation
#[cfg(feature = "jit")]
#[derive(Debug, Clone)]
pub enum JITRepr {
    /// Constant value
    Constant(f64),
    /// Variable reference by name
    Variable(String),
    /// Addition operation
    Add(Box<JITRepr>, Box<JITRepr>),
    /// Subtraction operation
    Sub(Box<JITRepr>, Box<JITRepr>),
    /// Multiplication operation
    Mul(Box<JITRepr>, Box<JITRepr>),
    /// Division operation
    Div(Box<JITRepr>, Box<JITRepr>),
    /// Power operation
    Pow(Box<JITRepr>, Box<JITRepr>),
    /// Negation operation
    Neg(Box<JITRepr>),
    /// Natural logarithm
    Ln(Box<JITRepr>),
    /// Exponential function
    Exp(Box<JITRepr>),
    /// Square root
    Sqrt(Box<JITRepr>),
    /// Sine function
    Sin(Box<JITRepr>),
    /// Cosine function
    Cos(Box<JITRepr>),
}

#[cfg(feature = "jit")]
impl JITRepr {
    /// Collect all variables in the expression
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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

    /// Generate Cranelift IR for this expression
    pub fn generate_ir(
        &self,
        builder: &mut FunctionBuilder,
        var_map: &HashMap<String, Value>,
    ) -> Result<Value, JITError> {
        match self {
            Self::Constant(c) => {
                let val = builder.ins().f64const(*c);
                Ok(val)
            }
            Self::Variable(name) => var_map
                .get(name)
                .copied()
                .ok_or_else(|| JITError::CompilationError(format!("Unknown variable: {name}"))),
            Self::Add(left, right) => {
                let left_val = left.generate_ir(builder, var_map)?;
                let right_val = right.generate_ir(builder, var_map)?;
                Ok(builder.ins().fadd(left_val, right_val))
            }
            Self::Sub(left, right) => {
                let left_val = left.generate_ir(builder, var_map)?;
                let right_val = right.generate_ir(builder, var_map)?;
                Ok(builder.ins().fsub(left_val, right_val))
            }
            Self::Mul(left, right) => {
                let left_val = left.generate_ir(builder, var_map)?;
                let right_val = right.generate_ir(builder, var_map)?;
                Ok(builder.ins().fmul(left_val, right_val))
            }
            Self::Div(left, right) => {
                let left_val = left.generate_ir(builder, var_map)?;
                let right_val = right.generate_ir(builder, var_map)?;
                Ok(builder.ins().fdiv(left_val, right_val))
            }
            Self::Pow(base, exp) => {
                let base_val = base.generate_ir(builder, var_map)?;

                // Special case optimization: if exponent is a constant, use repeated multiplication
                match exp.as_ref() {
                    Self::Constant(2.0) => {
                        // x^2 = x * x (most accurate)
                        Ok(builder.ins().fmul(base_val, base_val))
                    }
                    Self::Constant(3.0) => {
                        // x^3 = x * x * x
                        let x_squared = builder.ins().fmul(base_val, base_val);
                        Ok(builder.ins().fmul(x_squared, base_val))
                    }
                    Self::Constant(4.0) => {
                        // x^4 = (x^2)^2
                        let x_squared = builder.ins().fmul(base_val, base_val);
                        Ok(builder.ins().fmul(x_squared, x_squared))
                    }
                    Self::Constant(0.5) => {
                        // x^0.5 = sqrt(x)
                        Ok(builder.ins().sqrt(base_val))
                    }
                    Self::Constant(1.0) => {
                        // x^1 = x
                        Ok(base_val)
                    }
                    Self::Constant(0.0) => {
                        // x^0 = 1
                        Ok(builder.ins().f64const(1.0))
                    }
                    _ => {
                        // General case: use exp(exponent * ln(base)) with accurate implementations
                        let exp_val = exp.generate_ir(builder, var_map)?;
                        crate::jit::generate_pow_call(builder, base_val, exp_val)
                    }
                }
            }
            Self::Neg(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                Ok(builder.ins().fneg(inner_val))
            }
            Self::Ln(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                // Use the existing ln function from jit module
                crate::jit::generate_accurate_ln_call(builder, inner_val)
            }
            Self::Exp(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                // Use the existing exp function from jit module
                crate::jit::generate_exp_call(builder, inner_val)
            }
            Self::Sqrt(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                Ok(builder.ins().sqrt(inner_val))
            }
            Self::Sin(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                // Use the existing sin function from jit module
                crate::jit::generate_sin_call(builder, inner_val)
            }
            Self::Cos(inner) => {
                let inner_val = inner.generate_ir(builder, var_map)?;
                // Use the existing cos function from jit module
                crate::jit::generate_cos_call(builder, inner_val)
            }
        }
    }
}

#[cfg(feature = "jit")]
impl MathExpr for JITEval {
    type Repr<T> = JITRepr;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        // For JIT, we need to convert to f64
        // This is a limitation of the current JIT system
        let f64_value = format!("{value}")
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("Cannot convert {value} to f64 for JIT compilation"));
        JITRepr::Constant(f64_value)
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        JITRepr::Variable(name.to_string())
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITRepr::Add(Box::new(left), Box::new(right))
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITRepr::Sub(Box::new(left), Box::new(right))
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITRepr::Mul(Box::new(left), Box::new(right))
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        JITRepr::Div(Box::new(left), Box::new(right))
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Neg(Box::new(expr))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Ln(Box::new(expr))
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Exp(Box::new(expr))
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Sqrt(Box::new(expr))
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Sin(Box::new(expr))
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        JITRepr::Cos(Box::new(expr))
    }
}

#[cfg(feature = "jit")]
impl JITEval {
    /// Compile expression with automatic signature detection
    pub fn compile(expr: JITRepr) -> Result<GeneralJITFunction, JITError> {
        let variables = expr.collect_variables();
        let var_list: Vec<String> = variables.into_iter().collect();

        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(&expr, &var_list, &[], &HashMap::new())
    }

    /// Compile with explicit variable classification
    pub fn compile_with_signature(
        expr: JITRepr,
        data_vars: &[String],
        param_vars: &[String],
    ) -> Result<GeneralJITFunction, JITError> {
        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(&expr, data_vars, param_vars, &HashMap::new())
    }

    /// Compile single variable function: f(x)
    pub fn compile_single_var(
        expr: JITRepr,
        var_name: &str,
    ) -> Result<GeneralJITFunction, JITError> {
        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(&expr, &[var_name.to_string()], &[], &HashMap::new())
    }

    /// Compile data + single parameter function: f(x, θ)
    pub fn compile_data_param(
        expr: JITRepr,
        data_var: &str,
        param_var: &str,
    ) -> Result<GeneralJITFunction, JITError> {
        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(
            &expr,
            &[data_var.to_string()],
            &[param_var.to_string()],
            &HashMap::new(),
        )
    }

    /// Compile data + multiple parameters function: f(x, θ₁, θ₂, ...)
    pub fn compile_data_params(
        expr: JITRepr,
        data_var: &str,
        param_vars: &[String],
    ) -> Result<GeneralJITFunction, JITError> {
        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(&expr, &[data_var.to_string()], param_vars, &HashMap::new())
    }

    /// Compile with embedded constants for performance
    pub fn compile_with_constants(
        expr: JITRepr,
        data_vars: &[String],
        param_vars: &[String],
        constants: &HashMap<String, f64>,
    ) -> Result<GeneralJITFunction, JITError> {
        let compiler = GeneralJITCompiler::new()?;
        compiler.compile_final_tagless(&expr, data_vars, param_vars, constants)
    }
}

// Implement StatisticalExpr for JITEval
#[cfg(feature = "jit")]
impl StatisticalExpr for JITEval {}

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
    #[cfg(feature = "jit")]
    fn test_jit_eval_basic() {
        // Test: x + 1
        fn simple_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(x, E::constant(1.0))
        }

        let jit_expr = simple_expr::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();
        let result = compiled.call_single(5.0);
        assert_eq!(result, 6.0);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_complex() {
        // Test: x^2 + 2*x + 1 (should equal (x+1)^2)
        fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
        where
            E::Repr<f64>: Clone,
        {
            let two = E::constant(2.0);
            let one = E::constant(1.0);
            let x_squared = E::pow(x.clone(), E::constant(2.0));
            let two_x = E::mul(two, x);
            E::add(E::add(x_squared, two_x), one)
        }

        let jit_expr = quadratic::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();

        // Test several values
        for x in [0.0, 1.0, 2.0, -1.0, 3.5] {
            let result = compiled.call_single(x);
            let expected = (x + 1.0).powi(2);
            assert!(
                (result - expected).abs() < 1e-10,
                "x={x}, result={result}, expected={expected}"
            );
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_transcendental() {
        // Test: exp(ln(x)) should equal x
        fn identity<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::exp(E::ln(x))
        }

        let jit_expr = identity::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();

        // Test several positive values (ln is only defined for positive numbers)
        for x in [0.1, 1.0, 2.0, 5.0, 10.0] {
            let result = compiled.call_single(x);
            assert!((result - x).abs() < 1e-10, "x={x}, result={result}");
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_single_var() {
        // Test single variable compilation: f(x) = x^2 + 1
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::pow(x, E::constant(2.0)), E::constant(1.0))
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();

        let result = compiled.call_single(3.0);
        assert_eq!(result, 10.0); // 3^2 + 1 = 10
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_data_param() {
        // Test data+parameter compilation: f(x, θ) = θ * x + 1
        fn test_expr<E: MathExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::mul(theta, x), E::constant(1.0))
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var("x"), JITEval::var("theta"));
        let compiled = JITEval::compile_data_param(jit_expr, "x", "theta").unwrap();

        let result = compiled.call_data_param(2.0, 3.0);
        assert_eq!(result, 7.0); // 3 * 2 + 1 = 7
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_data_params() {
        // Test data+parameters compilation: f(x, θ₁, θ₂) = θ₁ * x + θ₂
        fn test_expr<E: MathExpr>(
            x: E::Repr<f64>,
            theta1: E::Repr<f64>,
            theta2: E::Repr<f64>,
        ) -> E::Repr<f64> {
            E::add(E::mul(theta1, x), theta2)
        }

        let jit_expr = test_expr::<JITEval>(
            JITEval::var("x"),
            JITEval::var("theta1"),
            JITEval::var("theta2"),
        );
        let compiled = JITEval::compile_data_params(
            jit_expr,
            "x",
            &["theta1".to_string(), "theta2".to_string()],
        )
        .unwrap();

        let result = compiled.call_data_params(2.0, &[3.0, 4.0]);
        assert_eq!(result, 10.0); // 3 * 2 + 4 = 10
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_with_constants() {
        // Test compilation with embedded constants: f(x) = π * x^2
        fn test_expr<E: MathExpr>(x: E::Repr<f64>, pi: E::Repr<f64>) -> E::Repr<f64> {
            E::mul(pi, E::pow(x, E::constant(2.0)))
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var("x"), JITEval::var("pi"));
        let mut constants = HashMap::new();
        constants.insert("pi".to_string(), std::f64::consts::PI);

        let compiled =
            JITEval::compile_with_constants(jit_expr, &["x".to_string()], &[], &constants).unwrap();

        let result = compiled.call_single(2.0);
        let expected = std::f64::consts::PI * 4.0; // π * 2^2
        println!(
            "Debug: result={}, expected={}, diff={}",
            result,
            expected,
            (result - expected).abs()
        );
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_eval_compilation_stats() {
        // Test that compilation statistics are properly generated
        fn simple_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(x, E::constant(1.0))
        }

        let jit_expr = simple_expr::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();

        // Verify the function works
        let result = compiled.call_single(5.0);
        assert_eq!(result, 6.0);

        // The compilation should have succeeded without errors
        // Additional stats testing would require access to internal compilation metrics
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
        assert!(pretty.contains("x"));
        assert!(pretty.contains("2"));
        assert!(pretty.contains("3"));
        assert!(pretty.contains("1"));
        assert!(pretty.contains("^"));
        assert!(pretty.contains("*"));
        assert!(pretty.contains("+"));

        // Test transcendental functions
        fn transcendental<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::exp(E::ln(x))
        }

        let pretty_trans = transcendental::<PrettyPrint>(PrettyPrint::var("x"));
        assert!(pretty_trans.contains("exp"));
        assert!(pretty_trans.contains("ln"));
        assert!(pretty_trans.contains("x"));
    }
}
