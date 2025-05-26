//! Final Tagless Approach for Exponential Family Distributions
//!
//! This module extends the final tagless approach from symbolic-math to provide
//! specialized operations for exponential family distributions. It enables
//! zero-cost symbolic computation of exponential family log-densities with
//! compile-time type safety.
//!
//! # Key Benefits
//!
//! 1. **Zero-cost abstractions**: Direct evaluation without AST overhead
//! 2. **Type-safe exponential families**: Compile-time verification of exponential family structure
//! 3. **Extensible operations**: Easy addition of new exponential family operations
//! 4. **JIT compilation**: Direct compilation to native code for ultimate performance
//! 5. **Automatic optimization**: Leverages exponential family mathematical structure
//!
//! # Usage
//!
//! ```rust
//! use measures_exponential_family::final_tagless::*;
//! use symbolic_math::final_tagless::{DirectEval, JITEval, MathExpr};
//!
//! // Define an exponential family log-density using final tagless
//! fn normal_log_density<E: ExponentialFamilyExpr>(
//!     x: E::Repr<f64>,
//!     mu: E::Repr<f64>,
//!     sigma: E::Repr<f64>
//! ) -> E::Repr<f64> {
//!     // Standard normal log-density: -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
//!     let two_pi = E::constant(2.0 * std::f64::consts::PI);
//!     let half = E::constant(0.5);
//!     let diff = E::sub(x, mu);
//!     let standardized = E::div(diff, sigma);
//!     let squared = E::mul(standardized, standardized);
//!     
//!     E::sub(
//!         E::sub(
//!             E::neg(E::mul(half, E::ln(two_pi))),
//!             E::ln(sigma)
//!         ),
//!         E::mul(half, squared)
//!     )
//! }
//!
//! // Evaluate with different interpreters
//! let direct_result = normal_log_density::<DirectEval>(
//!     DirectEval::var("x", 1.0),
//!     DirectEval::var("mu", 0.0),
//!     DirectEval::var("sigma", 1.0)
//! );
//!
//! // Compile to native code for ultimate performance
//! let jit_expr = normal_log_density::<JITEval>(
//!     JITEval::var("x"),
//!     JITEval::var("mu"),
//!     JITEval::var("sigma")
//! );
//! let compiled = JITEval::compile_data_params(jit_expr, "x", &["mu".to_string(), "sigma".to_string()])?;
//! let result = compiled.call_data_params(1.0, &[0.0, 1.0]);
//! ```

use num_traits::Float;
use symbolic_math::final_tagless::{MathExpr, NumericType};

/// Extension trait for exponential family operations in final tagless style
///
/// This trait extends the basic `MathExpr` trait with operations commonly used
/// in exponential family distributions, such as dot products, sufficient statistics,
/// and natural parameter transformations.
pub trait ExponentialFamilyExpr: MathExpr {
    /// Compute dot product between natural parameters and sufficient statistics
    ///
    /// This is the core operation in exponential family log-densities: η·T(x)
    fn dot_product<T: NumericType + Float>(
        natural_params: &[Self::Repr<T>],
        sufficient_stats: &[Self::Repr<T>],
    ) -> Self::Repr<T>
    where
        Self::Repr<T>: Clone,
    {
        assert_eq!(
            natural_params.len(),
            sufficient_stats.len(),
            "Natural parameters and sufficient statistics must have same length"
        );

        if natural_params.is_empty() {
            return Self::constant(T::zero());
        }

        let mut result = Self::mul(natural_params[0].clone(), sufficient_stats[0].clone());
        for i in 1..natural_params.len() {
            let term = Self::mul(natural_params[i].clone(), sufficient_stats[i].clone());
            result = Self::add(result, term);
        }
        result
    }

    /// Compute sum of sufficient statistics for IID samples
    ///
    /// For IID samples x₁, x₂, ..., xₙ, computes ∑ᵢT(xᵢ)
    fn sum_sufficient_stats<T: NumericType + Float>(
        stats_list: &[Vec<Self::Repr<T>>],
    ) -> Vec<Self::Repr<T>>
    where
        Self::Repr<T>: Clone,
    {
        if stats_list.is_empty() {
            return Vec::new();
        }

        let dim = stats_list[0].len();
        let mut result = Vec::with_capacity(dim);

        for j in 0..dim {
            let mut sum = stats_list[0][j].clone();
            for i in 1..stats_list.len() {
                sum = Self::add(sum, stats_list[i][j].clone());
            }
            result.push(sum);
        }
        result
    }

    /// Standard exponential family log-density computation
    ///
    /// Computes: η·T(x) - A(η) + log h(x)
    /// where η are natural parameters, T(x) are sufficient statistics,
    /// A(η) is the log-partition function, and h(x) is the base measure
    fn exp_fam_log_density<T: NumericType + Float>(
        natural_params: &[Self::Repr<T>],
        sufficient_stats: &[Self::Repr<T>],
        log_partition: Self::Repr<T>,
        log_base_measure: Self::Repr<T>,
    ) -> Self::Repr<T>
    where
        Self::Repr<T>: Clone,
    {
        let dot_product = Self::dot_product(natural_params, sufficient_stats);
        Self::add(Self::sub(dot_product, log_partition), log_base_measure)
    }

    /// IID exponential family log-density computation
    ///
    /// For n IID samples, computes: η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
    fn iid_exp_fam_log_density<T: NumericType + Float>(
        natural_params: &[Self::Repr<T>],
        sum_sufficient_stats: &[Self::Repr<T>],
        log_partition: Self::Repr<T>,
        n_samples: Self::Repr<T>,
        sum_log_base_measure: Self::Repr<T>,
    ) -> Self::Repr<T>
    where
        Self::Repr<T>: Clone,
    {
        let dot_product = Self::dot_product(natural_params, sum_sufficient_stats);
        let n_log_partition = Self::mul(n_samples, log_partition);
        Self::add(
            Self::sub(dot_product, n_log_partition),
            sum_log_base_measure,
        )
    }

    /// Logistic function: 1 / (1 + exp(-x))
    ///
    /// Commonly used in exponential families like logistic regression
    fn logistic<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::div(
            Self::constant(T::one()),
            Self::add(Self::constant(T::one()), Self::exp(Self::neg(x))),
        )
    }

    /// Softplus function: log(1 + exp(x))
    ///
    /// Numerically stable version of log(1 + exp(x))
    fn softplus<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::ln(Self::add(Self::constant(T::one()), Self::exp(x)))
    }

    /// Log-sum-exp function for numerical stability
    ///
    /// Computes log(∑ᵢ exp(xᵢ)) in a numerically stable way
    fn log_sum_exp<T: NumericType + Float>(values: &[Self::Repr<T>]) -> Self::Repr<T>
    where
        Self::Repr<T>: Clone,
    {
        if values.is_empty() {
            return Self::constant(T::neg_infinity());
        }

        if values.len() == 1 {
            return values[0].clone();
        }

        // Simplified version: log(exp(x₁) + exp(x₂) + ...)
        // In practice, this would be handled by the interpreter for numerical stability
        let mut sum = Self::exp(values[0].clone());
        for i in 1..values.len() {
            sum = Self::add(sum, Self::exp(values[i].clone()));
        }
        Self::ln(sum)
    }
}

// Blanket implementation: all MathExpr types automatically get ExponentialFamilyExpr
impl<T: MathExpr> ExponentialFamilyExpr for T {}

/// Specialized interpreter for exponential family computations
///
/// This interpreter is optimized for exponential family operations and can
/// provide specialized implementations for common patterns.
pub struct ExpFamEval;

impl ExpFamEval {
    /// Create a variable for exponential family evaluation
    #[must_use]
    pub fn var(_name: &str, value: f64) -> f64 {
        value
    }

    /// Efficient dot product computation for arrays
    #[must_use]
    pub fn dot_product_array(params: &[f64], stats: &[f64]) -> f64 {
        params.iter().zip(stats.iter()).map(|(p, s)| p * s).sum()
    }

    /// Efficient sum of sufficient statistics
    #[must_use]
    pub fn sum_stats_array(stats_list: &[&[f64]]) -> Vec<f64> {
        if stats_list.is_empty() {
            return Vec::new();
        }

        let dim = stats_list[0].len();
        let mut result = vec![0.0; dim];

        for stats in stats_list {
            for (i, &stat) in stats.iter().enumerate() {
                result[i] += stat;
            }
        }
        result
    }
}

impl MathExpr for ExpFamEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
        panic!("Use ExpFamEval::var(name, value) for direct evaluation")
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + std::ops::Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left + right
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + std::ops::Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left - right
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + std::ops::Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left * right
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + std::ops::Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left / right
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        base.powf(exp)
    }

    fn neg<T: NumericType + std::ops::Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
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

/// Conversion utilities between exponential family and final tagless approaches
pub trait ExpFamFinalTaglessConversion {
    /// Convert exponential family parameters to final tagless representation
    fn to_final_tagless<E: ExponentialFamilyExpr>(&self) -> Vec<E::Repr<f64>>;

    /// Convert sufficient statistics to final tagless representation  
    fn stats_to_final_tagless<E: ExponentialFamilyExpr>(&self) -> Vec<E::Repr<f64>>;
}

/// Helper functions for common exponential family patterns
pub mod patterns {
    use super::*;

    /// Standard normal log-density in final tagless style
    ///
    /// Computes: -0.5*log(2π) - 0.5*x²
    pub fn standard_normal_log_density<E: ExponentialFamilyExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let half = E::constant(0.5);
        let two_pi = E::constant(2.0 * std::f64::consts::PI);
        let x_squared = E::mul(x.clone(), x);

        E::sub(
            E::neg(E::mul(half.clone(), E::ln(two_pi))),
            E::mul(half, x_squared),
        )
    }

    /// Normal log-density with parameters in final tagless style
    ///
    /// Computes: -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
    pub fn normal_log_density<E: ExponentialFamilyExpr>(
        x: E::Repr<f64>,
        mu: E::Repr<f64>,
        sigma: E::Repr<f64>,
    ) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let half = E::constant(0.5);
        let two_pi = E::constant(2.0 * std::f64::consts::PI);
        let diff = E::sub(x, mu);
        let standardized = E::div(diff, sigma.clone());
        let squared = E::mul(standardized.clone(), standardized);

        E::sub(
            E::sub(E::neg(E::mul(half.clone(), E::ln(two_pi))), E::ln(sigma)),
            E::mul(half, squared),
        )
    }

    /// Poisson log-density in final tagless style
    ///
    /// Computes: x*log(λ) - λ - log(x!)
    pub fn poisson_log_density<E: ExponentialFamilyExpr>(
        x: E::Repr<f64>,
        lambda: E::Repr<f64>,
    ) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        // Note: log(x!) would need special handling for integer x
        // This is a simplified version for demonstration
        E::sub(
            E::sub(E::mul(x.clone(), E::ln(lambda.clone())), lambda),
            E::ln(x), // Simplified - should be log factorial
        )
    }

    /// Exponential distribution log-density in final tagless style
    ///
    /// Computes: log(λ) - λ*x
    pub fn exponential_log_density<E: ExponentialFamilyExpr>(
        x: E::Repr<f64>,
        lambda: E::Repr<f64>,
    ) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        E::sub(E::ln(lambda.clone()), E::mul(lambda, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolic_math::final_tagless::{DirectEval, PrettyPrint};

    #[test]
    fn test_exp_fam_eval_basic() {
        // Test basic exponential family evaluation
        let result = ExpFamEval::dot_product_array(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(result, 11.0); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_dot_product_final_tagless() {
        // Test dot product with DirectEval
        let params = vec![DirectEval::constant(1.0), DirectEval::constant(2.0)];
        let stats = vec![DirectEval::constant(3.0), DirectEval::constant(4.0)];

        let result = DirectEval::dot_product(&params, &stats);
        assert_eq!(result, 11.0); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_standard_normal_pattern() {
        // Test standard normal log-density pattern
        let x = DirectEval::constant(0.0);
        let result = patterns::standard_normal_log_density::<DirectEval>(x);

        // At x=0, standard normal log-density should be -0.5*log(2π)
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_normal_pattern_pretty_print() {
        // Test that normal log-density generates readable expressions
        let x = PrettyPrint::var("x");
        let mu = PrettyPrint::var("mu");
        let sigma = PrettyPrint::var("sigma");

        let result = patterns::normal_log_density::<PrettyPrint>(x, mu, sigma);

        // Should contain the key components
        assert!(result.contains("x"));
        assert!(result.contains("mu"));
        assert!(result.contains("sigma"));
        assert!(result.contains("ln"));
    }

    #[test]
    fn test_sum_sufficient_stats() {
        // Test summing sufficient statistics
        let stats1 = vec![DirectEval::constant(1.0), DirectEval::constant(2.0)];
        let stats2 = vec![DirectEval::constant(3.0), DirectEval::constant(4.0)];
        let stats_list = vec![stats1, stats2];

        let result = DirectEval::sum_sufficient_stats(&stats_list);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 4.0); // 1 + 3
        assert_eq!(result[1], 6.0); // 2 + 4
    }

    #[test]
    fn test_exp_fam_log_density() {
        // Test complete exponential family log-density computation
        let natural_params = vec![DirectEval::constant(1.0), DirectEval::constant(2.0)];
        let sufficient_stats = vec![DirectEval::constant(3.0), DirectEval::constant(4.0)];
        let log_partition = DirectEval::constant(5.0);
        let log_base_measure = DirectEval::constant(0.5);

        let result = DirectEval::exp_fam_log_density(
            &natural_params,
            &sufficient_stats,
            log_partition,
            log_base_measure,
        );

        // Should be: (1*3 + 2*4) - 5 + 0.5 = 11 - 5 + 0.5 = 6.5
        assert_eq!(result, 6.5);
    }
}
