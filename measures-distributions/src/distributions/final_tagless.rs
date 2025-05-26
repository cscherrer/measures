//! Final Tagless Approach for Probability Distributions
//!
//! This module extends the final tagless approach to provide specialized operations
//! for probability distributions. It enables zero-cost symbolic computation of
//! log-densities, CDFs, and other distribution operations with compile-time type safety.
//!
//! # Key Benefits
//!
//! 1. **Zero-cost abstractions**: Direct evaluation without AST overhead
//! 2. **Type-safe distributions**: Compile-time verification of distribution structure
//! 3. **Extensible operations**: Easy addition of new distribution operations
//! 4. **JIT compilation**: Direct compilation to native code for ultimate performance
//! 5. **Automatic optimization**: Leverages mathematical structure of distributions
//!
//! # Usage
//!
//! ```rust
//! use measures_distributions::final_tagless::*;
//! use symbolic_math::final_tagless::{DirectEval, JITEval, MathExpr};
//!
//! // Define normal log-density using final tagless
//! let x = DirectEval::var("x", 1.0);
//! let mu = DirectEval::var("mu", 0.0);
//! let sigma = DirectEval::var("sigma", 1.0);
//! let result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
//!
//! // Compile to native code for ultimate performance
//! let jit_expr = patterns::normal_log_density::<JITEval>(
//!     JITEval::var("x"),
//!     JITEval::var("mu"),
//!     JITEval::var("sigma")
//! );
//! let compiled = JITEval::compile_data_params(jit_expr, "x", &["mu".to_string(), "sigma".to_string()])?;
//! let result = compiled.call_data_params(1.0, &[0.0, 1.0]);
//! ```

use symbolic_math::final_tagless::{MathExpr, NumericType};
use num_traits::Float;

#[cfg(feature = "jit")]
use measures_exponential_family::final_tagless::ExponentialFamilyExpr;

/// Extension trait for probability distribution operations in final tagless style
///
/// This trait extends the basic `MathExpr` trait with operations commonly used
/// in probability distributions, such as log-densities, CDFs, quantiles, and
/// moment computations.
pub trait DistributionExpr: MathExpr {
    /// Compute log-density for a univariate distribution
    /// 
    /// This is the fundamental operation for probability distributions
    fn log_density<T: NumericType + Float>(
        x: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute cumulative distribution function (CDF)
    /// 
    /// For many distributions, this requires numerical integration or special functions
    fn cdf<T: NumericType + Float>(
        x: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute survival function (1 - CDF)
    /// 
    /// Often more numerically stable than computing 1 - CDF directly
    fn survival<T: NumericType + Float>(
        x: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone
    {
        Self::sub(Self::constant(T::one()), Self::cdf(x, params))
    }
    
    /// Compute log-CDF (log of cumulative distribution function)
    /// 
    /// Numerically stable for very small probabilities
    fn log_cdf<T: NumericType + Float>(
        x: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone
    {
        Self::ln(Self::cdf(x, params))
    }
    
    /// Compute log survival function (log(1 - CDF))
    /// 
    /// Numerically stable for probabilities close to 1
    fn log_survival<T: NumericType + Float>(
        x: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone
    {
        Self::ln(Self::survival(x, params))
    }
    
    /// Compute moment generating function (MGF)
    /// 
    /// MGF(t) = E[exp(tX)] for random variable X
    fn mgf<T: NumericType + Float>(
        t: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute characteristic function
    /// 
    /// φ(t) = E[exp(itX)] for random variable X
    fn characteristic<T: NumericType + Float>(
        t: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute raw moment E[X^k]
    fn raw_moment<T: NumericType + Float>(
        k: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute central moment E[(X-μ)^k]
    fn central_moment<T: NumericType + Float>(
        k: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute standardized moment E[((X-μ)/σ)^k]
    fn standardized_moment<T: NumericType + Float>(
        k: Self::Repr<T>,
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute entropy H(X) = -E[log f(X)]
    fn entropy<T: NumericType + Float>(
        params: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute Kullback-Leibler divergence KL(P||Q)
    fn kl_divergence<T: NumericType + Float>(
        params_p: &[Self::Repr<T>],
        params_q: &[Self::Repr<T>]
    ) -> Self::Repr<T>
    where 
        Self::Repr<T>: Clone;
    
    /// Compute Fisher information matrix
    fn fisher_information<T: NumericType + Float>(
        params: &[Self::Repr<T>]
    ) -> Vec<Vec<Self::Repr<T>>>
    where 
        Self::Repr<T>: Clone;
}

// Blanket implementation: all MathExpr types automatically get DistributionExpr
// with default implementations that may need to be overridden for specific distributions
impl<T: MathExpr> DistributionExpr for T {
    fn log_density<U: NumericType + Float>(
        _x: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("log_density not implemented for this distribution")
    }
    
    fn cdf<U: NumericType + Float>(
        _x: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("cdf not implemented for this distribution")
    }
    
    fn mgf<U: NumericType + Float>(
        _t: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("mgf not implemented for this distribution")
    }
    
    fn characteristic<U: NumericType + Float>(
        _t: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("characteristic not implemented for this distribution")
    }
    
    fn raw_moment<U: NumericType + Float>(
        _k: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("raw_moment not implemented for this distribution")
    }
    
    fn central_moment<U: NumericType + Float>(
        _k: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("central_moment not implemented for this distribution")
    }
    
    fn standardized_moment<U: NumericType + Float>(
        _k: Self::Repr<U>,
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("standardized_moment not implemented for this distribution")
    }
    
    fn entropy<U: NumericType + Float>(
        _params: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("entropy not implemented for this distribution")
    }
    
    fn kl_divergence<U: NumericType + Float>(
        _params_p: &[Self::Repr<U>],
        _params_q: &[Self::Repr<U>]
    ) -> Self::Repr<U>
    where 
        Self::Repr<U>: Clone
    {
        panic!("kl_divergence not implemented for this distribution")
    }
    
    fn fisher_information<U: NumericType + Float>(
        _params: &[Self::Repr<U>]
    ) -> Vec<Vec<Self::Repr<U>>>
    where 
        Self::Repr<U>: Clone
    {
        panic!("fisher_information not implemented for this distribution")
    }
}

/// Specialized interpreter for distribution computations
/// 
/// This interpreter is optimized for distribution operations and can
/// provide specialized implementations for common patterns.
pub struct DistributionEval;

impl DistributionEval {
    /// Create a variable for distribution evaluation
    #[must_use]
    pub fn var(_name: &str, value: f64) -> f64 {
        value
    }
    
    /// Efficient log-density computation for normal distribution
    #[must_use]
    pub fn normal_log_density(x: f64, mu: f64, sigma: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let standardized = (x - mu) / sigma;
        -0.5 * two_pi.ln() - sigma.ln() - 0.5 * standardized * standardized
    }
    
    /// Efficient log-density computation for exponential distribution
    #[must_use]
    pub fn exponential_log_density(x: f64, rate: f64) -> f64 {
        if x >= 0.0 {
            rate.ln() - rate * x
        } else {
            f64::NEG_INFINITY
        }
    }
    
    /// Efficient log-density computation for gamma distribution
    #[must_use]
    pub fn gamma_log_density(x: f64, shape: f64, rate: f64) -> f64 {
        if x > 0.0 {
            shape * rate.ln() - Self::log_gamma(shape) + (shape - 1.0) * x.ln() - rate * x
        } else {
            f64::NEG_INFINITY
        }
    }
    
    /// Log-gamma function approximation (Stirling's approximation for large values)
    #[must_use]
    pub fn log_gamma(x: f64) -> f64 {
        if x > 12.0 {
            // Stirling's approximation for large x
            (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
        } else {
            // Use built-in gamma function for smaller values
            special::Gamma::ln_gamma(x).0
        }
    }
}

impl MathExpr for DistributionEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
        panic!("Use DistributionEval::var(name, value) for direct evaluation")
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

/// Helper functions for common distribution patterns
pub mod patterns {
    use super::*;
    
    /// Normal distribution log-density in final tagless style
    /// 
    /// Computes: -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
    pub fn normal_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        mu: E::Repr<f64>,
        sigma: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let half = E::constant(0.5);
        let two_pi = E::constant(2.0 * std::f64::consts::PI);
        let diff = E::sub(x, mu);
        let standardized = E::div(diff, sigma.clone());
        let squared = E::mul(standardized.clone(), standardized);
        
        E::sub(
            E::sub(
                E::neg(E::mul(half.clone(), E::ln(two_pi))),
                E::ln(sigma)
            ),
            E::mul(half, squared)
        )
    }
    
    /// Standard normal log-density in final tagless style
    /// 
    /// Computes: -0.5*log(2π) - 0.5*x²
    pub fn standard_normal_log_density<E: DistributionExpr>(
        x: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let half = E::constant(0.5);
        let two_pi = E::constant(2.0 * std::f64::consts::PI);
        let x_squared = E::mul(x.clone(), x);
        
        E::sub(
            E::neg(E::mul(half.clone(), E::ln(two_pi))),
            E::mul(half, x_squared)
        )
    }
    
    /// Exponential distribution log-density in final tagless style
    /// 
    /// Computes: log(λ) - λ*x (for x ≥ 0)
    pub fn exponential_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        rate: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        E::sub(
            E::ln(rate.clone()),
            E::mul(rate, x)
        )
    }
    
    /// Gamma distribution log-density in final tagless style
    /// 
    /// Computes: α*log(β) - log(Γ(α)) + (α-1)*log(x) - β*x
    pub fn gamma_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        shape: E::Repr<f64>,
        rate: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let one = E::constant(1.0);
        let shape_minus_one = E::sub(shape.clone(), one);
        
        E::add(
            E::add(
                E::mul(shape.clone(), E::ln(rate.clone())),
                E::neg(E::ln(E::gamma_function(shape)))
            ),
            E::sub(
                E::mul(shape_minus_one, E::ln(x.clone())),
                E::mul(rate, x)
            )
        )
    }
    
    /// Beta distribution log-density in final tagless style
    /// 
    /// Computes: log(B(α,β)) + (α-1)*log(x) + (β-1)*log(1-x)
    pub fn beta_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        alpha: E::Repr<f64>,
        beta: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let one = E::constant(1.0);
        let alpha_minus_one = E::sub(alpha.clone(), one.clone());
        let beta_minus_one = E::sub(beta.clone(), one.clone());
        let one_minus_x = E::sub(one, x.clone());
        
        E::add(
            E::add(
                E::log_beta(alpha, beta),
                E::mul(alpha_minus_one, E::ln(x))
            ),
            E::mul(beta_minus_one, E::ln(one_minus_x))
        )
    }
    
    /// Cauchy distribution log-density in final tagless style
    /// 
    /// Computes: -log(π) - log(γ) - log(1 + ((x-x₀)/γ)²)
    pub fn cauchy_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        location: E::Repr<f64>,
        scale: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let one = E::constant(1.0);
        let pi = E::constant(std::f64::consts::PI);
        let diff = E::sub(x, location);
        let standardized = E::div(diff, scale.clone());
        let squared = E::mul(standardized.clone(), standardized);
        let one_plus_squared = E::add(one, squared);
        
        E::sub(
            E::sub(
                E::neg(E::ln(pi)),
                E::ln(scale)
            ),
            E::ln(one_plus_squared)
        )
    }
    
    /// Student's t-distribution log-density in final tagless style
    /// 
    /// Computes: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(νπ) - ((ν+1)/2)*log(1 + x²/ν)
    pub fn student_t_log_density<E: DistributionExpr>(
        x: E::Repr<f64>,
        nu: E::Repr<f64>
    ) -> E::Repr<f64> 
    where 
        E::Repr<f64>: Clone
    {
        let one = E::constant(1.0);
        let two = E::constant(2.0);
        let half = E::constant(0.5);
        let pi = E::constant(std::f64::consts::PI);
        
        let nu_plus_one = E::add(nu.clone(), one.clone());
        let nu_plus_one_half = E::div(nu_plus_one.clone(), two.clone());
        let nu_half = E::div(nu.clone(), two.clone());
        let nu_pi = E::mul(nu.clone(), pi);
        let x_squared = E::mul(x.clone(), x);
        let x_squared_over_nu = E::div(x_squared, nu);
        let one_plus_ratio = E::add(one, x_squared_over_nu);
        
        E::sub(
            E::sub(
                E::sub(
                    E::ln(E::gamma_function(nu_plus_one_half)),
                    E::ln(E::gamma_function(nu_half))
                ),
                E::mul(half, E::ln(nu_pi))
            ),
            E::mul(
                E::div(nu_plus_one, two),
                E::ln(one_plus_ratio)
            )
        )
    }
}

// Extension methods for MathExpr to add distribution-specific functions
pub trait DistributionMathExpr: MathExpr {
    /// Gamma function Γ(x)
    fn gamma_function<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>;
    
    /// Log-gamma function log(Γ(x))
    fn log_gamma_function<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::ln(Self::gamma_function(x))
    }
    
    /// Beta function B(α,β) = Γ(α)Γ(β)/Γ(α+β)
    fn beta_function<T: NumericType + Float>(
        alpha: Self::Repr<T>, 
        beta: Self::Repr<T>
    ) -> Self::Repr<T> 
    where 
        Self::Repr<T>: Clone
    {
        let alpha_plus_beta = Self::add(alpha.clone(), beta.clone());
        Self::div(
            Self::mul(
                Self::gamma_function(alpha),
                Self::gamma_function(beta)
            ),
            Self::gamma_function(alpha_plus_beta)
        )
    }
    
    /// Log-beta function log(B(α,β))
    fn log_beta<T: NumericType + Float>(
        alpha: Self::Repr<T>, 
        beta: Self::Repr<T>
    ) -> Self::Repr<T> 
    where 
        Self::Repr<T>: Clone
    {
        let alpha_plus_beta = Self::add(alpha.clone(), beta.clone());
        Self::sub(
            Self::add(
                Self::log_gamma_function(alpha),
                Self::log_gamma_function(beta)
            ),
            Self::log_gamma_function(alpha_plus_beta)
        )
    }
    
    /// Error function erf(x)
    fn erf<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>;
    
    /// Complementary error function erfc(x) = 1 - erf(x)
    fn erfc<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::sub(Self::constant(T::one()), Self::erf(x))
    }
}

// Blanket implementation for DistributionMathExpr
impl<T: MathExpr> DistributionMathExpr for T {
    fn gamma_function<U: NumericType + Float>(_x: Self::Repr<U>) -> Self::Repr<U> {
        panic!("gamma_function not implemented for this interpreter")
    }
    
    fn erf<U: NumericType + Float>(_x: Self::Repr<U>) -> Self::Repr<U> {
        panic!("erf not implemented for this interpreter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolic_math::final_tagless::{DirectEval, PrettyPrint};
    
    #[test]
    fn test_distribution_eval_normal() {
        // Test normal log-density evaluation
        let result = DistributionEval::normal_log_density(0.0, 0.0, 1.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_distribution_eval_exponential() {
        // Test exponential log-density evaluation
        let result = DistributionEval::exponential_log_density(1.0, 2.0);
        let expected = 2.0_f64.ln() - 2.0;
        assert!((result - expected).abs() < 1e-10);
        
        // Test outside support
        let result = DistributionEval::exponential_log_density(-1.0, 2.0);
        assert_eq!(result, f64::NEG_INFINITY);
    }
    
    #[test]
    fn test_normal_pattern_direct_eval() {
        // Test normal log-density pattern with DirectEval
        let x = DirectEval::constant(0.0);
        let mu = DirectEval::constant(0.0);
        let sigma = DirectEval::constant(1.0);
        
        let result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_exponential_pattern_direct_eval() {
        // Test exponential log-density pattern with DirectEval
        let x = DirectEval::constant(1.0);
        let rate = DirectEval::constant(2.0);
        
        let result = patterns::exponential_log_density::<DirectEval>(x, rate);
        let expected = 2.0_f64.ln() - 2.0;
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
    fn test_standard_normal_pattern() {
        // Test standard normal log-density pattern
        let x = DirectEval::constant(1.0);
        let result = patterns::standard_normal_log_density::<DirectEval>(x);
        
        // At x=1, standard normal log-density should be -0.5*log(2π) - 0.5
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5;
        assert!((result - expected).abs() < 1e-10);
    }
} 