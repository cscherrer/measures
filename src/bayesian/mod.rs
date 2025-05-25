//! Bayesian Inference and Modeling
//!
//! This module provides specialized functionality for Bayesian statistical modeling,
//! built on top of the general symbolic IR system. It includes tools for:
//!
//! - Posterior log-density compilation
//! - MCMC optimization
//! - Variational inference support
//! - Model comparison utilities
//!
//! ## Features
//!
//! - **Posterior Compilation**: JIT-compile Bayesian posteriors for fast inference
//! - **Likelihood + Prior**: Automatic combination of likelihood and prior expressions
//! - **Parameter Inference**: Support for variable parameters in addition to data
//! - **Performance Optimization**: Native speed for MCMC and optimization algorithms
//!
//! ## Usage Example
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use measures::bayesian::BayesianJITOptimizer;
//! use measures::symbolic_ir::expr::Expr;
//! use std::collections::HashMap;
//!
//! // Define a simple Bayesian model
//! struct SimpleModel {
//!     data: Vec<f64>,
//! }
//!
//! impl BayesianJITOptimizer for SimpleModel {
//!     fn compile_posterior_jit(
//!         &self,
//!         data: &[f64],
//!     ) -> Result<measures::symbolic_ir::jit::GeneralJITFunction, measures::symbolic_ir::jit::JITError> {
//!         // Implementation would build likelihood + prior expressions
//!         // and compile them to native code
//!         todo!()
//!     }
//!     
//!     fn compile_likelihood_jit(&self) -> Result<measures::symbolic_ir::jit::GeneralJITFunction, measures::symbolic_ir::jit::JITError> {
//!         todo!()
//!     }
//!     
//!     fn compile_prior_jit(&self) -> Result<measures::symbolic_ir::jit::GeneralJITFunction, measures::symbolic_ir::jit::JITError> {
//!         todo!()
//!     }
//! }
//! # }
//! ```

#[cfg(feature = "jit")]
use crate::symbolic_ir::jit::{GeneralJITFunction, JITError};

/// Trait for Bayesian models that can be JIT-compiled for fast inference
#[cfg(feature = "jit")]
pub trait BayesianJITOptimizer {
    /// Compile the posterior log-density function: log p(θ|x) ∝ log p(x|θ) + log p(θ)
    fn compile_posterior_jit(
        &self,
        data: &[f64],
    ) -> Result<GeneralJITFunction, JITError>;
    
    /// Compile the likelihood function with variable parameters: log p(x|θ)
    fn compile_likelihood_jit(&self) -> Result<GeneralJITFunction, JITError>;
    
    /// Compile the prior log-density function: log p(θ)
    fn compile_prior_jit(&self) -> Result<GeneralJITFunction, JITError>;
}

/// Utilities for building Bayesian model expressions
pub mod expressions {
    use crate::symbolic_ir::expr::Expr;
    
    /// Build a normal likelihood expression: log p(x|μ,σ) = -½((x-μ)/σ)² - log(σ√(2π))
    pub fn normal_likelihood(x_var: &str, mu_var: &str, sigma_var: &str) -> Expr {
        let x = Expr::Var(x_var.to_string());
        let mu = Expr::Var(mu_var.to_string());
        let sigma = Expr::Var(sigma_var.to_string());
        
        // (x - μ)
        let diff = Expr::Sub(Box::new(x), Box::new(mu));
        
        // (x - μ) / σ
        let standardized = Expr::Div(Box::new(diff), Box::new(sigma.clone()));
        
        // ((x - μ) / σ)²
        let squared = Expr::Pow(Box::new(standardized), Box::new(Expr::Const(2.0)));
        
        // -½((x - μ) / σ)²
        let quadratic_term = Expr::Mul(Box::new(Expr::Const(-0.5)), Box::new(squared));
        
        // log(σ)
        let log_sigma = Expr::Ln(Box::new(sigma));
        
        // log(√(2π)) = ½log(2π)
        let log_sqrt_2pi = Expr::Const(0.5 * (2.0 * std::f64::consts::PI).ln());
        
        // -log(σ√(2π)) = -log(σ) - log(√(2π))
        let normalization = Expr::Sub(
            Box::new(Expr::Neg(Box::new(log_sigma))),
            Box::new(log_sqrt_2pi)
        );
        
        // Final: -½((x-μ)/σ)² - log(σ√(2π))
        Expr::Add(Box::new(quadratic_term), Box::new(normalization))
    }
    
    /// Build a normal prior expression: log p(μ) = -½((μ-μ₀)/σ₀)² - log(σ₀√(2π))
    pub fn normal_prior(param_var: &str, prior_mean: f64, prior_std: f64) -> Expr {
        let param = Expr::Var(param_var.to_string());
        let prior_mean_expr = Expr::Const(prior_mean);
        let prior_std_expr = Expr::Const(prior_std);
        
        // (μ - μ₀)
        let diff = Expr::Sub(Box::new(param), Box::new(prior_mean_expr));
        
        // (μ - μ₀) / σ₀
        let standardized = Expr::Div(Box::new(diff), Box::new(prior_std_expr));
        
        // ((μ - μ₀) / σ₀)²
        let squared = Expr::Pow(Box::new(standardized), Box::new(Expr::Const(2.0)));
        
        // -½((μ - μ₀) / σ₀)²
        let quadratic_term = Expr::Mul(Box::new(Expr::Const(-0.5)), Box::new(squared));
        
        // Normalization constant (can be omitted for MCMC)
        let log_normalization = Expr::Const(-0.5 * (2.0 * std::f64::consts::PI * prior_std * prior_std).ln());
        
        Expr::Add(Box::new(quadratic_term), Box::new(log_normalization))
    }
    
    /// Combine likelihood and prior into posterior (up to normalization constant)
    pub fn posterior_log_density(likelihood: Expr, prior: Expr) -> Expr {
        Expr::Add(Box::new(likelihood), Box::new(prior))
    }
}

#[cfg(test)]
mod tests {
    use super::expressions::*;
    
    #[test]
    fn test_normal_likelihood_expression() {
        let likelihood = normal_likelihood("x", "mu", "sigma");
        // Just test that it builds without panicking
        assert!(matches!(likelihood, crate::symbolic_ir::expr::Expr::Add(_, _)));
    }
    
    #[test]
    fn test_normal_prior_expression() {
        let prior = normal_prior("mu", 0.0, 1.0);
        assert!(matches!(prior, crate::symbolic_ir::expr::Expr::Add(_, _)));
    }
    
    #[test]
    fn test_posterior_combination() {
        let likelihood = normal_likelihood("x", "mu", "sigma");
        let prior = normal_prior("mu", 0.0, 1.0);
        let posterior = posterior_log_density(likelihood, prior);
        assert!(matches!(posterior, crate::symbolic_ir::expr::Expr::Add(_, _)));
    }
} 