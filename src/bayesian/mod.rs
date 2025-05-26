//! Bayesian Inference and Modeling
//!
//! This module provides tools for Bayesian statistical modeling and inference,
//! including:
//!
//! - Expression building for Bayesian models
//! - Posterior composition (likelihood + prior)
//! - Hierarchical model support
//! - JIT compilation for Bayesian computations
//!
//! ## Quick Start
//!
//! ```rust
//! # #[cfg(feature = "symbolic")]
//! # {
//! use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};
//!
//! // Build Bayesian model expressions
//! let likelihood = normal_likelihood("x", "mu", "sigma");
//! let prior = normal_prior("mu", 0.0, 1.0);
//! let posterior = posterior_log_density(likelihood, prior);
//! # }
//! ```
//!
//! ## JIT Compilation
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use symbolic_math::{Expr, builders, display};
//! use measures::bayesian::BayesianJITOptimizer;
//!
//! // Build and compile Bayesian models for maximum performance
//! # }
//! ```

use std::collections::HashMap;

/// Bayesian model expressions and utilities
pub mod expressions {
    #[cfg(feature = "symbolic")]
    use symbolic_math::Expr;

    /// Create a normal likelihood expression: -0.5 * ln(2π) - ln(σ) - 0.5 * (x - μ)² / σ²
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn normal_likelihood(x_var: &str, mu_var: &str, sigma_var: &str) -> Expr {
        symbolic_math::builders::normal_log_pdf(
            Expr::variable(x_var),
            Expr::variable(mu_var),
            Expr::variable(sigma_var),
        )
    }

    /// Create a normal prior expression: -0.5 * ln(2π) - ln(σ₀) - 0.5 * (μ - μ₀)² / σ₀²
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn normal_prior(param_var: &str, prior_mean: f64, prior_std: f64) -> Expr {
        symbolic_math::builders::normal_log_pdf(
            Expr::variable(param_var),
            Expr::constant(prior_mean),
            Expr::constant(prior_std),
        )
    }

    /// Combine likelihood and prior into posterior log-density
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn posterior_log_density(likelihood: Expr, prior: Expr) -> Expr {
        Expr::add(likelihood, prior)
    }

    /// Create a hierarchical normal model
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn hierarchical_normal(
        x_var: &str,
        mu_var: &str,
        sigma_var: &str,
        tau_var: &str,
        mu_prior_mean: f64,
        tau_prior_scale: f64,
    ) -> Expr {
        let likelihood = normal_likelihood(x_var, mu_var, sigma_var);
        let mu_prior = normal_prior(mu_var, mu_prior_mean, tau_prior_scale);
        let tau_prior = normal_prior(tau_var, 0.0, tau_prior_scale);
        likelihood + mu_prior + tau_prior
    }

    /// Create a mixture model likelihood (legacy API for backward compatibility)
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn mixture_likelihood_old(components: &[(f64, Expr)]) -> Expr {
        if components.is_empty() {
            return Expr::constant(f64::NEG_INFINITY);
        }

        let mut log_sum_exp = Expr::constant(0.0);
        for (weight, component) in components {
            let log_weight = Expr::constant(weight.ln());
            let weighted_component = log_weight + component.clone();
            log_sum_exp = log_sum_exp + Expr::exp(weighted_component);
        }

        Expr::ln(log_sum_exp)
    }

    /// Create a mixture model likelihood with separate arrays (for test compatibility)
    #[cfg(feature = "symbolic")]
    #[must_use]
    pub fn mixture_likelihood(x_var: &str, weights: &[f64], means: &[f64], stds: &[f64]) -> Expr {
        assert_eq!(weights.len(), means.len());
        assert_eq!(means.len(), stds.len());

        if weights.is_empty() {
            return Expr::constant(f64::NEG_INFINITY);
        }

        let mut log_sum_exp = Expr::constant(0.0);
        for i in 0..weights.len() {
            let log_weight = Expr::constant(weights[i].ln());
            let component = symbolic_math::builders::normal_log_pdf(
                Expr::variable(x_var),
                Expr::constant(means[i]),
                Expr::constant(stds[i]),
            );
            let weighted_component = log_weight + component;
            log_sum_exp = log_sum_exp + Expr::exp(weighted_component);
        }

        Expr::ln(log_sum_exp)
    }
}

/// JIT compilation for Bayesian models
#[cfg(feature = "jit")]
pub struct BayesianJITOptimizer {
    /// Compiled posterior function
    posterior_fn: Option<symbolic_math::GeneralJITFunction>,
    /// Compiled likelihood function  
    likelihood_fn: Option<symbolic_math::GeneralJITFunction>,
    /// Compiled prior function
    prior_fn: Option<symbolic_math::GeneralJITFunction>,
}

#[cfg(feature = "jit")]
impl BayesianJITOptimizer {
    /// Create a new Bayesian JIT optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            posterior_fn: None,
            likelihood_fn: None,
            prior_fn: None,
        }
    }

    /// Compile the posterior log-density function
    pub fn compile_posterior_jit(
        &mut self,
        posterior_expr: &symbolic_math::Expr,
        data_vars: &[String],
        param_vars: &[String],
        constants: &HashMap<String, f64>,
    ) -> Result<(), symbolic_math::JITError> {
        let compiler = symbolic_math::GeneralJITCompiler::new()?;
        let jit_fn =
            compiler.compile_expression(posterior_expr, data_vars, param_vars, constants)?;
        self.posterior_fn = Some(jit_fn);
        Ok(())
    }

    /// Compile the likelihood function
    pub fn compile_likelihood_jit(
        &mut self,
        likelihood_expr: &symbolic_math::Expr,
        data_vars: &[String],
        param_vars: &[String],
        constants: &HashMap<String, f64>,
    ) -> Result<(), symbolic_math::JITError> {
        let compiler = symbolic_math::GeneralJITCompiler::new()?;
        let jit_fn =
            compiler.compile_expression(likelihood_expr, data_vars, param_vars, constants)?;
        self.likelihood_fn = Some(jit_fn);
        Ok(())
    }

    /// Compile the prior function
    pub fn compile_prior_jit(
        &mut self,
        prior_expr: &symbolic_math::Expr,
        param_vars: &[String],
        constants: &HashMap<String, f64>,
    ) -> Result<(), symbolic_math::JITError> {
        let compiler = symbolic_math::GeneralJITCompiler::new()?;
        let jit_fn = compiler.compile_expression(prior_expr, &[], param_vars, constants)?;
        self.prior_fn = Some(jit_fn);
        Ok(())
    }

    /// Evaluate the compiled posterior function
    pub fn evaluate_posterior(&self, data: &[f64], params: &[f64]) -> Option<f64> {
        self.posterior_fn
            .as_ref()
            .map(|f| f.call_batch(data, params))
    }

    /// Evaluate the compiled likelihood function
    pub fn evaluate_likelihood(&self, data: &[f64], params: &[f64]) -> Option<f64> {
        self.likelihood_fn
            .as_ref()
            .map(|f| f.call_batch(data, params))
    }

    /// Evaluate the compiled prior function
    pub fn evaluate_prior(&self, params: &[f64]) -> Option<f64> {
        self.prior_fn.as_ref().map(|f| f.call_batch(&[], params))
    }
}

#[cfg(feature = "jit")]
impl Default for BayesianJITOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::expressions::*;

    #[test]
    fn test_normal_likelihood_expression() {
        let likelihood = normal_likelihood("x", "mu", "sigma");
        // normal_log_pdf returns normalization + quadratic, which is an Add expression
        assert!(matches!(likelihood, symbolic_math::expr::Expr::Add(_, _)));
    }

    #[test]
    fn test_normal_prior_expression() {
        let prior = normal_prior("mu", 0.0, 1.0);
        // normal_log_pdf returns normalization + quadratic, which is an Add expression
        assert!(matches!(prior, symbolic_math::expr::Expr::Add(_, _)));
    }

    #[test]
    fn test_posterior_combination() {
        let likelihood = normal_likelihood("x", "mu", "sigma");
        let prior = normal_prior("mu", 0.0, 1.0);
        let posterior = posterior_log_density(likelihood, prior);
        assert!(matches!(posterior, symbolic_math::expr::Expr::Add(_, _)));
    }
}

// Re-export commonly used functions
#[cfg(feature = "symbolic")]
pub use expressions::{
    hierarchical_normal, mixture_likelihood, normal_likelihood, normal_prior, posterior_log_density,
};

// Re-export Expr for convenience
#[cfg(feature = "symbolic")]
pub use symbolic_math::Expr;

/// Create a variable expression
#[cfg(feature = "symbolic")]
#[must_use]
pub fn variable(name: &str) -> Expr {
    Expr::variable(name)
}

/// Create a likelihood expression (alias for `normal_likelihood`)
#[cfg(feature = "symbolic")]
#[must_use]
pub fn likelihood(x_var: &str, mu_var: &str, sigma_var: &str) -> Expr {
    normal_likelihood(x_var, mu_var, sigma_var)
}

/// Create a prior expression (alias for `normal_prior`)
#[cfg(feature = "symbolic")]
#[must_use]
pub fn prior(param_var: &str, prior_mean: f64, prior_std: f64) -> Expr {
    normal_prior(param_var, prior_mean, prior_std)
}

/// Create a posterior expression (alias for `posterior_log_density`)
#[cfg(feature = "symbolic")]
#[must_use]
pub fn posterior(likelihood: Expr, prior: Expr) -> Expr {
    posterior_log_density(likelihood, prior)
}

/// Bayesian modeling example
///
/// This example demonstrates how to build Bayesian models using the expression system.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "symbolic")]
/// # {
/// use measures::bayesian::{normal_likelihood, normal_prior, posterior_log_density};
/// use symbolic_math::{Expr, builders, display};
/// use std::collections::HashMap;
///
/// // Create likelihood: log p(x | μ, σ)
/// let likelihood = normal_likelihood("x", "mu", "sigma");
///
/// // Create prior: log p(μ)
/// let prior = normal_prior("mu", 0.0, 1.0);
///
/// // Combine into posterior: log p(μ | x, σ) ∝ log p(x | μ, σ) + log p(μ)
/// let posterior = posterior_log_density(likelihood, prior);
/// # }
/// ```
#[cfg(feature = "symbolic")]
pub fn bayesian_example() {
    // This function serves as documentation for the example above
}
