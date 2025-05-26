//! Log-density decomposition framework for all distributions.
//!
//! This module provides a unified framework for decomposing log-densities into
//! structured components, enabling efficient computation for both exponential
//! families and non-exponential families.
//!
//! Every log-density can be decomposed as:
//! log p(x|θ) = f_data(x) + f_param(θ) + f_mixed(x,θ) + f_const
//!
//! Where:
//! - f_data(x): Terms that depend only on data
//! - f_param(θ): Terms that depend only on parameters  
//! - f_mixed(x,θ): Terms that depend on both data and parameters
//! - f_const: Fixed constants
//!
//! This decomposition enables:
//! - Efficient IID sample computation: Σᵢ f_data(xᵢ) + n·f_param(θ) + Σᵢ f_mixed(xᵢ,θ) + n·f_const
//! - Parameter optimization: Only recompute parameter-dependent terms
//! - Caching strategies: Cache data-dependent terms across parameter updates

use num_traits::Float;
use std::marker::PhantomData;

/// A decomposed log-density that separates data, parameter, and mixed terms.
///
/// This provides a structured representation that enables efficient computation
/// for IID samples and parameter optimization.
#[derive(Clone, Debug)]
pub struct LogDensityDecomposition<X, Theta, F> {
    /// Terms that depend only on data: `f_data(x)`
    pub data_terms: Vec<DataTerm<X, F>>,
    /// Terms that depend only on parameters: `f_param(θ)`
    pub param_terms: Vec<ParamTerm<Theta, F>>,
    /// Terms that depend on both data and parameters: `f_mixed(x,θ)`
    pub mixed_terms: Vec<MixedTerm<X, Theta, F>>,
    /// Fixed constant terms
    pub constant_terms: Vec<F>,
    /// Phantom data for type parameters
    _phantom: PhantomData<(X, Theta)>,
}

/// A term that depends only on data.
#[derive(Clone, Debug)]
pub struct DataTerm<X, F> {
    /// Function computing the data-dependent term
    pub compute: fn(&X) -> F,
    /// Optional description for debugging/optimization
    pub description: Option<String>,
}

/// A term that depends only on parameters.
#[derive(Clone, Debug)]
pub struct ParamTerm<Theta, F> {
    /// Function computing the parameter-dependent term
    pub compute: fn(&Theta) -> F,
    /// Optional description for debugging/optimization
    pub description: Option<String>,
}

/// A term that depends on both data and parameters.
#[derive(Clone, Debug)]
pub struct MixedTerm<X, Theta, F> {
    /// Function computing the mixed term
    pub compute: fn(&X, &Theta) -> F,
    /// Optional description for debugging/optimization
    pub description: Option<String>,
}

impl<X, Theta, F: Float> LogDensityDecomposition<X, Theta, F> {
    /// Create a new empty decomposition.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_terms: Vec::new(),
            param_terms: Vec::new(),
            mixed_terms: Vec::new(),
            constant_terms: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a data-only term.
    pub fn add_data_term(mut self, compute: fn(&X) -> F, description: Option<String>) -> Self {
        self.data_terms.push(DataTerm {
            compute,
            description,
        });
        self
    }

    /// Add a parameter-only term.
    pub fn add_param_term(mut self, compute: fn(&Theta) -> F, description: Option<String>) -> Self {
        self.param_terms.push(ParamTerm {
            compute,
            description,
        });
        self
    }

    /// Add a mixed term.
    pub fn add_mixed_term(
        mut self,
        compute: fn(&X, &Theta) -> F,
        description: Option<String>,
    ) -> Self {
        self.mixed_terms.push(MixedTerm {
            compute,
            description,
        });
        self
    }

    /// Add a constant term.
    pub fn add_constant(mut self, value: F) -> Self {
        self.constant_terms.push(value);
        self
    }

    /// Evaluate the complete log-density at given data and parameters.
    pub fn evaluate(&self, x: &X, theta: &Theta) -> F {
        let data_sum: F = self
            .data_terms
            .iter()
            .map(|term| (term.compute)(x))
            .fold(F::zero(), |acc, val| acc + val);

        let param_sum: F = self
            .param_terms
            .iter()
            .map(|term| (term.compute)(theta))
            .fold(F::zero(), |acc, val| acc + val);

        let mixed_sum: F = self
            .mixed_terms
            .iter()
            .map(|term| (term.compute)(x, theta))
            .fold(F::zero(), |acc, val| acc + val);

        let constant_sum: F = self
            .constant_terms
            .iter()
            .fold(F::zero(), |acc, &val| acc + val);

        data_sum + param_sum + mixed_sum + constant_sum
    }

    /// Evaluate for IID samples efficiently.
    ///
    /// For n IID samples x₁, x₂, ..., xₙ, computes:
    /// Σᵢ `f_data(xᵢ)` + `n·f_param(θ)` + Σᵢ `f_mixed(xᵢ,θ)` + `n·f_const`
    pub fn evaluate_iid(&self, samples: &[X], theta: &Theta) -> F
    where
        F: std::iter::Sum,
    {
        let n = F::from(samples.len()).unwrap();

        // Sum data terms across all samples
        let data_sum: F = samples
            .iter()
            .map(|x| {
                self.data_terms
                    .iter()
                    .map(|term| (term.compute)(x))
                    .fold(F::zero(), |acc, val| acc + val)
            })
            .sum();

        // Parameter terms scaled by sample size
        let param_sum: F = self
            .param_terms
            .iter()
            .map(|term| (term.compute)(theta))
            .fold(F::zero(), |acc, val| acc + val);
        let scaled_param_sum = n * param_sum;

        // Sum mixed terms across all samples
        let mixed_sum: F = samples
            .iter()
            .map(|x| {
                self.mixed_terms
                    .iter()
                    .map(|term| (term.compute)(x, theta))
                    .fold(F::zero(), |acc, val| acc + val)
            })
            .sum();

        // Constant terms scaled by sample size
        let constant_sum: F = self
            .constant_terms
            .iter()
            .fold(F::zero(), |acc, &val| acc + val);
        let scaled_constant_sum = n * constant_sum;

        data_sum + scaled_param_sum + mixed_sum + scaled_constant_sum
    }

    /// Evaluate only the parameter-dependent terms.
    ///
    /// Useful for parameter optimization when data terms are cached.
    pub fn evaluate_param_terms(&self, theta: &Theta) -> F {
        self.param_terms
            .iter()
            .map(|term| (term.compute)(theta))
            .fold(F::zero(), |acc, val| acc + val)
    }

    /// Evaluate only the data-dependent terms.
    ///
    /// Useful for caching when parameters are fixed.
    pub fn evaluate_data_terms(&self, x: &X) -> F {
        self.data_terms
            .iter()
            .map(|term| (term.compute)(x))
            .fold(F::zero(), |acc, val| acc + val)
    }

    /// Get the sum of all constant terms.
    #[must_use]
    pub fn constant_sum(&self) -> F {
        self.constant_terms
            .iter()
            .fold(F::zero(), |acc, &val| acc + val)
    }
}

impl<X, Theta, F: Float> Default for LogDensityDecomposition<X, Theta, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for distributions that can provide structured log-density decomposition.
pub trait HasLogDensityDecomposition<X, Theta, F: Float> {
    /// Get the structured decomposition of the log-density.
    fn log_density_decomposition(&self) -> LogDensityDecomposition<X, Theta, F>;
}

/// Builder for creating log-density decompositions with a fluent interface.
pub struct DecompositionBuilder<X, Theta, F> {
    decomposition: LogDensityDecomposition<X, Theta, F>,
}

impl<X, Theta, F: Float> DecompositionBuilder<X, Theta, F> {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decomposition: LogDensityDecomposition::new(),
        }
    }

    /// Add a data-only term with description.
    pub fn data_term(mut self, compute: fn(&X) -> F, description: &str) -> Self {
        self.decomposition = self
            .decomposition
            .add_data_term(compute, Some(description.to_string()));
        self
    }

    /// Add a parameter-only term with description.
    pub fn param_term(mut self, compute: fn(&Theta) -> F, description: &str) -> Self {
        self.decomposition = self
            .decomposition
            .add_param_term(compute, Some(description.to_string()));
        self
    }

    /// Add a mixed term with description.
    pub fn mixed_term(mut self, compute: fn(&X, &Theta) -> F, description: &str) -> Self {
        self.decomposition = self
            .decomposition
            .add_mixed_term(compute, Some(description.to_string()));
        self
    }

    /// Add a constant term.
    pub fn constant(mut self, value: F) -> Self {
        self.decomposition = self.decomposition.add_constant(value);
        self
    }

    /// Build the final decomposition.
    #[must_use]
    pub fn build(self) -> LogDensityDecomposition<X, Theta, F> {
        self.decomposition
    }
}

impl<X, Theta, F: Float> Default for DecompositionBuilder<X, Theta, F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_builder() {
        // Example: Normal distribution N(μ, σ²)
        // log p(x|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²

        let decomp = DecompositionBuilder::<f64, (f64, f64), f64>::new()
            .constant(-0.5 * (2.0 * std::f64::consts::PI).ln()) // -0.5*log(2π)
            .param_term(|(_, sigma): &(f64, f64)| -sigma.ln(), "negative log scale") // -log(σ)
            .mixed_term(
                |x: &f64, (mu, sigma): &(f64, f64)| -0.5 * (x - mu).powi(2) / (sigma * sigma),
                "negative squared error term",
            ) // -0.5*(x-μ)²/σ²
            .build();

        let x = 1.0;
        let theta = (0.0, 1.0); // μ=0, σ=1

        let result = decomp.evaluate(&x, &theta);

        // Should match standard normal log-density at x=1
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.0 - 0.5 * 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_iid_evaluation() {
        // Simple example: exponential distribution
        // log p(x|λ) = log(λ) - λx

        let decomp = DecompositionBuilder::<f64, f64, f64>::new()
            .param_term(|lambda: &f64| lambda.ln(), "log rate") // log(λ)
            .mixed_term(
                |x: &f64, lambda: &f64| -lambda * x,
                "negative rate times data",
            ) // -λx
            .build();

        let samples = vec![1.0, 2.0, 3.0];
        let lambda = 0.5;

        let iid_result = decomp.evaluate_iid(&samples, &lambda);

        // Manual calculation
        let manual_result = samples
            .iter()
            .map(|&x| lambda.ln() - lambda * x)
            .sum::<f64>();

        assert!((iid_result - manual_result).abs() < 1e-10);
    }

    #[test]
    fn test_partial_evaluation() {
        let decomp = DecompositionBuilder::<f64, f64, f64>::new()
            .data_term(|x: &f64| x.powi(2), "x squared")
            .param_term(|theta: &f64| theta.ln(), "log theta")
            .mixed_term(|x: &f64, theta: &f64| x * theta, "x times theta")
            .constant(5.0)
            .build();

        let x = 2.0;
        let theta = 3.0;

        // Test partial evaluations
        let data_only = decomp.evaluate_data_terms(&x);
        let param_only = decomp.evaluate_param_terms(&theta);
        let constants = decomp.constant_sum();
        let mixed = (decomp.mixed_terms[0].compute)(&x, &theta);

        let total = decomp.evaluate(&x, &theta);
        let manual_total = data_only + param_only + mixed + constants;

        assert!((total - manual_total).abs() < 1e-10);
    }
}
