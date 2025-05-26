//! Automatic Differentiation Example using ad-trait
//!
//! This example demonstrates how to use automatic differentiation
//! with measure theory concepts, particularly for computing gradients
//! of density functions and log-likelihood functions.

use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::reverse_ad::adr::adr;
use std::f64::consts::PI;

/// A Gaussian density function that can work with automatic differentiation
#[derive(Clone)]
pub struct GaussianDensity<T: AD> {
    /// Mean parameter
    mu: T,
    /// Standard deviation parameter (must be positive)
    sigma: T,
}

impl<T: AD> DifferentiableFunctionTrait<T> for GaussianDensity<T> {
    const NAME: &'static str = "GaussianDensity";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];

        // Compute Gaussian density: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
        let two_pi = T::constant(2.0 * PI);
        let half = T::constant(0.5);
        let one = T::constant(1.0);

        let normalization = one / (self.sigma * two_pi.sqrt());
        let z_score = (x - self.mu) / self.sigma;
        let exponent = -half * z_score * z_score;

        vec![normalization * exponent.exp()]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

impl<T: AD> GaussianDensity<T> {
    pub fn new(mu: T, sigma: T) -> Self {
        Self { mu, sigma }
    }

    pub fn to_other_ad_type<T2: AD>(&self) -> GaussianDensity<T2> {
        GaussianDensity {
            mu: self.mu.to_other_ad_type::<T2>(),
            sigma: self.sigma.to_other_ad_type::<T2>(),
        }
    }
}

/// A log-likelihood function for multiple Gaussian observations
#[derive(Clone)]
pub struct GaussianLogLikelihood<T: AD> {
    /// Observed data points
    observations: Vec<T>,
}

impl<T: AD> DifferentiableFunctionTrait<T> for GaussianLogLikelihood<T> {
    const NAME: &'static str = "GaussianLogLikelihood";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let mu = inputs[0];
        let sigma = inputs[1];

        let two_pi = T::constant(2.0 * PI);
        let half = T::constant(0.5);
        let two = T::constant(2.0);
        let n = T::constant(self.observations.len() as f64);

        // Log-likelihood: -n/2 * log(2*pi) - n * log(sigma) - sum((x_i - mu)^2) / (2 * sigma^2)
        let log_normalization = -half * n * two_pi.ln() - n * sigma.ln();

        let mut sum_squared_errors = T::constant(0.0);
        for obs in &self.observations {
            let error = *obs - mu;
            sum_squared_errors += error * error;
        }

        let log_exp_term = -sum_squared_errors / (two * sigma * sigma);

        vec![log_normalization + log_exp_term]
    }

    fn num_inputs(&self) -> usize {
        2 // mu and sigma
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

impl<T: AD> GaussianLogLikelihood<T> {
    pub fn new(observations: Vec<f64>) -> Self {
        Self {
            observations: observations.into_iter().map(T::constant).collect(),
        }
    }

    #[must_use]
    pub fn to_other_ad_type<T2: AD>(&self) -> GaussianLogLikelihood<T2> {
        GaussianLogLikelihood {
            observations: self
                .observations
                .iter()
                .map(ad_trait::AD::to_other_ad_type::<T2>)
                .collect(),
        }
    }
}

/// A simple polynomial function for demonstration
#[derive(Clone)]
pub struct Polynomial<T: AD> {
    coefficients: Vec<T>,
}

impl<T: AD> DifferentiableFunctionTrait<T> for Polynomial<T> {
    const NAME: &'static str = "Polynomial";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        let mut result = T::constant(0.0);
        let mut x_power = T::constant(1.0);

        for coeff in &self.coefficients {
            result += *coeff * x_power;
            x_power *= x;
        }

        vec![result]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

impl<T: AD> Polynomial<T> {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self {
            coefficients: coefficients.into_iter().map(T::constant).collect(),
        }
    }

    #[must_use]
    pub fn to_other_ad_type<T2: AD>(&self) -> Polynomial<T2> {
        Polynomial {
            coefficients: self
                .coefficients
                .iter()
                .map(ad_trait::AD::to_other_ad_type::<T2>)
                .collect(),
        }
    }
}

fn main() {
    println!("=== Automatic Differentiation Examples with Measure Theory ===\n");

    // Example 1: Gaussian Density Derivative
    println!("1. Gaussian Density Function Derivative");
    println!("   f(x) = (1/(σ√(2π))) * exp(-0.5 * ((x-μ)/σ)²)");
    println!("   Parameters: μ = 0.0, σ = 1.0");

    let gaussian_standard = GaussianDensity::new(0.0, 1.0);
    let gaussian_derivative = gaussian_standard.to_other_ad_type::<adr>();
    let gaussian_engine =
        FunctionEngine::new(gaussian_standard, gaussian_derivative, ReverseAD::new());

    let x_values = vec![0.0, 1.0, -1.0, 2.0];
    for x in x_values {
        let (density, derivative) = gaussian_engine.derivative(&[x]);
        println!(
            "   At x = {:.1}: density = {:.6}, derivative = {:.6}",
            x,
            density[0],
            derivative[(0, 0)]
        );
    }
    println!();

    // Example 2: Log-Likelihood Gradient
    println!("2. Gaussian Log-Likelihood Gradient");
    println!("   Optimizing μ and σ for observed data");

    // Generate some sample data from N(2.5, 1.5)
    let observations = vec![2.1, 3.2, 1.8, 2.9, 2.3, 3.1, 1.9, 2.7, 2.4, 3.0];
    println!("   Observations: {observations:?}");

    let loglik_standard = GaussianLogLikelihood::new(observations);
    let loglik_derivative = loglik_standard.to_other_ad_type::<adr>();
    let loglik_engine = FunctionEngine::new(loglik_standard, loglik_derivative, ReverseAD::new());

    // Try different parameter values
    let param_sets = vec![
        (2.0, 1.0), // Initial guess
        (2.5, 1.5), // True parameters
        (3.0, 2.0), // Different guess
    ];

    for (mu, sigma) in param_sets {
        let (loglik, gradient) = loglik_engine.derivative(&[mu, sigma]);
        println!(
            "   μ = {:.1}, σ = {:.1}: log-likelihood = {:.4}, ∇μ = {:.4}, ∇σ = {:.4}",
            mu,
            sigma,
            loglik[0],
            gradient[(0, 0)],
            gradient[(0, 1)]
        );
    }
    println!();

    // Example 3: Forward vs Reverse AD comparison
    println!("3. Forward vs Reverse AD Comparison");
    println!("   Polynomial: f(x) = 2x³ - 3x² + x - 1");

    let poly_coeffs = vec![-1.0, 1.0, -3.0, 2.0]; // coefficients for x^0, x^1, x^2, x^3

    // Forward AD
    let poly_standard_fwd = Polynomial::new(poly_coeffs.clone());
    let poly_derivative_fwd = poly_standard_fwd.to_other_ad_type::<adfn<1>>();
    let poly_engine_fwd =
        FunctionEngine::new(poly_standard_fwd, poly_derivative_fwd, ForwardAD::new());

    // Reverse AD
    let poly_standard_rev = Polynomial::new(poly_coeffs);
    let poly_derivative_rev = poly_standard_rev.to_other_ad_type::<adr>();
    let poly_engine_rev =
        FunctionEngine::new(poly_standard_rev, poly_derivative_rev, ReverseAD::new());

    let test_points = vec![0.0, 1.0, -1.0, 2.0];
    for x in test_points {
        let (f_fwd, df_fwd) = poly_engine_fwd.derivative(&[x]);
        let (f_rev, df_rev) = poly_engine_rev.derivative(&[x]);

        println!("   x = {:.1}: f(x) = {:.4}", x, f_fwd[0]);
        println!("     Forward AD:  f'(x) = {:.4}", df_fwd[(0, 0)]);
        println!("     Reverse AD:  f'(x) = {:.4}", df_rev[(0, 0)]);
        println!(
            "     Difference: {:.2e}",
            (df_fwd[(0, 0)] - df_rev[(0, 0)]).abs()
        );
        println!();
    }

    // Example 4: Measure transformation example
    println!("4. Measure Transformation with Jacobian");
    println!("   Transform y = x² and compute Jacobian");

    #[derive(Clone)]
    struct SquareTransform<T: AD> {
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T: AD> DifferentiableFunctionTrait<T> for SquareTransform<T> {
        const NAME: &'static str = "SquareTransform";

        fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
            vec![inputs[0] * inputs[0]]
        }

        fn num_inputs(&self) -> usize {
            1
        }
        fn num_outputs(&self) -> usize {
            1
        }
    }

    impl<T: AD> SquareTransform<T> {
        fn new() -> Self {
            Self {
                _phantom: std::marker::PhantomData,
            }
        }

        fn to_other_ad_type<T2: AD>(&self) -> SquareTransform<T2> {
            SquareTransform::new()
        }
    }

    let transform_standard = SquareTransform::new();
    let transform_derivative = transform_standard.to_other_ad_type::<adr>();
    let transform_engine =
        FunctionEngine::new(transform_standard, transform_derivative, ReverseAD::new());

    let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    for x in x_vals {
        let (y, jacobian) = transform_engine.derivative(&[x]);
        println!(
            "   x = {:.1}: y = x² = {:.1}, dy/dx = {:.1}",
            x,
            y[0],
            jacobian[(0, 0)]
        );
    }

    println!("\n=== Summary ===");
    println!("This example demonstrates:");
    println!("• Computing derivatives of probability density functions");
    println!("• Gradient-based optimization for maximum likelihood estimation");
    println!("• Comparison between forward and reverse mode AD");
    println!("• Jacobian computation for measure transformations");
    println!("• Integration with the measures framework for statistical computing");
}
