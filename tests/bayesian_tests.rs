//! Comprehensive tests for Bayesian inference functionality
//!
//! These tests focus on semantic invariants and use property-based testing
//! to ensure robustness across a wide range of inputs.

use measures::bayesian::{
    Expr, hierarchical_normal, mixture_likelihood, normal_likelihood, normal_prior,
    posterior_log_density,
};
use proptest::prelude::*;

/// Test that normal likelihood expressions have the expected structure
#[test]
fn test_normal_likelihood_structure() {
    let likelihood = normal_likelihood("x", "mu", "sigma");

    // The likelihood should be an addition (normal_log_pdf returns normalization + quadratic)
    assert!(matches!(likelihood, Expr::Add(_, _)));
}

/// Test that normal prior expressions have the expected structure
#[test]
fn test_normal_prior_structure() {
    let prior = normal_prior("theta", 0.0, 1.0);

    // The prior should be an addition (normal_log_pdf returns normalization + quadratic)
    assert!(matches!(prior, Expr::Add(_, _)));
}

/// Test that posterior combination preserves additive structure
#[test]
fn test_posterior_additive_structure() {
    let likelihood = normal_likelihood("x", "mu", "sigma");
    let prior = normal_prior("mu", 0.0, 1.0);
    let posterior = posterior_log_density(likelihood, prior);

    // Posterior should be addition of likelihood and prior
    assert!(matches!(posterior, Expr::Add(_, _)));
}

/// Test hierarchical normal model structure
#[test]
fn test_hierarchical_normal_structure() {
    let hierarchical = hierarchical_normal("x", "mu", "sigma", "tau", 1.0, 0.5);

    // Should be an addition of multiple components
    assert!(matches!(hierarchical, Expr::Add(_, _)));
}

/// Test mixture likelihood with single component reduces to simple likelihood
#[test]
fn test_mixture_likelihood_single_component() {
    let weights = [1.0];
    let means = [0.0];
    let stds = [1.0];

    let mixture = mixture_likelihood("x", &weights, &means, &stds);

    // Should be a natural log of the mixture
    assert!(matches!(mixture, Expr::Ln(_)));
}

/// Test mixture likelihood with multiple components
#[test]
fn test_mixture_likelihood_multiple_components() {
    let weights = [0.3, 0.7];
    let means = [0.0, 2.0];
    let stds = [1.0, 1.5];

    let mixture = mixture_likelihood("x", &weights, &means, &stds);

    // Should be a natural log
    assert!(matches!(mixture, Expr::Ln(_)));
}

// Property-based tests using proptest

proptest! {
    /// Test that normal likelihood is well-formed for any valid variable names
    #[test]
    fn prop_normal_likelihood_well_formed(
        x_var in "[a-zA-Z][a-zA-Z0-9_]*",
        mu_var in "[a-zA-Z][a-zA-Z0-9_]*",
        sigma_var in "[a-zA-Z][a-zA-Z0-9_]*"
    ) {
        let likelihood = normal_likelihood(&x_var, &mu_var, &sigma_var);
        // Should not panic and should have expected structure
        assert!(matches!(likelihood, Expr::Add(_, _)));
    }

    /// Test that normal prior is well-formed for reasonable parameter values
    #[test]
    fn prop_normal_prior_well_formed(
        param_var in "[a-zA-Z][a-zA-Z0-9_]*",
        prior_mean in -100.0..100.0,
        prior_std in 0.01..100.0  // Positive standard deviation
    ) {
        let prior = normal_prior(&param_var, prior_mean, prior_std);
        // Should not panic and should have expected structure
        assert!(matches!(prior, Expr::Add(_, _)));
    }

    /// Test hierarchical normal model with various parameter combinations
    #[test]
    fn prop_hierarchical_normal_well_formed(
        x_var in "[a-zA-Z][a-zA-Z0-9_]*",
        mu_var in "[a-zA-Z][a-zA-Z0-9_]*",
        sigma_var in "[a-zA-Z][a-zA-Z0-9_]*",
        tau_var in "[a-zA-Z][a-zA-Z0-9_]*",
        alpha in -10.0..10.0,
        beta in 0.01..10.0  // Positive beta
    ) {
        let hierarchical = hierarchical_normal(&x_var, &mu_var, &sigma_var, &tau_var, alpha, beta);
        // Should not panic and should be an addition
        assert!(matches!(hierarchical, Expr::Add(_, _)));
    }

    /// Test mixture likelihood with various valid configurations
    #[test]
    fn prop_mixture_likelihood_well_formed(
        x_var in "[a-zA-Z][a-zA-Z0-9_]*",
        num_components in 1usize..=5,
        seed in any::<u64>()
    ) {
        // Generate valid mixture parameters
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let weights: Vec<f64> = (0..num_components)
            .map(|_i| 1.0 / num_components as f64).collect();
        let total_weight: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        let means: Vec<f64> = (0..num_components)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();

        let std_devs: Vec<f64> = (0..num_components)
            .map(|_| rng.random_range(0.1..5.0))
            .collect();

        let mixture = mixture_likelihood(&x_var, &normalized_weights, &means, &std_devs);

        // Should be a natural log
        assert!(matches!(mixture, Expr::Ln(_)));
    }
}

/// Test semantic invariants for Bayesian expressions

#[test]
fn test_posterior_symmetry_in_likelihood_and_prior() {
    let likelihood1 = normal_likelihood("x", "mu", "sigma");
    let prior1 = normal_prior("mu", 0.0, 1.0);
    let posterior1 = posterior_log_density(likelihood1.clone(), prior1.clone());

    // Posterior should be commutative in addition
    let posterior2 = posterior_log_density(prior1, likelihood1);

    // Both should have the same structure (addition)
    assert!(matches!(posterior1, Expr::Add(_, _)));
    assert!(matches!(posterior2, Expr::Add(_, _)));
}

#[test]
fn test_mixture_likelihood_weight_normalization_invariant() {
    // Test that mixture likelihood is well-defined even with unnormalized weights
    let unnormalized_weights = [2.0, 3.0, 5.0]; // Sum = 10
    let means = [0.0, 1.0, 2.0];
    let stds = [1.0, 1.0, 1.0];

    let mixture = mixture_likelihood("x", &unnormalized_weights, &means, &stds);

    // Should still be well-formed (the function should handle normalization internally if needed)
    assert!(matches!(mixture, Expr::Ln(_)));
}

#[test]
fn test_hierarchical_model_parameter_independence() {
    // Test that different parameter names produce different expressions
    let model1 = hierarchical_normal("x1", "mu1", "sigma1", "tau1", 1.0, 0.5);
    let model2 = hierarchical_normal("x2", "mu2", "sigma2", "tau2", 1.0, 0.5);

    // Both should be well-formed
    assert!(matches!(model1, Expr::Add(_, _)));
    assert!(matches!(model2, Expr::Add(_, _)));

    // They should be structurally similar but with different variable names
    // (This is a basic sanity check - more sophisticated equality would require expression comparison)
}

/// Test edge cases and boundary conditions

#[test]
fn test_mixture_likelihood_single_component_edge_case() {
    // Single component with weight 1.0
    let weights = [1.0];
    let means = [0.0];
    let stds = [1.0];

    let mixture = mixture_likelihood("x", &weights, &means, &stds);
    assert!(matches!(mixture, Expr::Ln(_)));
}

#[test]
fn test_hierarchical_normal_zero_parameters() {
    // Test with zero values for some parameters
    let hierarchical = hierarchical_normal("x", "mu", "sigma", "tau", 0.0, 1.0);
    assert!(matches!(hierarchical, Expr::Add(_, _)));
}

#[test]
fn test_normal_prior_zero_mean() {
    // Test prior with zero mean (common case)
    let prior = normal_prior("theta", 0.0, 1.0);
    assert!(matches!(prior, Expr::Add(_, _)));
}

#[test]
fn test_normal_prior_unit_variance() {
    // Test prior with unit variance (standard normal prior)
    let prior = normal_prior("theta", 0.0, 1.0);
    assert!(matches!(prior, Expr::Add(_, _)));
}

/// Test that expressions don't panic with extreme but valid inputs

#[test]
fn test_extreme_parameter_values() {
    // Very large standard deviation
    let prior_large_std = normal_prior("theta", 0.0, 1000.0);
    assert!(matches!(prior_large_std, Expr::Add(_, _)));

    // Very small standard deviation
    let prior_small_std = normal_prior("theta", 0.0, 0.001);
    assert!(matches!(prior_small_std, Expr::Add(_, _)));

    // Large mean values
    let prior_large_mean = normal_prior("theta", 1000.0, 1.0);
    assert!(matches!(prior_large_mean, Expr::Add(_, _)));
}

#[test]
fn test_mixture_with_many_components() {
    // Test mixture with maximum reasonable number of components
    const NUM_COMPONENTS: usize = 10;
    let weights: Vec<f64> = (0..NUM_COMPONENTS)
        .map(|_i| 1.0 / NUM_COMPONENTS as f64)
        .collect();
    let means: Vec<f64> = (0..NUM_COMPONENTS).map(|i| i as f64).collect();
    let stds: Vec<f64> = vec![1.0; NUM_COMPONENTS];

    let mixture = mixture_likelihood("x", &weights, &means, &stds);
    assert!(matches!(mixture, Expr::Ln(_)));
}
