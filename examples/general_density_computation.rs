//! General Density Computation Example
//!
//! This example demonstrates how to compute log-densities with respect to any base measure,
//! not just the root measure. This is useful for:
//! - Computing relative densities between different distributions
//! - Importance sampling with different proposal distributions
//! - Bayesian model comparison
//! - Change of measure computations
//!
//! Run with: cargo run --example `general_density_computation`

use measures::{LogDensityBuilder, Normal};

#[cfg(feature = "jit")]
use measures::exponential_family::jit::ZeroOverheadOptimizer;

fn main() {
    println!("🎯 === General Density Computation === 🎯\n");

    // Create two different normal distributions
    let normal1 = Normal::new(0.0, 1.0); // Standard normal
    let normal2 = Normal::new(1.0, 2.0); // Different mean and variance
    let test_point = 0.5;

    println!("Distribution 1: Normal(μ=0.0, σ=1.0)");
    println!("Distribution 2: Normal(μ=1.0, σ=2.0)");
    println!("Test point: x = {test_point}\n");

    demonstrate_standard_density_computation(&normal1, &normal2, test_point);
    demonstrate_relative_density_computation(&normal1, &normal2, test_point);
    demonstrate_general_log_density_trait(&normal1, &normal2, test_point);

    #[cfg(feature = "jit")]
    demonstrate_optimized_relative_density(&normal1, &normal2, test_point);

    demonstrate_practical_applications(&normal1, &normal2, test_point);

    println!("\n🎉 === General Density Computation Complete! === 🎉");
    println!("✅ Demonstrated density computation with respect to any base measure");
    println!("✅ Showed practical applications for statistical computing");
    println!("🚀 This enables more flexible and powerful statistical computations!");
}

fn demonstrate_standard_density_computation(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) {
    println!("=== 1. Standard Density Computation (wrt root measure) ===");

    let density1 = normal1.log_density().at(&x);
    let density2 = normal2.log_density().at(&x);

    println!("log p₁(x) = {density1:.6} (Normal1 wrt Lebesgue measure)");
    println!("log p₂(x) = {density2:.6} (Normal2 wrt Lebesgue measure)");
    println!(
        "Difference: log p₁(x) - log p₂(x) = {:.6}",
        density1 - density2
    );
    println!("This tells us the relative likelihood under the two distributions\n");
}

fn demonstrate_relative_density_computation(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) {
    println!("=== 2. Relative Density Computation (Normal1 wrt Normal2) ===");

    // Compute log-density of normal1 with respect to normal2 as base measure
    let relative_density = normal1.log_density().wrt(normal2.clone()).at(&x);

    println!("log(p₁/p₂)(x) = {relative_density:.6}");
    println!("This directly gives us the log-ratio of the two densities");

    // Verify this matches the manual computation
    let manual_computation = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    println!("Manual verification: {manual_computation:.6}");
    println!(
        "Difference: {:.10} (should be ~0)",
        (relative_density - manual_computation).abs()
    );
    println!();
}

fn demonstrate_general_log_density_trait(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) {
    println!("=== 3. Using Builder Pattern for Relative Densities ===");

    // Use the builder pattern for computing relative densities
    let relative_density = normal1.log_density().wrt(normal2.clone()).at(&x);

    println!("Using builder pattern: {relative_density:.6}");
    println!("This provides a fluent API for computing densities wrt any measure");
    println!("The .wrt() method changes the base measure for the computation");
    println!();
}

#[cfg(feature = "jit")]
fn demonstrate_optimized_relative_density(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) {
    println!("=== 4. Optimized Relative Density Computation ===");

    // Create optimized function for computing density wrt normal2
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let optimized_result: f64 = optimized_fn(&x);

    println!("Optimized computation: {optimized_result:.6}");
    println!("This pre-computes constants for maximum performance");

    // Note: Macro version available but requires explicit type annotations
    println!("Macro version: measures::optimized_exp_fam!(dist, wrt: base) also available");
    println!();
}

#[cfg(not(feature = "jit"))]
fn demonstrate_optimized_relative_density(_normal1: &Normal<f64>, _normal2: &Normal<f64>, _x: f64) {
    println!("=== 4. Optimized Relative Density Computation ===");
    println!("(Requires --features jit to demonstrate optimization)");
    println!();
}

fn demonstrate_practical_applications(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) {
    println!("=== 5. Practical Applications ===");

    println!("🎯 Importance Sampling:");
    let importance_weight = normal1.log_density().wrt(normal2.clone()).at(&x).exp();
    println!("   Weight for importance sampling: {importance_weight:.6}");
    println!("   (when using Normal2 as proposal, Normal1 as target)");

    println!("\n📊 Model Comparison:");
    let log_bayes_factor = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    println!("   Log Bayes factor: {log_bayes_factor:.6}");
    println!("   Bayes factor: {:.6}", log_bayes_factor.exp());

    println!("\n🔄 Change of Measure:");
    println!("   Radon-Nikodym derivative: dP₁/dP₂ = {importance_weight:.6}");
    println!("   This quantifies how the measures differ at point x");

    println!("\n💡 Use Cases:");
    println!("   • MCMC with different proposal distributions");
    println!("   • Variational inference with flexible base measures");
    println!("   • Robust statistics with different reference measures");
    println!("   • Financial risk modeling with measure changes");
}
