//! General Density Computation Example
//!
//! Demonstrates computing log-densities with respect to any base measure,
//! useful for relative densities, importance sampling, and model comparison.

use measures::{LogDensityBuilder, Normal};

#[cfg(feature = "jit")]
use measures::exponential_family::jit::ZeroOverheadOptimizer;

fn main() {
    let normal1 = Normal::new(0.0, 1.0); // Standard normal
    let normal2 = Normal::new(1.0, 2.0); // Different mean and variance
    let test_point = 0.5;

    // Standard density computation (wrt root measure)
    let density1 = normal1.log_density().at(&test_point);
    let density2 = normal2.log_density().at(&test_point);
    println!("Standard densities: p1={:.6}, p2={:.6}, diff={:.6}", 
             density1, density2, density1 - density2);

    // Relative density computation (normal1 wrt normal2)
    let relative_density = normal1.log_density().wrt(normal2.clone()).at(&test_point);
    let manual_computation = density1 - density2;
    println!("Relative density: {:.6}, manual: {:.6}, diff: {:.10}", 
             relative_density, manual_computation, 
             (relative_density - manual_computation).abs());

    #[cfg(feature = "jit")]
    {
        // Optimized relative density computation
        let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
        let optimized_result: f64 = optimized_fn(&test_point);
        println!("Optimized result: {:.6}", optimized_result);
    }

    // Practical applications
    let importance_weight = relative_density.exp();
    let log_bayes_factor = density1 - density2;
    
    println!("Applications:");
    println!("  Importance weight: {:.6}", importance_weight);
    println!("  Log Bayes factor: {:.6}", log_bayes_factor);
    println!("  Bayes factor: {:.6}", log_bayes_factor.exp());
}
