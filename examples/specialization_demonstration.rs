//! Specialization Demonstration
//!
//! This example shows how zero-overhead optimization works for exponential family
//! relative density computation, which would be automatic with Rust specialization.
//!
//! Run with: cargo run --example `specialization_demonstration` --features jit

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::{LogDensityBuilder, Normal};

fn main() {
    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let x = 0.5;

    // Standard computation
    let standard_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);

    // Zero-overhead optimized computation
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let optimized_result: f64 = optimized_fn(&x);

    println!("Standard computation:  {standard_result:.10}");
    println!("Zero-overhead result:  {optimized_result:.10}");
    println!(
        "Difference:            {:.2e}",
        (standard_result - optimized_result).abs()
    );

    // Verify they produce identical results
    assert!(
        (standard_result - optimized_result).abs() < 1e-12,
        "Zero-overhead optimization should produce identical results"
    );

    println!("✓ Zero-overhead optimization verified");
}
