//! Exponential Family Relative Density Optimization Example
//!
//! This example demonstrates the specialized optimization for computing relative densities
//! between distributions from the same exponential family. When both distributions are
//! from the same exponential family, the base measure terms cancel out, allowing for
//! a much more efficient computation.
//!
//! Mathematical insight:
//! For two exponential families p₁(x|θ₁) and p₂(x|θ₂) from the same family:
//! log(p₁(x)/p₂(x)) = [η₁·T(x) - A(η₁) + log h(x)] - [η₂·T(x) - A(η₂) + log h(x)]
//!                   = (η₁ - η₂)·T(x) - (A(η₁) - A(η₂))
//!
//! The base measure terms log h(x) cancel out completely!
//!
//! Run with: cargo run --example `exponential_family_relative_density`

use measures::exponential_family::ExponentialFamily;
use measures::traits::DotProduct;
use measures::{LogDensityBuilder, Normal, distributions::Exponential};

fn main() {
    println!("🎯 === Exponential Family Relative Density Optimization === 🎯\n");

    demonstrate_normal_optimization();
    demonstrate_exponential_optimization();
    demonstrate_mathematical_insight();
    demonstrate_performance_benefit();

    println!("\n🎉 === Exponential Family Optimization Complete! === 🎉");
    println!("✅ Demonstrated efficient relative density computation for exponential families");
    println!("✅ Showed mathematical correctness of the optimization");
    println!("🚀 Base measure terms cancel out, making computation much more efficient!");
}

fn demonstrate_normal_optimization() {
    println!("=== 1. Normal Distribution Relative Density ===");

    let normal1 = Normal::new(0.0, 1.0); // Standard normal
    let normal2 = Normal::new(2.0, 1.5); // Different parameters
    let x = 1.0;

    // Standard computation (computes both densities separately)
    let standard_result = normal1.log_density().at(&x) - normal2.log_density().at(&x);

    // Optimized computation using builder pattern (automatically uses optimization)
    let optimized_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);

    // Direct optimized computation
    let direct_optimized =
        measures::core::density::compute_exp_fam_relative_density(&normal1, &normal2, &x);

    println!("Normal₁: N(μ=0.0, σ=1.0)");
    println!("Normal₂: N(μ=2.0, σ=1.5)");
    println!("Point: x = {x}");
    println!();
    println!("Standard computation:  {standard_result:.10}");
    println!("Builder optimization:  {optimized_result:.10}");
    println!("Direct optimization:   {direct_optimized:.10}");
    println!(
        "Difference (should be ~0): {:.2e}",
        (optimized_result - standard_result).abs()
    );
    println!();
}

fn demonstrate_exponential_optimization() {
    println!("=== 2. Exponential Distribution Relative Density ===");

    let exp1 = Exponential::new(1.0); // Rate λ = 1.0
    let exp2 = Exponential::new(2.5); // Rate λ = 2.5
    let x = 0.5;

    // Standard computation
    let standard_result = exp1.log_density().at(&x) - exp2.log_density().at(&x);

    // Optimized computation
    let optimized_result: f64 = exp1.log_density().wrt(exp2.clone()).at(&x);

    println!("Exponential₁: Exp(λ=1.0)");
    println!("Exponential₂: Exp(λ=2.5)");
    println!("Point: x = {x}");
    println!();
    println!("Standard computation: {standard_result:.10}");
    println!("Optimized computation: {optimized_result:.10}");
    println!(
        "Difference (should be ~0): {:.2e}",
        (optimized_result - standard_result).abs()
    );
    println!();
}

fn demonstrate_mathematical_insight() {
    println!("=== 3. Mathematical Insight: Base Measure Cancellation ===");

    let normal1 = Normal::new(1.0, 0.8);
    let normal2 = Normal::new(-0.5, 1.2);
    let x = 0.3;

    // Get the exponential family components
    let (eta1, log_partition1) = normal1.natural_and_log_partition();
    let (eta2, log_partition2) = normal2.natural_and_log_partition();
    let sufficient_stat = normal1.sufficient_statistic(&x);

    println!("Normal₁: N(μ=1.0, σ=0.8)");
    println!("Normal₂: N(μ=-0.5, σ=1.2)");
    println!("Point: x = {x}");
    println!();

    // Show the exponential family structure
    println!("Exponential family components:");
    println!("η₁ = [{:.6}, {:.6}]", eta1[0], eta1[1]);
    println!("η₂ = [{:.6}, {:.6}]", eta2[0], eta2[1]);
    println!(
        "T(x) = [{:.6}, {:.6}]",
        sufficient_stat[0], sufficient_stat[1]
    );
    println!("A(η₁) = {log_partition1:.6}");
    println!("A(η₂) = {log_partition2:.6}");
    println!();

    // Manual computation using the optimized formula
    let eta1_dot = eta1.dot(&sufficient_stat);
    let eta2_dot = eta2.dot(&sufficient_stat);
    let manual_optimized = (eta1_dot - eta2_dot) - (log_partition1 - log_partition2);

    // Compare with automatic computation
    let automatic_result: f64 = normal1.log_density().wrt(normal2).at(&x);

    println!("Manual optimized formula: (η₁·T(x) - η₂·T(x)) - (A(η₁) - A(η₂))");
    println!(
        "                        = ({eta1_dot:.6} - {eta2_dot:.6}) - ({log_partition1:.6} - {log_partition2:.6})"
    );
    println!("                        = {manual_optimized:.10}");
    println!("Automatic computation:    {automatic_result:.10}");
    println!(
        "Difference (should be ~0): {:.2e}",
        (manual_optimized - automatic_result).abs()
    );
    println!();
    println!("🔑 Key insight: The base measure terms log h(x) completely cancel out!");
    println!("   This makes the computation much more efficient and numerically stable.");
    println!();
}

fn demonstrate_performance_benefit() {
    println!("=== 4. Performance Benefit Analysis ===");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 2.0);
    let test_points: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01).collect();

    println!(
        "Computing relative density for {} points...",
        test_points.len()
    );

    // Time the standard approach
    let start = std::time::Instant::now();
    let _standard_results: Vec<f64> = test_points
        .iter()
        .map(|&x| normal1.log_density().at(&x) - normal2.log_density().at(&x))
        .collect();
    let standard_time = start.elapsed();

    // Time the optimized approach
    let start = std::time::Instant::now();
    let _optimized_results: Vec<f64> = test_points
        .iter()
        .map(|&x| normal1.log_density().wrt(normal2.clone()).at(&x))
        .collect();
    let optimized_time = start.elapsed();

    println!("Standard approach:  {standard_time:?}");
    println!("Optimized approach: {optimized_time:?}");

    if standard_time > optimized_time {
        let speedup = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        println!("Speedup: {speedup:.2}x faster");
    } else {
        println!("Note: For small computations, the difference may not be measurable");
        println!("The benefit becomes more apparent with:");
        println!("  • More complex exponential families");
        println!("  • Higher-dimensional sufficient statistics");
        println!("  • Repeated computations");
    }
    println!();
    println!("💡 The optimization provides:");
    println!("   • Fewer floating-point operations");
    println!("   • Better numerical stability");
    println!("   • Cleaner mathematical expression");
    println!("   • Automatic base measure handling");
}
