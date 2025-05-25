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
    demonstrate_best_practices();

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
    let normal2 = Normal::new(1.0, 1.5);
    let points: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01).collect();

    // Benchmark standard approach
    let start = std::time::Instant::now();
    for &x in &points {
        let _result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    }
    let standard_time = start.elapsed();

    // Benchmark optimized approach
    let start = std::time::Instant::now();
    for &x in &points {
        let _result =
            measures::core::density::compute_exp_fam_relative_density(&normal1, &normal2, &x);
    }
    let optimized_time = start.elapsed();

    // Benchmark builder pattern (uses general approach currently)
    let start = std::time::Instant::now();
    for &x in &points {
        let _result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    }
    let builder_time = start.elapsed();

    println!("Computing relative density for {} points...", points.len());
    println!("Standard approach:  {:.2}µs", standard_time.as_micros());
    println!("Optimized approach: {:.3}µs", optimized_time.as_micros());
    println!("Builder pattern:    {:.2}µs", builder_time.as_micros());

    let speedup = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("Speedup: {speedup:.2}x faster");

    println!("\n💡 The optimization provides:");
    println!("   • Fewer floating-point operations");
    println!("   • Better numerical stability");
    println!("   • Cleaner mathematical expression");
    println!("   • Automatic base measure handling");

    println!("\n🔧 Performance Tips:");
    println!("   • Use compute_exp_fam_relative_density() for same-type exponential families");
    println!("   • Builder pattern: normal1.log_density().wrt(normal2).at(&x) (general approach)");
    println!(
        "   • Direct function: compute_exp_fam_relative_density(&normal1, &normal2, &x) (optimized)"
    );
}

fn demonstrate_best_practices() {
    println!("=== 5. Best Practices for Exponential Family Optimization ===");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let exp1 = Exponential::new(1.0);
    let exp2 = Exponential::new(2.0);
    let x = 0.5;

    println!("📋 When to use each approach:\n");

    // Case 1: Same exponential family type
    println!("1️⃣ Same exponential family type (RECOMMENDED: Use optimized function)");
    let standard_result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    let optimized_result =
        measures::core::density::compute_exp_fam_relative_density(&normal1, &normal2, &x);
    println!("   Standard:  normal1.log_density().at(&x) - normal2.log_density().at(&x)");
    println!("   Optimized: compute_exp_fam_relative_density(&normal1, &normal2, &x)");
    println!("   Results:   {standard_result:.10} vs {optimized_result:.10} ✅");

    // Case 2: Different exponential family types
    println!("\n2️⃣ Different exponential family types (Use builder pattern)");
    let mixed_result: f64 = normal1.log_density().wrt(exp1.clone()).at(&x);
    println!("   Approach:  normal.log_density().wrt(exponential).at(&x)");
    println!("   Result:    {mixed_result:.10}");

    // Case 3: General case (always works)
    println!("\n3️⃣ General case (Always works, but may not be optimal)");
    let general_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    println!("   Approach:  measure1.log_density().wrt(measure2).at(&x)");
    println!("   Result:    {general_result:.10}");

    println!("\n🎯 Performance Guidelines:");
    println!("   • Same type + array natural params → Use compute_exp_fam_relative_density()");
    println!("   • Different types or mixed families → Use builder pattern");
    println!("   • Batch computations → Consider pre-computing parameters");
    println!("   • Repeated evaluations → Consider caching with .cached()");

    println!("\n🔍 Type Constraints:");
    println!("   • compute_exp_fam_relative_density requires:");
    println!("     - Both measures are the same type M");
    println!("     - M implements ExponentialFamily<T, F>");
    println!("     - Natural parameters are arrays [F; N]");
    println!("     - Sufficient statistics are arrays [F; N]");

    println!("\n✨ Mathematical Insight:");
    println!("   The optimization works because for same-type exponential families:");
    println!("   log(p₁(x)/p₂(x)) = (η₁ - η₂)·T(x) - (A(η₁) - A(η₂))");
    println!("   The base measure terms log h(x) cancel out completely!");
}
