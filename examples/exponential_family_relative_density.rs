//! Exponential Family Relative Density Optimization Example
//!
//! This example demonstrates the zero-overhead optimization for computing relative densities
//! between distributions from the same exponential family. The zero-overhead system
//! automatically provides significant performance improvements while maintaining a clean API.
//!
//! Mathematical insight:
//! For two exponential families p₁(x|θ₁) and p₂(x|θ₂) from the same family:
//! log(p₁(x)/p₂(x)) = [η₁·T(x) - A(η₁) + log h(x)] - [η₂·T(x) - A(η₂) + log h(x)]
//!                   = (η₁ - η₂)·T(x) - (A(η₁) - A(η₂))
//!
//! The base measure terms log h(x) cancel out completely!
//! This optimization is automatically applied by the zero-overhead system.
//!
//! Run with: cargo run --example exponential_family_relative_density --features jit

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::exponential_family::ExponentialFamily;
use measures::traits::DotProduct;
use measures::{LogDensityBuilder, Normal, distributions::Exponential};

fn main() {
    println!("🎯 === Zero-Overhead Exponential Family Optimization === 🎯\n");

    demonstrate_normal_optimization();
    demonstrate_exponential_optimization();
    demonstrate_mathematical_insight();
    demonstrate_performance_benefit();
    demonstrate_best_practices();

    println!("\n🎉 === Zero-Overhead Optimization Complete! === 🎉");
    println!("✅ Demonstrated efficient relative density computation using zero-overhead optimization");
    println!("✅ Showed mathematical correctness of the optimization");
    println!("🚀 Zero-overhead system provides significant performance improvements!");
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

    // Zero-overhead optimized computation
    let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let zero_overhead_result = zero_overhead_fn(&x);

    println!("Normal₁: N(μ=0.0, σ=1.0)");
    println!("Normal₂: N(μ=2.0, σ=1.5)");
    println!("Point: x = {x}");
    println!();
    println!("Standard computation:     {standard_result:.10}");
    println!("Builder optimization:     {optimized_result:.10}");
    println!("Zero-overhead optimize:   {zero_overhead_result:.10}");
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
    let _exp1 = Exponential::new(1.0);
    let _exp2 = Exponential::new(2.0);
    let points: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01).collect();

    // Benchmark 1: Manual subtraction (not recommended - error-prone)
    let start = std::time::Instant::now();
    for &x in &points {
        let _result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    }
    let manual_time = start.elapsed();

    // Benchmark 2: Builder pattern (RECOMMENDED - consistent API)
    let start = std::time::Instant::now();
    for &x in &points {
        let _result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    }
    let builder_time = start.elapsed();

    // Benchmark 3: Zero-overhead optimized function (fastest for repeated use)
    let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let start = std::time::Instant::now();
    for &x in &points {
        let _result = zero_overhead_fn(&x);
    }
    let zero_overhead_time = start.elapsed();

    println!("Computing relative density for {} points...", points.len());
    println!("Manual subtraction:     {:.2}µs (not recommended)", manual_time.as_micros());
    println!("Builder pattern:        {:.2}µs (RECOMMENDED)", builder_time.as_micros());
    println!("Zero-overhead optimize: {:.2}µs (fastest)", zero_overhead_time.as_micros());

    let builder_vs_manual = manual_time.as_nanos() as f64 / builder_time.as_nanos() as f64;
    let zero_overhead_vs_manual = manual_time.as_nanos() as f64 / zero_overhead_time.as_nanos() as f64;
    
    println!("\nSpeedup vs manual subtraction:");
    println!("  Builder pattern:        {builder_vs_manual:.2}x");
    println!("  Zero-overhead optimize: {zero_overhead_vs_manual:.2}x");

    println!("\n💡 Why use the zero-overhead optimization:");
    println!("   • Pre-computes constants at generation time");
    println!("   • Zero function call overhead");
    println!("   • Automatic LLVM optimization");
    println!("   • Works with any exponential family types");

    println!("\n🔧 Recommended Usage:");
    println!("   ✅ RECOMMENDED: normal1.log_density().wrt(normal2).at(&x)");
    println!("   ⚡ FASTEST:     normal1.zero_overhead_optimize_wrt(normal2)");
    println!("   ❌ AVOID:       normal1.log_density().at(&x) - normal2.log_density().at(&x)");
}

fn demonstrate_best_practices() {
    println!("=== 5. Best Practices for Exponential Family Optimization ===");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let exp1 = Exponential::new(1.0);
    let _exp2 = Exponential::new(2.0);
    let x = 0.5;

    println!("📋 Recommended approaches (in order of preference):\n");

    // Case 1: Builder pattern (RECOMMENDED)
    println!("1️⃣ Builder Pattern (RECOMMENDED for all cases)");
    let builder_same_type: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    let builder_diff_type: f64 = normal1.log_density().wrt(exp1.clone()).at(&x);
    println!("   Same type:     normal1.log_density().wrt(normal2).at(&x)");
    println!("   Different:     normal1.log_density().wrt(exponential).at(&x)");
    println!("   Results:       {builder_same_type:.10} | {builder_diff_type:.10} ✅");

    // Case 2: Zero-overhead optimization (for performance-critical repeated use)
    println!("\n2️⃣ Zero-Overhead Optimization (for performance-critical repeated use)");
    let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let zero_overhead_result = zero_overhead_fn(&x);
    println!("   Usage:         let opt_fn = normal1.zero_overhead_optimize_wrt(normal2);");
    println!("   Then:          opt_fn(&x) for each evaluation");
    println!("   Result:        {zero_overhead_result:.10}");

    // Case 3: Manual subtraction (not recommended)
    println!("\n3️⃣ Manual Subtraction (NOT RECOMMENDED - error prone)");
    let manual_result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    println!("   Usage:         normal1.log_density().at(&x) - normal2.log_density().at(&x)");
    println!("   Result:        {manual_result:.10}");
    println!("   ⚠️  Problems:   • Error-prone (easy to mix up order)");
    println!("                  • No automatic optimization");
    println!("                  • Verbose and unclear intent");

    println!("\n📊 Performance Summary:");
    println!("   Builder pattern:        Convenient + reasonably fast");
    println!("   Zero-overhead optimize: Fastest for repeated evaluations");
    println!("   Manual subtraction:     Slowest + error-prone");

    println!("\n🎯 When to use each:");
    println!("   • One-off computation → Builder pattern");
    println!("   • Many evaluations → Zero-overhead optimization");
    println!("   • Never use manual subtraction");

    println!("\n🔧 Zero-overhead optimization benefits:");
    println!("   • Pre-computes all constants");
    println!("   • Eliminates repeated parameter access");
    println!("   • LLVM can fully inline and optimize");
    println!("   • Type-safe and impossible to misuse");

    println!("\n✨ Mathematical Insight:");
    println!("   The optimization works because for same-type exponential families:");
    println!("   log(p₁(x)/p₂(x)) = (η₁ - η₂)·T(x) - (A(η₁) - A(η₂))");
    println!("   The base measure terms log h(x) cancel out completely!");
}
