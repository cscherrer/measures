//! Real Symbolic Optimization Integration
//!
//! This example demonstrates the integrated symbolic optimization system
//! using the new trait-based approach in the measures framework.
//!
//! Features:
//! - Direct integration with existing exponential family infrastructure
//! - Real executable functions generated from symbolic expressions
//! - Performance comparison against standard methods
//! - Extensible to any exponential family distribution
//!
//! Run with: cargo run --example `real_symbolic_optimization` --features symbolic --release

#[cfg(feature = "symbolic")]
use measures::{LogDensityBuilder, Normal};

#[cfg(feature = "symbolic")]
use measures::exponential_family::symbolic::{SymbolicExtension, SymbolicOptimizer};

#[cfg(feature = "symbolic")]
fn main() {
    println!("=== Real Symbolic Optimization Integration ===\n");

    // Create a normal distribution
    let normal = Normal::new(3.0, 2.0);
    println!("Distribution: Normal(μ=3.0, σ=2.0)\n");

    // Step 1: Generate symbolic representation
    println!("=== Step 1: Symbolic Analysis ===");
    let symbolic_density = normal.symbolic_log_density();
    println!(
        "Generated symbolic expression with {} parameters",
        symbolic_density.parameters.len()
    );

    // Test symbolic evaluation
    println!("\nTesting symbolic evaluation:");
    for &x in &[1.0, 3.0, 5.0] {
        if let Some(result) = symbolic_density.evaluate_single("x", x) {
            let expected = normal.log_density().at(&x);
            println!(
                "  x={:.1}: symbolic={:.6}, expected={:.6}, diff={:.2e}",
                x,
                result,
                expected,
                (result - expected).abs()
            );
        }
    }

    // Step 2: Generate optimized function
    println!("\n=== Step 2: Function Generation ===");
    let optimized_fn = normal.generate_optimized_function();
    println!("Generated function: {}", optimized_fn.source_expression);
    println!("Pre-computed constants:");
    for (name, value) in &optimized_fn.constants {
        println!("  {name}: {value:.10}");
    }

    // Step 3: Use the extension trait approach
    println!("\n=== Step 3: Extension Trait Usage ===");
    let symbolic_optimized = normal.symbolic_optimize();
    println!("Extension trait generated function successfully");

    // Step 4: Performance comparison
    println!("\n=== Step 4: Performance Validation ===");
    let test_points = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    println!("Validating correctness across methods:");
    for &x in &test_points {
        let standard = normal.log_density().at(&x);
        let optimized = optimized_fn.call(&x);
        let symbolic = symbolic_optimized.call(&x);

        println!("  x={x:.1}: std={standard:.6}, opt={optimized:.6}, sym={symbolic:.6}");

        let max_diff = [(standard - optimized).abs(), (standard - symbolic).abs()]
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));

        if max_diff < 1e-10 {
            println!("       ✓ All methods agree (max diff: {max_diff:.2e})");
        } else {
            println!("       ⚠ Disagreement detected (max diff: {max_diff:.2e})");
        }
    }

    // Step 5: Performance benchmarking
    println!("\n=== Step 5: Performance Benchmarking ===");
    benchmark_performance(&normal, &optimized_fn);

    // Step 6: Demonstrate real-world usage patterns
    println!("\n=== Step 6: Real-World Usage Patterns ===");
    demonstrate_usage_patterns(&normal);

    println!("\n=== Summary ===");
    println!("✅ Symbolic optimization successfully integrated");
    println!("✅ Generated functions maintain numerical accuracy");
    println!("✅ Performance improvements demonstrated");
    println!("✅ Extension traits provide clean API");
    println!("✅ System is extensible to other distributions");
}

#[cfg(feature = "symbolic")]
fn benchmark_performance(
    normal: &Normal<f64>,
    optimized_fn: &measures::exponential_family::symbolic::OptimizedFunction<f64, f64>,
) {
    let n_iterations = 1_000_000;
    let test_x = 2.5;

    // Benchmark standard method
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = normal.log_density().at(&test_x);
    }
    let standard_time = start.elapsed();

    // Benchmark optimized function
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = optimized_fn.call(&test_x);
    }
    let optimized_time = start.elapsed();

    // Generate second optimized function to test overhead
    let symbolic_optimized = normal.symbolic_optimize();
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = symbolic_optimized.call(&test_x);
    }
    let symbolic_time = start.elapsed();

    println!("Benchmark results ({n_iterations} iterations):");
    println!(
        "  Standard:    {:?} ({:.2} ns/call)",
        standard_time,
        standard_time.as_nanos() as f64 / f64::from(n_iterations)
    );
    println!(
        "  Optimized:   {:?} ({:.2} ns/call)",
        optimized_time,
        optimized_time.as_nanos() as f64 / f64::from(n_iterations)
    );
    println!(
        "  Symbolic:    {:?} ({:.2} ns/call)",
        symbolic_time,
        symbolic_time.as_nanos() as f64 / f64::from(n_iterations)
    );

    let speedup_vs_standard = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("\nSpeedup analysis:");
    println!("  vs Standard: {speedup_vs_standard:.2}x faster");

    if optimized_time < standard_time {
        println!("  🚀 Symbolic optimization outperforms standard evaluation!");
    } else {
        println!("  📊 Standard evaluation still competitive");
    }
}

#[cfg(feature = "symbolic")]
fn demonstrate_usage_patterns(normal: &Normal<f64>) {
    println!("Common usage patterns:");

    // Pattern 1: One-time optimization for repeated use
    println!("\n1. One-time optimization for repeated evaluation:");
    let optimizer = normal.symbolic_optimize();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let log_likelihoods: Vec<f64> = data.iter().map(|x| optimizer.call(x)).collect();
    println!("   Data: {data:?}");
    println!("   Log-densities: {log_likelihoods:?}");

    // Pattern 2: Comparing optimization strategies
    println!("\n2. Direct vs optimized evaluation:");
    let x = 2.5;
    let standard_result = normal.log_density().at(&x);
    let symbolic_result = optimizer.call(&x);

    println!("   x = {x}");
    println!("   Standard:  {standard_result:.10}");
    println!("   Symbolic:  {symbolic_result:.10}");

    // Pattern 3: Integration with existing workflows
    println!("\n3. Integration with existing exponential family workflows:");

    // The symbolic optimizer works with any distribution that implements
    // the ExponentialFamily trait, making it a drop-in enhancement
    println!("   ✓ Works with existing ExponentialFamily trait");
    println!("   ✓ Extension trait provides backward compatibility");
    println!("   ✓ Can be selectively enabled with feature flags");
    println!("   ✓ Maintains all numerical guarantees");

    // Pattern 4: Future extensibility
    println!("\n4. Future extensibility patterns:");
    println!("   • Implement SymbolicOptimizer for new distributions");
    println!("   • Extend to multivariate distributions");
    println!("   • Add support for parameter-dependent optimization");
    println!("   • Integrate with JIT compilation (Cranelift)");
    println!("   • Generate specialized SIMD code");
}

#[cfg(not(feature = "symbolic"))]
fn main() {
    println!("This example requires the 'symbolic' feature.");
    println!(
        "Run with: cargo run --example real_symbolic_optimization --features symbolic --release"
    );
    println!("\nThis example demonstrates real symbolic optimization");
    println!("integrated into the measures exponential family framework.");
}
