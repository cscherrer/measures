//! Optimization Approaches Comparison
//!
//! This example demonstrates available optimization approaches for computing
//! relative densities between exponential family distributions, showing their
//! performance characteristics and current implementation status.
//!
//! Run with: cargo run --example `optimization_comparison` --features jit --release

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::{LogDensityBuilder, Normal};
use std::time::Instant;

fn main() {
    println!("=== Optimization Approaches Comparison ===\n");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let x = 0.5;

    println!("Computing log(Normal(0,1)/Normal(1,1.5)) at x = {x}\n");

    demonstrate_approaches(&normal1, &normal2, &x);
    benchmark_performance(&normal1, &normal2);
    explain_technical_differences();

    println!("\n=== Comparison Complete ===");
    println!("See docs/OPTIMIZATION_APPROACHES.md for detailed technical analysis!");
}

fn demonstrate_approaches(normal1: &Normal<f64>, normal2: &Normal<f64>, x: &f64) {
    println!("=== Available Optimization Approaches ===\n");

    println!("=== 1. Manual Subtraction ===");
    println!("Approach: Compute each log-density separately and subtract");
    println!("Code:     normal1.log_density().at(&x) - normal2.log_density().at(&x)");

    let manual_result = normal1.log_density().at(x) - normal2.log_density().at(x);
    println!("Result:   {manual_result:.10}");
    println!("Status:   Working but inefficient");
    println!("Issues:   Redundant computation, error-prone");
    println!();

    println!("=== 2. Builder Pattern ===");
    println!("Approach: Use fluent API for relative density computation");
    println!("Code:     normal1.log_density().wrt(normal2).at(&x)");

    let builder_result: f64 = normal1.log_density().wrt(normal2.clone()).at(x);
    println!("Result:   {builder_result:.10}");
    println!("Status:   Working, recommended for general use");
    println!("Benefits: Clean API, type-safe, consistent interface");
    println!("Limitations: Uses general computation path");
    println!();

    println!("=== 3. Zero-Overhead Optimization ===");
    println!("Approach: Pre-compute constants, return optimized closure");
    println!("Code:     let f = normal1.zero_overhead_optimize_wrt(normal2); f(&x)");

    let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let zero_overhead_result = zero_overhead_fn(x);
    println!("Result:   {zero_overhead_result:.10}");
    println!("Status:   Working, good for performance-critical code");
    println!("Benefits: Reduced overhead, LLVM optimizable");
    println!("Limitations: Still computes base measures, requires explicit function generation");
    println!();

    println!("=== 4. JIT Compilation (Cranelift) ===");
    println!("Approach: Compile to native machine code at runtime");
    println!("Code:     let jit = normal.compile_jit()?; jit.call(x)");

    #[cfg(feature = "jit")]
    {
        println!(
            "Status:   EXPERIMENTAL - Infrastructure exists but uses placeholder implementations"
        );
        println!(
            "Result:   Would return incorrect values (ln/exp functions use sqrt placeholders)"
        );
    }
    #[cfg(not(feature = "jit"))]
    {
        println!("Status:   JIT feature not enabled");
    }
    println!("Current limitations: Placeholder math functions, performance overhead");
    println!("Potential benefits: Native code generation, CPU-specific optimizations");
    println!();

    println!("=== 5. Rust Specialization (Future) ===");
    println!("Approach: Automatic optimization in builder pattern");
    println!("Code:     normal1.log_density().wrt(normal2).at(&x)  // Auto-optimized!");
    println!("Status:   Waiting for Rust language feature (RFC 1210)");
    println!("Benefits: Automatic optimization, zero-cost abstraction");
    println!("Limitations: Uncertain timeline, requires language changes");
    println!();

    // Verify working approaches give the same result
    let epsilon = 1e-10;
    assert!((manual_result - builder_result).abs() < epsilon);
    assert!((manual_result - zero_overhead_result).abs() < epsilon);
    println!("All working approaches produce identical results (within {epsilon:.0e})");
}

fn benchmark_performance(normal1: &Normal<f64>, normal2: &Normal<f64>) {
    println!("\n=== Performance Benchmark ===");

    let iterations = 100_000;
    let x = 0.5;

    println!("Test: {iterations} evaluations of relative density");
    println!("Hardware: Modern x86-64 CPU with LLVM optimizations");
    println!();

    // Benchmark 1: Manual subtraction
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
    }
    let manual_time = start.elapsed();

    // Benchmark 2: Builder pattern
    let start = Instant::now();
    for _ in 0..iterations {
        let _result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    }
    let builder_time = start.elapsed();

    // Benchmark 3: Zero-overhead optimization
    let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = zero_overhead_fn(&x);
    }
    let zero_overhead_time = start.elapsed();

    // Results
    println!("Results:");
    println!(
        "  Manual Subtraction:     {:>8.2}µs  (1.0x baseline)",
        manual_time.as_micros()
    );
    println!(
        "  Builder Pattern:        {:>8.2}µs  ({:.1}x vs manual)",
        builder_time.as_micros(),
        manual_time.as_nanos() as f64 / builder_time.as_nanos() as f64
    );
    println!(
        "  Zero-Overhead:          {:>8.2}µs  ({:.1}x vs manual)",
        zero_overhead_time.as_micros(),
        manual_time.as_nanos() as f64 / zero_overhead_time.as_nanos() as f64
    );
    println!(
        "  JIT (theoretical):      {:>8.2}µs  (theoretical estimate)",
        manual_time.as_micros() / 25,
    );

    println!("\nPerformance Analysis:");
    println!("• Zero-overhead optimization provides best available performance");
    println!("• Builder pattern offers good performance with clean API");
    println!("• Manual subtraction wastes computation on redundant base measures");
    println!("• JIT compilation currently has overhead due to implementation limitations");
}

fn explain_technical_differences() {
    println!("\n=== Technical Differences ===");

    println!("\nMathematical Foundation:");
    println!("For exponential families: log(p₁(x)/p₂(x)) = (η₁-η₂)·T(x) - (A(η₁)-A(η₂))");
    println!("Key insight: Base measure terms log h(x) cancel out completely!");
    println!();

    println!("Computational Complexity:");
    println!("┌─────────────────────┬──────────────┬─────────────────┬─────────────────┐");
    println!("│ Approach            │ Parameters   │ Base Measures   │ Function Calls  │");
    println!("├─────────────────────┼──────────────┼─────────────────┼─────────────────┤");
    println!("│ Manual Subtraction  │ 2x computed  │ 2x computed     │ High overhead   │");
    println!("│ Builder Pattern     │ 2x computed  │ 2x computed     │ Medium overhead │");
    println!("│ Zero-Overhead       │ 1x precomp   │ 2x computed     │ Low overhead    │");
    println!("│ JIT (when ready)    │ Embedded     │ Optimized away  │ Zero overhead   │");
    println!("│ Specialization      │ 1x precomp   │ Optimized away  │ Zero overhead   │");
    println!("└─────────────────────┴──────────────┴─────────────────┴─────────────────┘");
    println!();

    println!("When to Use Each:");
    println!("• Manual Subtraction:  Avoid (kept for comparison only)");
    println!("• Builder Pattern:     General use, mixed distribution types");
    println!("• Zero-Overhead:       Performance-critical loops, same family types");
    println!("• JIT:                 Future use when implementation is complete");
    println!("• Specialization:      Future use when Rust supports it");
    println!();

    println!("Optimization Techniques:");
    println!("• Constant pre-computation: Calculate parameters once");
    println!("• Base measure cancellation: Exploit mathematical structure");
    println!("• LLVM optimization: Inlining, vectorization, constant folding");
    println!("• Type-level dispatch: Zero-cost abstraction via monomorphization");
    println!("• Native compilation: Direct machine code generation (future)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_approaches_consistent() {
        let normal1 = Normal::new(2.0, 1.5);
        let normal2 = Normal::new(-1.0, 2.0);
        let x = 1.5;

        let manual = normal1.log_density().at(&x) - normal2.log_density().at(&x);
        let builder: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
        let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
        let zero_overhead = zero_overhead_fn(&x);

        assert!((manual - builder).abs() < 1e-10);
        assert!((manual - zero_overhead).abs() < 1e-10);
    }

    #[test]
    fn test_performance_ordering() {
        // This test verifies that our performance claims are reasonable
        // by checking that zero-overhead is indeed faster than manual subtraction

        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(1.0, 1.5);
        let x = 0.5;
        let iterations = 10_000;

        // Manual approach
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = normal1.log_density().at(&x) - normal2.log_density().at(&x);
        }
        let manual_time = start.elapsed();

        // Zero-overhead approach
        let zero_overhead_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = zero_overhead_fn(&x);
        }
        let zero_overhead_time = start.elapsed();

        // Zero-overhead should be faster (though exact speedup depends on hardware)
        assert!(
            zero_overhead_time < manual_time,
            "Zero-overhead ({:?}) should be faster than manual ({:?})",
            zero_overhead_time,
            manual_time
        );
    }
}
