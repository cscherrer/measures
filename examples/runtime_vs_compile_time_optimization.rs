//! Runtime vs Compile-time Optimization Analysis
//!
//! This example demonstrates why compile-time optimization (zero-overhead)
//! is preferred over runtime type checking for exponential family optimization.
//!
//! Run with: cargo run --example runtime_vs_compile_time_optimization --features jit

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::{LogDensityBuilder, Normal};
use std::time::Instant;

fn main() {
    println!("⚡ === Runtime vs Compile-time Optimization Analysis === ⚡\n");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let x = 0.5;

    // Compile-time optimization (zero-overhead)
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let optimized_result: f64 = optimized_fn(&x);

    // Standard builder pattern
    let builder_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);

    println!("🎯 Results Comparison:");
    println!("Zero-overhead optimize: {optimized_result:.10}");
    println!("Builder pattern:        {builder_result:.10}");
    println!("Difference:             {:.2e}", (optimized_result - builder_result).abs());

    println!("\n📊 Performance Analysis:");
    benchmark_approaches(&normal1, &normal2, &x);

    println!("\n🔍 Why Compile-time Optimization Wins:");
    println!("  ✅ Zero runtime overhead");
    println!("  ✅ All constants pre-computed");
    println!("  ✅ LLVM can fully inline and optimize");
    println!("  ✅ Type-safe at compile time");
    println!("  ✅ No dynamic dispatch costs");

    println!("\n❌ Runtime Type Checking Drawbacks:");
    println!("  • TypeId::of() overhead on every call");
    println!("  • Dynamic dispatch costs");
    println!("  • Still can't access optimized functions due to type constraints");
    println!("  • Violates zero-cost abstraction principle");

    println!("\n🎯 Conclusion:");
    println!("  • Zero-overhead optimization: Best performance");
    println!("  • Builder pattern: Best convenience");
    println!("  • Runtime checking: Not recommended");
    println!("  • Specialization would combine best of both worlds");
}

fn benchmark_approaches(normal1: &Normal<f64>, normal2: &Normal<f64>, x: &f64) {
    let iterations = 100_000;
    
    // Benchmark zero-overhead optimization
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = optimized_fn(x);
    }
    let zero_overhead_time = start.elapsed();

    // Benchmark builder pattern
    let start = Instant::now();
    for _ in 0..iterations {
        let _result: f64 = normal1.log_density().wrt(normal2.clone()).at(x);
    }
    let builder_time = start.elapsed();

    // Benchmark manual subtraction (baseline)
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = normal1.log_density().at(x) - normal2.log_density().at(x);
    }
    let manual_time = start.elapsed();

    println!("Benchmark results ({iterations} iterations):");
    println!("  Zero-overhead optimize: {:.2}µs", zero_overhead_time.as_micros());
    println!("  Builder pattern:        {:.2}µs", builder_time.as_micros());
    println!("  Manual subtraction:     {:.2}µs", manual_time.as_micros());

    let zero_overhead_speedup = manual_time.as_nanos() as f64 / zero_overhead_time.as_nanos() as f64;
    let builder_speedup = manual_time.as_nanos() as f64 / builder_time.as_nanos() as f64;

    println!("\nSpeedup vs manual subtraction:");
    println!("  Zero-overhead optimize: {zero_overhead_speedup:.2}x");
    println!("  Builder pattern:        {builder_speedup:.2}x");
} 