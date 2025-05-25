//! Runtime vs Compile-time Optimization Analysis
//!
//! Compares zero-overhead optimization with the standard builder pattern
//! to demonstrate performance differences.

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::{LogDensityBuilder, Normal};
use std::time::Instant;

fn main() {
    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let x = 0.5;

    // Compare approaches
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let optimized_result: f64 = optimized_fn(&x);
    let builder_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);

    println!("Results:");
    println!("  Zero-overhead: {:.10}", optimized_result);
    println!("  Builder:       {:.10}", builder_result);
    println!("  Difference:    {:.2e}", (optimized_result - builder_result).abs());

    // Benchmark performance
    benchmark_approaches(&normal1, &normal2, &x);
}

fn benchmark_approaches(normal1: &Normal<f64>, normal2: &Normal<f64>, x: &f64) {
    let iterations = 100_000;

    // Zero-overhead optimization
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = optimized_fn(x);
    }
    let zero_overhead_time = start.elapsed();

    // Builder pattern
    let start = Instant::now();
    for _ in 0..iterations {
        let _result: f64 = normal1.log_density().wrt(normal2.clone()).at(x);
    }
    let builder_time = start.elapsed();

    // Manual subtraction (baseline)
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = normal1.log_density().at(x) - normal2.log_density().at(x);
    }
    let manual_time = start.elapsed();

    println!("\nBenchmark ({} iterations):", iterations);
    println!("  Zero-overhead: {:.2}µs", zero_overhead_time.as_micros());
    println!("  Builder:       {:.2}µs", builder_time.as_micros());
    println!("  Manual:        {:.2}µs", manual_time.as_micros());

    let zero_overhead_speedup = manual_time.as_nanos() as f64 / zero_overhead_time.as_nanos() as f64;
    let builder_speedup = manual_time.as_nanos() as f64 / builder_time.as_nanos() as f64;

    println!("\nSpeedup vs manual:");
    println!("  Zero-overhead: {:.2}x", zero_overhead_speedup);
    println!("  Builder:       {:.2}x", builder_speedup);
}
