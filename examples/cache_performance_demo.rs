//! Demonstration of exponential family cache performance benefits.
//!
//! This example shows how cached computations can significantly improve
//! performance for repeated log-density evaluations.

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::normal::Normal;
use measures::exponential_family::{ExponentialFamilyCache, GenericExpFamCache};
use std::time::Instant;

/// Benchmark direct vs cached computation for Normal distribution
fn benchmark_normal_caching() {
    let normal = Normal::new(1.0, 2.0);
    let test_points: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();

    // Direct computation (no caching)
    let start = Instant::now();
    let _direct_results: Vec<f64> = test_points
        .iter()
        .map(|&x| normal.log_density().at(&x))
        .collect();
    let direct_time = start.elapsed();

    // Cached computation
    let cache: GenericExpFamCache<Normal<f64>, f64, f64> = GenericExpFamCache::from_distribution(&normal);
    let start = Instant::now();
    let _cached_results: Vec<f64> = test_points.iter().map(|&x| cache.log_density(&x)).collect();
    let cached_time = start.elapsed();

    println!("Direct computation: {:?}", direct_time);
    println!("Cached computation: {:?}", cached_time);
    println!(
        "Speedup: {:.2}x",
        direct_time.as_nanos() as f64 / cached_time.as_nanos() as f64
    );
}

fn main() {
    println!("=== Exponential Family Cache Performance Demo ===\n");

    println!("Benchmarking Normal distribution log-density computation:");
    benchmark_normal_caching();

    println!("\n=== Cache eliminates redundant computation ===");
    println!("• Natural parameters computed once and reused");
    println!("• Log partition function cached");
    println!("• Base measure computation optimized");
    println!("• Perfect for batch operations!");
}
