//! Performance demonstration of the `ExponentialFamilyCache` trait.
//!
//! This example compares the performance of different approaches to
//! computing log-densities for multiple points:
//! 1. Naive approach: recompute everything for each point
//! 2. Old caching approach: `precompute_cache()` + `cached_log_density()`
//! 3. New caching approach: `ExponentialFamilyCache` trait

use measures::core::HasLogDensity;
use measures::distributions::continuous::normal::{Normal, NormalCache};
use measures::exponential_family::{ExponentialFamily, ExponentialFamilyCache};
use std::time::Instant;

fn main() {
    let normal = Normal::new(1.0_f64, 2.0_f64);
    let points: Vec<f64> = (0..10000).map(|i| f64::from(i) * 0.001).collect();

    println!("Computing log-densities for {} points...\n", points.len());

    // Approach 1: Naive - recompute everything each time
    let start = Instant::now();
    let naive_results: Vec<f64> = points
        .iter()
        .map(|&x| normal.log_density_wrt_root(&x))
        .collect();
    let naive_duration = start.elapsed();

    println!("1. Naive approach (recompute everything):");
    println!("   Time: {naive_duration:?}");
    println!("   First 5 results: {:?}\n", &naive_results[0..5]);

    // Approach 2: Old caching (precompute_cache + cached_log_density)
    let start = Instant::now();
    let cache = normal.precompute_cache();
    let old_cache_results: Vec<f64> = points
        .iter()
        .map(|&x| normal.cached_log_density(&cache, &x))
        .collect();
    let old_cache_duration = start.elapsed();

    println!("2. Old caching approach (precompute_cache + cached_log_density):");
    println!("   Time: {old_cache_duration:?}");
    println!("   First 5 results: {:?}\n", &old_cache_results[0..5]);

    // Approach 3: New caching trait
    let start = Instant::now();
    let new_cache = NormalCache::from_distribution(&normal);
    let new_cache_results = new_cache.log_density_batch(&points);
    let new_cache_duration = start.elapsed();

    println!("3. New caching trait approach:");
    println!("   Time: {new_cache_duration:?}");
    println!("   First 5 results: {:?}\n", &new_cache_results[0..5]);

    // Approach 4: New caching trait with closure
    let start = Instant::now();
    let cache_fn = new_cache.log_density_fn();
    let closure_results: Vec<f64> = points.iter().map(cache_fn).collect();
    let closure_duration = start.elapsed();

    println!("4. New caching trait with closure:");
    println!("   Time: {closure_duration:?}");
    println!("   First 5 results: {:?}\n", &closure_results[0..5]);

    // Verify all approaches give the same results
    let all_match = naive_results
        .iter()
        .zip(old_cache_results.iter())
        .zip(new_cache_results.iter())
        .zip(closure_results.iter())
        .all(|(((a, b), c), d)| {
            (a - b).abs() < 1e-10 && (a - c).abs() < 1e-10 && (a - d).abs() < 1e-10
        });

    println!("\nAll approaches produce identical results: {all_match}");

    // Performance comparison
    println!("\nPerformance comparison:");
    println!(
        "  Old cache vs Naive: {:.2}x speedup",
        naive_duration.as_nanos() as f64 / old_cache_duration.as_nanos() as f64
    );
    println!(
        "  New cache vs Naive: {:.2}x speedup",
        naive_duration.as_nanos() as f64 / new_cache_duration.as_nanos() as f64
    );
    println!(
        "  New cache vs Old cache: {:.2}x speedup",
        old_cache_duration.as_nanos() as f64 / new_cache_duration.as_nanos() as f64
    );

    // Demonstrate additional features of the new trait
    println!("\n=== Additional Features ===");

    // Easy access to cached values
    println!("Cached log partition: {:.6}", new_cache.log_partition());
    println!("Cached natural params: {:?}", new_cache.natural_params());

    // Compare with direct computation (which now uses cache internally)
    let direct_log_partition = normal.log_partition();
    println!("Direct log_partition(): {direct_log_partition:.6}");
    println!(
        "Cache matches direct: {}",
        (new_cache.log_partition() - direct_log_partition).abs() < 1e-10
    );
}
