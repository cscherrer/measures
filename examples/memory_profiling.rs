//! Memory profiling example for the measures crate.
//!
//! Demonstrates DHAT heap profiling integration.
//! Run with: `cargo run --example memory_profiling --features dhat-heap`

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    println!("Memory Profiling Analysis");

    // Test clone-heavy patterns
    test_cloning_patterns();

    // Test allocation-efficient patterns
    test_efficient_patterns();

    // Test factorial computation allocations
    test_factorial_allocations();

    // Test different usage patterns
    test_usage_patterns();

    #[cfg(feature = "dhat-heap")]
    println!("Memory profile saved to dhat-heap.json");

    #[cfg(not(feature = "dhat-heap"))]
    println!("Run with --features dhat-heap for memory profiling");
}

/// Test different cloning patterns to identify hotspots
fn test_cloning_patterns() {
    println!("Testing cloning patterns...");

    let normal = Normal::new(0.0_f64, 1.0_f64);
    let points: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();

    // Pattern 1: Clone LogDensity on every evaluation (worst case)
    let mut sum1 = 0.0;
    for &x in &points {
        let ld = normal.log_density(); // Clone here!
        sum1 += ld.at(&x);
    }

    // Pattern 2: Clone LogDensity once, reuse (better)
    let mut sum2 = 0.0;
    let ld = normal.log_density(); // Clone once
    for &x in &points {
        sum2 += ld.at(&x);
    }

    // Pattern 3: No LogDensity wrapper (best for performance)
    let mut sum3 = 0.0;
    for &x in &points {
        sum3 += normal.log_density_wrt_root(&x);
    }

    println!("Results: {sum1:.3}, {sum2:.3}, {sum3:.3}");
}

/// Test allocation-efficient patterns
fn test_efficient_patterns() {
    println!("Testing efficient patterns...");

    let normal = Normal::new(1.0_f64, 2.0_f64);
    let x = 0.5_f64;

    // Test different computation paths
    for _ in 0..1000 {
        let _ = normal.log_density_wrt_root(&x);
    }

    for _ in 0..1000 {
        let _ = normal.exp_fam_log_density(&x);
    }

    for _ in 0..1000 {
        let nat_params = normal.to_natural();
        let suff_stats = normal.sufficient_statistic(&x);
        let log_partition = normal.log_partition();
        let base_measure = normal.base_measure();
        let base_density: f64 = base_measure.log_density_wrt_root(&x);
        let _ = nat_params[0] * suff_stats[0] + nat_params[1] * suff_stats[1] - log_partition
            + base_density;
    }
}

/// Test factorial computation memory patterns
fn test_factorial_allocations() {
    println!("Testing factorial computations...");

    let poisson = Poisson::new(2.5_f64);

    // Test different k values to see allocation scaling
    for k in [1, 5, 10, 20, 50] {
        for _ in 0..100 {
            let _ = poisson.exp_fam_log_density(&k);
        }
    }

    // Test base measure creation
    for _ in 0..500 {
        let factorial_measure = poisson.base_measure();
        let _: f64 = factorial_measure.log_density_wrt_root(&10u64);
    }
}

/// Test realistic usage patterns
fn test_usage_patterns() {
    println!("Testing realistic usage patterns...");

    // Monte Carlo sampling pattern
    let normal = Normal::new(0.0_f64, 1.0_f64);
    let ld = normal.log_density();
    for i in 0..10_000 {
        let x = f64::from(i) * 0.0001;
        let _ = ld.at(&x);
    }

    // ML optimization pattern
    let distributions = vec![
        Normal::new(0.0, 1.0),
        Normal::new(1.0, 0.5),
        Normal::new(-1.0, 2.0),
    ];
    let data_points: Vec<f64> = (0..500).map(|i| f64::from(i) * 0.01).collect();

    for dist in &distributions {
        for &x in &data_points {
            let _ = dist.log_density_wrt_root(&x);
        }
    }

    // Statistical inference pattern
    let normal1 = Normal::new(0.0_f64, 1.0_f64);
    let normal2 = Normal::new(0.1_f64, 1.1_f64);
    let ld_relative = normal1.log_density().wrt(normal2);

    for i in 0..5_000 {
        let x = f64::from(i) * 0.001;
        let _ = ld_relative.at(&x);
    }

    println!("Usage pattern testing complete");
}
