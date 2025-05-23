//! Example demonstrating the general exponential family caching framework.
//!
//! This example shows how the caching framework provides consistent optimization
//! across different exponential family distributions, eliminating redundant
//! computations for both single evaluations and batch operations.

use measures::exponential_family::ExponentialFamily;
use measures::{HasLogDensity, Normal, distributions::discrete::poisson::Poisson};

fn main() {
    println!("=== Exponential Family Caching Framework Demo ===\n");

    // Example 1: Normal distribution caching
    demo_normal_caching();

    // Example 2: Poisson distribution caching
    demo_poisson_caching();

    // Example 3: Consistent interface across distributions
    demo_unified_interface();
}

fn demo_normal_caching() {
    println!("1. Normal Distribution Caching:");

    let normal = Normal::new(2.0, 1.5);
    let data_points = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];

    // Method 1: Standard (uncached) evaluation - computes σ² repeatedly
    println!("   Standard evaluation (recomputes values):");
    for &x in &data_points {
        let ld: f64 = normal.log_density_wrt_root(&x);
        println!("     f({x}) = {ld:.6}");
    }

    // Method 2: Cached batch evaluation - computes σ² once
    println!("   Cached batch evaluation (values computed once):");
    let batch_results = normal.cached_log_density_batch(&data_points);
    for (x, ld) in data_points.iter().zip(batch_results.iter()) {
        println!("     f({x}) = {ld:.6}");
    }

    // Method 3: Manual caching for custom operations
    println!("   Manual caching (for custom loops):");
    let cache = normal.precompute_cache();
    for &x in &data_points {
        let ld: f64 = normal.cached_log_density(&cache, &x);
        println!("     f({x}) = {ld:.6}");
    }

    // Method 4: Functional style with cached closure
    println!("   Functional style with cached closure:");
    let log_density_fn = normal.cached_log_density_fn();
    let functional_results: Vec<_> = data_points.iter().map(log_density_fn).collect();
    for (x, ld) in data_points.iter().zip(functional_results.iter()) {
        println!("     f({x}) = {ld:.6}");
    }

    println!();
}

fn demo_poisson_caching() {
    println!("2. Poisson Distribution Caching:");

    let poisson = Poisson::new(3.5);
    let counts = vec![0u64, 1, 2, 3, 4, 5, 6];

    // Method 1: Standard (uncached) evaluation
    println!("   Standard evaluation (recomputes ln(λ) repeatedly):");
    for &k in &counts {
        let ld: f64 = poisson.log_density_wrt_root(&k);
        println!("     P(X={}) = exp({:.6}) = {:.6}", k, ld, ld.exp());
    }

    // Method 2: Cached batch evaluation
    println!("   Cached batch evaluation (ln(λ) computed once):");
    let batch_results = poisson.cached_log_density_batch(&counts);
    for (k, ld) in counts.iter().zip(batch_results.iter()) {
        println!("     P(X={}) = exp({:.6}) = {:.6}", k, ld, ld.exp());
    }

    // Method 3: Manual caching
    println!("   Manual caching (for custom loops):");
    let cache = poisson.precompute_cache();
    for &k in &counts {
        let ld: f64 = poisson.cached_log_density(&cache, &k);
        println!("     P(X={}) = exp({:.6}) = {:.6}", k, ld, ld.exp());
    }

    println!();
}

fn demo_unified_interface() {
    println!("3. Unified Caching Interface:");
    println!("   All exponential families support the same caching methods:");

    let normal = Normal::new(0.0, 1.0);
    let poisson = Poisson::new(2.0);

    // Both distributions implement the same ExponentialFamily trait
    println!("   Both Normal and Poisson implement ExponentialFamily with caching");

    // Both support the same caching methods
    let normal_cache = normal.precompute_cache();
    let poisson_cache = poisson.precompute_cache();

    let normal_ld: f64 = normal.cached_log_density(&normal_cache, &0.0);
    let poisson_ld: f64 = poisson.cached_log_density(&poisson_cache, &2);

    println!("   Normal cached log-density at x=0: {normal_ld:.6}");
    println!("   Poisson cached log-density at k=2: {poisson_ld:.6}");

    // Both support batch operations
    let normal_batch = normal.cached_log_density_batch(&[0.0, 1.0, 2.0]);
    let poisson_batch = poisson.cached_log_density_batch(&[0, 1, 2]);

    println!("   Normal batch results: {normal_batch:?}");
    println!("   Poisson batch results: {poisson_batch:?}");

    println!("\n   ✅ Consistent caching interface across all exponential families!");
}
