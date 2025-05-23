//! Demonstration of efficient exponential family cache implementation.
//!
//! This example shows how the new `natural_and_log_partition` method eliminates
//! redundant computation during cache creation.

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::normal::Normal;
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::{ExponentialFamilyCache, GenericExpFamCache};

fn main() {
    println!("=== Efficient Exponential Family Cache Demo ===\n");

    let normal = Normal::new(1.0, 2.0);
    let poisson = Poisson::new(3.0);

    println!("1. Normal Distribution Cache Creation:");
    println!("   The Normal distribution overrides `natural_and_log_partition`");
    println!("   to compute variance (σ²) and mean² only once, not twice!");

    // Cache creation now calls the efficient method
    let normal_cache: GenericExpFamCache<Normal<f64>, f64, f64> =
        GenericExpFamCache::from_distribution(&normal);

    println!("   ✅ Cache created with zero redundant computation");

    println!("\n2. Poisson Distribution Cache Creation:");
    println!("   Poisson uses the default implementation (no shared computation)");

    let poisson_cache: GenericExpFamCache<Poisson<f64>, u64, f64> =
        GenericExpFamCache::from_distribution(&poisson);

    println!("   ✅ Cache created using default method");

    println!("\n3. Performance Benefits:");
    println!("   Normal: One call to natural_and_log_partition → σ² computed once");
    println!("   Previous: Two separate calls → σ² computed twice");

    // Demonstrate that cached computation works correctly
    let normal_density = normal_cache.log_density(&0.5);
    let poisson_density = poisson_cache.log_density(&2u64);

    // Verify against direct computation
    let normal_direct = normal.log_density().at(&0.5);
    let poisson_direct = poisson.log_density().at(&2u64);

    println!("\n4. Correctness Verification:");
    println!("   Normal cached: {normal_density:.6}");
    println!("   Normal direct: {normal_direct:.6}");
    println!(
        "   Difference: {:.2e}",
        (normal_density - normal_direct).abs()
    );

    println!("   Poisson cached: {poisson_density:.6}");
    println!("   Poisson direct: {poisson_direct:.6}");
    println!(
        "   Difference: {:.2e}",
        (poisson_density - poisson_direct).abs()
    );

    println!("\n✅ Efficient cache implementation eliminates redundant computation!");
    println!("   Distributions can override for shared computations (Normal does)");
    println!("   Or use default implementation when no sharing is needed (Poisson)");
}
