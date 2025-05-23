//! Demonstration of generic exponential family implementations.
//!
//! This example shows how the `GenericExpFamImpl` trait provides reusable
//! implementations for exponential family distributions, eliminating boilerplate.

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::normal::Normal;
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::GenericExpFamImpl;

fn main() {
    println!("=== Generic Exponential Family Implementation Demo ===\n");

    // Both Normal and Poisson now use the same generic implementation!
    let normal = Normal::new(1.0, 2.0);
    let poisson = Poisson::new(3.0);

    println!("1. Generic Cache Creation:");
    println!("   Both distributions use the same precompute_generic_cache() method");

    // Both use the same generic helper methods
    let normal_cache = normal.precompute_generic_cache();
    let poisson_cache = poisson.precompute_generic_cache();

    println!("   ✅ Normal cache created using generic implementation");
    println!("   ✅ Poisson cache created using generic implementation");

    println!("\n2. Generic Log-Density Computation:");
    println!("   Both distributions use the same cached_log_density_generic() method");

    let normal_density = normal.cached_log_density_generic(&normal_cache, &0.5);
    let poisson_density = poisson.cached_log_density_generic(&poisson_cache, &2u64);

    println!("   Normal log-density at x=0.5: {normal_density:.6}");
    println!("   Poisson log-density at k=2: {poisson_density:.6}");

    println!("\n3. Distribution Implementation Simplification:");
    println!("   Before GenericExpFamImpl:");
    println!("   ```rust");
    println!("   fn precompute_cache(&self) -> Self::Cache {{");
    println!("       GenericExpFamCache::new(self)  // Manual call");
    println!("   }}");
    println!("   fn cached_log_density(&self, cache: &Self::Cache, x: &X) -> F {{");
    println!("       cache.log_density(x)  // Manual delegation");
    println!("   }}");
    println!("   ```");
    println!();
    println!("   After GenericExpFamImpl:");
    println!("   ```rust");
    println!("   fn precompute_cache(&self) -> Self::Cache {{");
    println!("       self.precompute_generic_cache()  // Generic helper");
    println!("   }}");
    println!("   fn cached_log_density(&self, cache: &Self::Cache, x: &X) -> F {{");
    println!("       self.cached_log_density_generic(cache, x)  // Generic helper");
    println!("   }}");
    println!("   ```");

    println!("\n4. Verification:");
    println!("   Generic implementations produce identical results to direct computation");

    let normal_direct: f64 = normal.log_density().at(&0.5);
    let poisson_direct: f64 = poisson.log_density().at(&2u64);

    let normal_diff: f64 = (normal_density - normal_direct).abs();
    let poisson_diff: f64 = (poisson_density - poisson_direct).abs();

    println!(
        "   Normal: generic={normal_density:.10}, direct={normal_direct:.10}, diff={normal_diff:.2e}"
    );
    println!(
        "   Poisson: generic={poisson_density:.10}, direct={poisson_direct:.10}, diff={poisson_diff:.2e}"
    );

    assert!(normal_diff < 1e-10, "Normal results should match");
    assert!(poisson_diff < 1e-10, "Poisson results should match");

    println!("\n   ✅ All results match perfectly!");
    println!("\n=== Generic Implementation Success! ===");
    println!("• Eliminated boilerplate in distribution implementations");
    println!("• Single source of truth for GenericExpFamCache usage");
    println!("• Consistent interface across all exponential families");
    println!("• Zero runtime overhead - all methods inline");
}
