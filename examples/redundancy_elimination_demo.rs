//! Demonstration of redundancy elimination in `ExponentialFamilyCache`.
//!
//! This example shows how the new design implements the exponential family
//! log-density computation once in the trait, eliminating redundancy across
//! different distributions.

use measures::distributions::continuous::normal::{Normal, NormalCache};
use measures::distributions::discrete::poisson::{Poisson, PoissonCache};
use measures::exponential_family::ExponentialFamilyCache;

fn main() {
    println!("=== Redundancy Elimination Demo ===\n");

    println!("The ExponentialFamilyCache trait now implements log-density computation");
    println!("once for ALL exponential families using the formula:");
    println!("log p(x|θ) = η·T(x) - A(η) + log h(x)\n");

    // Normal Distribution
    println!("1. Normal Distribution Cache:");
    let normal = Normal::new(1.0, 2.0);
    let normal_cache = NormalCache::from_distribution(&normal);

    println!("   - Natural params: {:?}", normal_cache.natural_params());
    println!("   - Log partition: {:.6}", normal_cache.log_partition());

    let x = 0.5;
    let normal_density = normal_cache.log_density(&x);
    println!("   - Log density at {x}: {normal_density:.6}");

    // Poisson Distribution
    println!("\n2. Poisson Distribution Cache:");
    let poisson = Poisson::new(2.5);
    let poisson_cache = PoissonCache::from_distribution(&poisson);

    println!("   - Natural params: {:?}", poisson_cache.natural_params());
    println!("   - Log partition: {:.6}", poisson_cache.log_partition());

    let k = 3u64;
    let poisson_density = poisson_cache.log_density(&k);
    println!("   - Log density at {k}: {poisson_density:.6}");

    println!("\n=== Implementation Simplicity ===\n");

    println!("Each cache implementation is now extremely simple:");
    println!("✓ Stores cached values (natural params, log partition, base measure)");
    println!("✓ Provides accessors to cached values");
    println!("✓ NO redundant log_density implementations!");
    println!("✓ Single generic implementation in the trait handles all distributions");

    println!("\nCache implementation for Normal:");
    println!("```rust");
    println!("impl ExponentialFamilyCache<T, T> for NormalCache<T> {{");
    println!("    type Distribution = Normal<T>;");
    println!(
        "    fn from_distribution(dist: &Self::Distribution) -> Self {{ Self::new(dist.mean, dist.std_dev) }}"
    );
    println!("    fn log_partition(&self) -> T {{ self.log_partition }}");
    println!("    fn natural_params(&self) -> &[T; 2] {{ &self.natural_params }}");
    println!("    fn base_measure(&self) -> &LebesgueMeasure<T> {{ &self.base_measure }}");
    println!("    // log_density method is provided by the trait's default implementation!");
    println!("}}");
    println!("```");

    println!("\nCache implementation for Poisson:");
    println!("```rust");
    println!("impl ExponentialFamilyCache<u64, F> for PoissonCache<F> {{");
    println!("    type Distribution = Poisson<F>;");
    println!(
        "    fn from_distribution(dist: &Self::Distribution) -> Self {{ Self::new(dist.lambda) }}"
    );
    println!("    fn log_partition(&self) -> F {{ self.log_partition }}");
    println!("    fn natural_params(&self) -> &[F; 1] {{ &self.natural_param }}");
    println!("    fn base_measure(&self) -> &FactorialMeasure<F> {{ &self.base_measure }}");
    println!("    // log_density method is provided by the trait's default implementation!");
    println!("}}");
    println!("```");

    println!("\n=== Benefits ===\n");
    println!("✓ Zero redundancy: exponential family logic implemented once");
    println!("✓ Automatic correctness: all distributions use the same proven formula");
    println!("✓ Easy to add new distributions: just implement the storage accessors");
    println!("✓ Performance: cached values eliminate repeated computations");
    println!("✓ Clean API: same interface works for all exponential families");

    // Demonstrate batch operations work the same for both
    println!("\n=== Batch Operations (Same API for All Distributions) ===\n");

    let normal_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let normal_batch = normal_cache.log_density_batch(&normal_points);
    println!("Normal batch results: {normal_batch:?}");

    let poisson_points = vec![0u64, 1, 2, 3, 4];
    let poisson_batch = poisson_cache.log_density_batch(&poisson_points);
    println!("Poisson batch results: {poisson_batch:?}");
}
