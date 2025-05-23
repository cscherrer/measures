//! Tests for the `ExponentialFamilyCache` trait.
//!
//! This module tests the new caching API that separates cache management
//! from distribution definitions, providing cleaner and more efficient
//! batch operations.

use measures::core::HasLogDensity;
use measures::distributions::continuous::normal::{Normal, NormalCache};
use measures::distributions::discrete::poisson::{Poisson, PoissonCache};
use measures::exponential_family::{ExponentialFamily, ExponentialFamilyCache};

#[test]
fn test_normal_cache_trait_api() {
    let normal = Normal::new(1.0_f64, 2.0_f64);

    // Create cache once using the trait
    let cache = NormalCache::from_distribution(&normal);

    // Reuse cache for multiple computations
    let x1 = 0.5_f64;
    let x2 = 1.5_f64;

    let density1 = cache.log_density(&x1);
    let density2 = cache.log_density(&x2);

    // Verify results match the distribution's direct computation
    let expected1 = normal.log_density_wrt_root(&x1);
    let expected2 = normal.log_density_wrt_root(&x2);

    assert!((density1 - expected1).abs() < 1e-10);
    assert!((density2 - expected2).abs() < 1e-10);
}

#[test]
fn test_poisson_cache_trait_api() {
    let poisson = Poisson::new(2.5_f64);

    // Create cache once using the trait
    let cache = PoissonCache::from_distribution(&poisson);

    // Reuse cache for multiple computations
    let k1 = 1u64;
    let k2 = 3u64;

    let density1 = cache.log_density(&k1);
    let density2 = cache.log_density(&k2);

    // Verify results match the distribution's direct computation
    let expected1 = poisson.log_density_wrt_root(&k1);
    let expected2 = poisson.log_density_wrt_root(&k2);

    assert!((density1 - expected1).abs() < 1e-10);
    assert!((density2 - expected2).abs() < 1e-10);
}

#[test]
fn test_cached_log_partition() {
    let normal = Normal::new(2.0_f64, 1.5_f64);
    let cache = NormalCache::from_distribution(&normal);

    // Get cached log partition
    let cached_log_partition = cache.log_partition();

    // Compare with distribution's log_partition (which now also uses cache internally)
    let direct_log_partition = normal.log_partition();

    assert!((cached_log_partition - direct_log_partition).abs() < 1e-10);
}

#[test]
fn test_poisson_cached_log_partition() {
    let poisson = Poisson::new(3.5_f64);
    let cache = PoissonCache::from_distribution(&poisson);

    // Get cached log partition
    let cached_log_partition = cache.log_partition();

    // Compare with distribution's log_partition (which now also uses cache internally)
    let direct_log_partition = poisson.log_partition();

    assert!((cached_log_partition - direct_log_partition).abs() < 1e-10);
}

#[test]
fn test_cache_batch_operations() {
    let normal = Normal::new(0.0_f64, 1.0_f64);
    let cache = NormalCache::from_distribution(&normal);

    let points = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    // Batch computation using cache
    let batch_results = cache.log_density_batch(&points);

    // Individual computations for comparison
    let individual_results: Vec<f64> = points.iter().map(|&x| cache.log_density(&x)).collect();

    // Verify batch and individual results match
    for (batch, individual) in batch_results.iter().zip(individual_results.iter()) {
        assert!((batch - individual).abs() < 1e-10);
    }

    // Also verify against direct distribution computation
    for (i, &x) in points.iter().enumerate() {
        let expected = normal.log_density_wrt_root(&x);
        assert!((batch_results[i] - expected).abs() < 1e-10);
    }
}

#[test]
fn test_poisson_batch_operations() {
    let poisson = Poisson::new(2.0_f64);
    let cache = PoissonCache::from_distribution(&poisson);

    let points = vec![0u64, 1, 2, 3, 4, 5];

    // Batch computation using cache
    let batch_results = cache.log_density_batch(&points);

    // Individual computations for comparison
    let individual_results: Vec<f64> = points.iter().map(|&k| cache.log_density(&k)).collect();

    // Verify batch and individual results match
    for (batch, individual) in batch_results.iter().zip(individual_results.iter()) {
        assert!((batch - individual).abs() < 1e-10);
    }

    // Also verify against direct distribution computation
    for (i, &k) in points.iter().enumerate() {
        let expected = poisson.log_density_wrt_root(&k);
        assert!((batch_results[i] - expected).abs() < 1e-10);
    }
}

#[test]
fn test_cache_closure_api() {
    let normal = Normal::new(1.0_f64, 0.5_f64);
    let cache = NormalCache::from_distribution(&normal);

    // Create closure for functional programming
    let log_density_fn = cache.log_density_fn();

    let points = [0.0, 0.5, 1.0, 1.5, 2.0];

    // Use closure with iterator
    let results: Vec<f64> = points.iter().map(log_density_fn).collect();

    // Verify results
    for (i, &x) in points.iter().enumerate() {
        let expected = normal.log_density_wrt_root(&x);
        assert!((results[i] - expected).abs() < 1e-10);
    }
}

#[test]
fn test_natural_params_access() {
    let normal = Normal::new(3.0_f64, 2.0_f64);
    let cache = NormalCache::from_distribution(&normal);

    // Access cached natural parameters
    let natural_params = cache.natural_params();

    // Verify they match the distribution's computation
    let expected_natural_params = normal.to_natural();

    assert!((natural_params[0] - expected_natural_params[0]).abs() < 1e-10);
    assert!((natural_params[1] - expected_natural_params[1]).abs() < 1e-10);
}

#[test]
fn test_poisson_natural_params_access() {
    let poisson = Poisson::new(2.5_f64);
    let cache = PoissonCache::from_distribution(&poisson);

    // Access cached natural parameters
    let natural_params = cache.natural_params();

    // Verify they match the distribution's computation
    let expected_natural_params = poisson.to_natural();

    assert!((natural_params[0] - expected_natural_params[0]).abs() < 1e-10);
}
