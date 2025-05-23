//! Tests for the `ExponentialFamilyCache` trait with generic cache.

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::normal::Normal;
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::{ExponentialFamily, ExponentialFamilyCache, GenericExpFamCache};

#[test]
fn test_normal_cache_trait_api() {
    let normal = Normal::new(1.0, 2.0);
    let cache: GenericExpFamCache<Normal<f64>, f64, f64> =
        GenericExpFamCache::from_distribution(&normal);

    // Test basic cache functionality
    let log_density_value: f64 = cache.log_density(&0.5);
    assert!(log_density_value.is_finite());

    // Compare with direct computation
    let direct_log_density: f64 = normal.log_density().at(&0.5);
    let diff: f64 = log_density_value - direct_log_density;
    assert!(
        diff.abs() < 1e-10,
        "Cached computation {log_density_value} != direct computation {direct_log_density}"
    );
}

#[test]
fn test_cached_log_partition() {
    let normal = Normal::new(2.0, 1.5);
    let cache = GenericExpFamCache::from_distribution(&normal);

    let cached_log_partition: f64 = cache.log_partition();
    let direct_log_partition: f64 = normal.log_partition();
    let diff: f64 = cached_log_partition - direct_log_partition;

    assert!(
        diff.abs() < 1e-10,
        "Cached log partition {cached_log_partition} != direct log partition {direct_log_partition}"
    );
}

#[test]
fn test_poisson_cache_trait_api() {
    let poisson = Poisson::new(2.5);
    let cache: GenericExpFamCache<Poisson<f64>, u64, f64> =
        GenericExpFamCache::from_distribution(&poisson);

    // Test basic cache functionality
    let log_density_value: f64 = cache.log_density(&3u64);
    assert!(log_density_value.is_finite());

    // Compare with direct computation
    let direct_log_density: f64 = poisson.log_density().at(&3u64);
    let diff: f64 = log_density_value - direct_log_density;
    assert!(
        diff.abs() < 1e-10,
        "Cached computation {log_density_value} != direct computation {direct_log_density}"
    );
}
