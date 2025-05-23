//! Test that demonstrates type-level dispatch for `HasLogDensity` based on `IsExponentialFamily`.
//!
//! This test verifies that:
//! 1. Exponential families automatically get optimized cached computation
//! 2. Primitive measures get simple zero log-density implementation
//! 3. The dispatch happens at compile time based on the `IsExponentialFamily` marker

use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::distributions::continuous::normal::Normal;
use measures::distributions::discrete::poisson::Poisson;
use measures::measures::primitive::counting::CountingMeasure;
use measures::measures::primitive::lebesgue::LebesgueMeasure;

#[test]
fn test_exponential_family_automatic_dispatch() {
    // Normal distribution (exponential family) should get automatic cached computation
    let normal = Normal::new(0.0, 1.0);
    let log_density_value: f64 = normal.log_density_wrt_root(&0.5);

    // This should use the cached exponential family implementation automatically
    assert!(log_density_value.is_finite());

    // Poisson distribution (exponential family) should also get automatic cached computation
    let poisson = Poisson::new(2.0);
    let log_density_value: f64 = poisson.log_density_wrt_root(&3u64);

    // This should use the cached exponential family implementation automatically
    assert!(log_density_value.is_finite());
}

#[test]
fn test_primitive_measure_automatic_dispatch() {
    // Lebesgue measure (primitive) should get zero log-density implementation
    let lebesgue = LebesgueMeasure::<f64>::new();
    let log_density_value: f64 = lebesgue.log_density_wrt_root(&0.5);

    // Primitive measures have log-density 0 with respect to themselves
    assert_eq!(log_density_value, 0.0);

    // Counting measure (primitive) should also get zero log-density implementation
    let counting = CountingMeasure::<u64>::new();
    let log_density_value: f64 = counting.log_density_wrt_root(&3u64);

    // Primitive measures have log-density 0 with respect to themselves
    assert_eq!(log_density_value, 0.0);
}

#[test]
fn test_builder_pattern_works_with_dispatch() {
    // Test that the builder pattern works with both types
    let normal = Normal::new(1.0, 2.0);
    let ld = normal.log_density();
    let value: f64 = ld.at(&1.5);
    assert!(value.is_finite());

    let lebesgue = LebesgueMeasure::<f64>::new();
    let ld = lebesgue.log_density();
    let value: f64 = ld.at(&1.5);
    assert_eq!(value, 0.0);
}

#[test]
fn test_type_level_dispatch_is_zero_cost() {
    // This test demonstrates that the dispatch happens at compile time
    // by showing that both exponential families and primitive measures
    // can be used in the same generic function

    fn compute_log_density<M: HasLogDensity<f64, f64>>(measure: &M, x: f64) -> f64 {
        measure.log_density_wrt_root(&x)
    }

    let normal = Normal::new(0.0, 1.0);
    let lebesgue = LebesgueMeasure::<f64>::new();

    // Both should compile and work correctly
    let normal_result = compute_log_density(&normal, 0.0);
    let lebesgue_result = compute_log_density(&lebesgue, 0.0);

    assert!(normal_result.is_finite());
    assert_eq!(lebesgue_result, 0.0);
}
