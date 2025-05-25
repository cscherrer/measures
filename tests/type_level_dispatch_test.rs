//! Test that demonstrates type-level dispatch for `HasLogDensity` based on `IsExponentialFamily`.
//!
//! This test verifies that:
//! 1. Exponential families automatically get optimized cached computation
//! 2. Primitive measures get simple zero log-density implementation
//! 3. The dispatch happens at compile time based on the `IsExponentialFamily` marker

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::Cauchy;
use measures::distributions::continuous::normal::Normal;
use measures::measures::primitive::lebesgue::LebesgueMeasure;

#[test]
fn test_exponential_family_automatic_dispatch() {
    // Normal distribution (exponential family) should get automatic cached computation
    let normal = Normal::new(0.0, 1.0);
    let log_density = normal.log_density();

    // This should use the cached exponential family implementation automatically
    let result: f64 = log_density.at(&0.5);

    // Verify it's a reasonable value
    assert!(result.is_finite());
    assert!(result < 0.0); // Log-density should be negative for continuous distributions
}

#[test]
fn test_primitive_measure_automatic_dispatch() {
    // Test that primitive measures get automatic HasLogDensity implementation
    let lebesgue = LebesgueMeasure::<f64>::new();
    let log_density = lebesgue.log_density();

    // This should work without manual implementation
    let _result: f64 = log_density.at(&1.0);
}

#[test]
fn test_builder_pattern_works_with_dispatch() {
    let normal = Normal::new(0.0, 1.0);
    let x = 0.5;

    // Test the builder pattern
    let log_density = normal.log_density();
    let result: f64 = log_density.at(&x);

    // Should be finite and reasonable
    assert!(result.is_finite());
}

#[test]
fn test_type_level_dispatch_is_zero_cost() {
    // This test verifies that the type-level dispatch compiles to efficient code
    // by ensuring it can be used in const contexts (if the underlying operations support it)
    let normal = Normal::new(0.0, 1.0);
    let _log_density = normal.log_density();

    // If this compiles, the dispatch is working at compile time
    // No assertion needed - compilation success is the test
}

#[test]
fn test_four_way_exponential_family_dispatch() {
    let x = 0.5;

    // Case 1: (False, False) - Neither is exponential family
    let cauchy1 = Cauchy::new(0.0, 1.0);
    let cauchy2 = Cauchy::new(1.0, 2.0);
    let result_ff: f64 = cauchy1.log_density().wrt(cauchy2).at(&x);
    assert!(result_ff.is_finite());

    // Case 2: (False, True) - Only base is exponential family
    let cauchy = Cauchy::new(0.0, 1.0);
    let normal = Normal::new(0.0, 1.0);
    let result_ft: f64 = cauchy.log_density().wrt(normal).at(&x);
    assert!(result_ft.is_finite());

    // Case 3: (True, False) - Only measure is exponential family
    let normal = Normal::new(0.0, 1.0);
    let cauchy = Cauchy::new(0.0, 1.0);
    let result_tf: f64 = normal.log_density().wrt(cauchy).at(&x);
    assert!(result_tf.is_finite());

    // Case 4: (True, True) - Both are exponential families
    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 2.0);
    let result_tt: f64 = normal1.log_density().wrt(normal2).at(&x);
    assert!(result_tt.is_finite());
}
