//! Tests for exponential family functionality
//!
//! This module tests that exponential family distributions can actually perform
//! proper computations using their natural parameter representation.

use measures::Normal;
use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::ExponentialFamily;
use num_traits::Float;

#[test]
fn test_normal_exponential_family_conversion() {
    let normal = Normal::new(2.0_f64, 1.5_f64);

    // Test natural parameter conversion
    let natural_params = normal.to_natural();
    let reconstructed = Normal::from_natural(natural_params);

    // Should reconstruct the same distribution (within floating point tolerance)
    assert!((reconstructed.mean - normal.mean).abs() < 1e-10);
    assert!((reconstructed.std_dev - normal.std_dev).abs() < 1e-10);
}

#[test]
fn test_normal_sufficient_statistics() {
    let normal = Normal::new(1.0_f64, 2.0_f64);
    let x = 3.0_f64;

    let sufficient_stats = normal.sufficient_statistic(&x);

    // For normal distribution, sufficient statistics are [x, x²]
    assert_eq!(sufficient_stats[0], x);
    assert_eq!(sufficient_stats[1], x * x);
}

#[test]
fn test_normal_log_partition() {
    let normal = Normal::new(2.0_f64, 1.5_f64);

    let log_partition = normal.log_partition();

    // For Normal(μ, σ²), A(η) = μ²/(2σ²) + log(σ√(2π))
    let sigma2 = normal.std_dev * normal.std_dev;
    let expected = (normal.mean * normal.mean) / (2.0 * sigma2)
        + 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln();

    assert!((log_partition - expected).abs() < 1e-10);
}

#[test]
fn test_normal_exponential_family_log_density() {
    let normal = Normal::new(1.0_f64, 2.0_f64);
    let x = 0.5_f64;

    // Compute log-density using exponential family formula
    let natural_params = normal.to_natural();
    let sufficient_stats = normal.sufficient_statistic(&x);
    let log_partition = normal.log_partition();

    // η·T(x) - A(η)
    let dot_product =
        natural_params[0] * sufficient_stats[0] + natural_params[1] * sufficient_stats[1];
    let exp_fam_log_density = dot_product - log_partition;

    // Compare with direct computation
    let direct_log_density = normal.log_density_wrt_root(&x);

    assert!(
        (exp_fam_log_density - direct_log_density).abs() < 1e-10,
        "Exponential family computation {exp_fam_log_density} != direct computation {direct_log_density}"
    );
}

#[test]
fn test_poisson_exponential_family_conversion() {
    let poisson = Poisson::new(3.5_f64);

    // Test natural parameter conversion
    let natural_param = poisson.to_natural();
    let reconstructed = Poisson::from_natural(natural_param);

    // Should reconstruct the same distribution
    assert!((reconstructed.lambda - poisson.lambda).abs() < 1e-10);
}

#[test]
fn test_poisson_sufficient_statistics() {
    let poisson = Poisson::new(2.0_f64);
    let k = 5u64;

    let sufficient_stat = poisson.sufficient_statistic(&k);

    // For Poisson distribution, sufficient statistic is just k
    assert_eq!(sufficient_stat, k);
}

#[test]
fn test_poisson_log_partition() {
    let poisson = Poisson::new(3.5_f64);

    let log_partition = poisson.log_partition();

    // For Poisson(λ), A(η) = e^η = λ
    assert!((log_partition - poisson.lambda).abs() < 1e-10);
}

#[test]
fn test_poisson_exponential_family_log_density() {
    let poisson = Poisson::new(2.5_f64);
    let k = 3u64;

    // Compute log-density using exponential family formula
    let natural_param = poisson.to_natural();
    let sufficient_stat = poisson.sufficient_statistic(&k);
    let log_partition = poisson.log_partition();

    // η·T(x) - A(η) - log(k!)
    let k_f64 = k as f64;
    let mut log_factorial = 0.0_f64;
    for i in 1..=k {
        log_factorial += (i as f64).ln();
    }

    let exp_fam_log_density =
        natural_param * (sufficient_stat as f64) - log_partition - log_factorial;

    // Compare with direct computation
    let direct_log_density = poisson.log_density_wrt_root(&k);

    assert!(
        (exp_fam_log_density - direct_log_density).abs() < 1e-10,
        "Exponential family computation {exp_fam_log_density} != direct computation {direct_log_density}"
    );
}

#[test]
fn test_normal_log_density_at_different_points() {
    let normal = Normal::new(0.0_f64, 1.0_f64);

    let points = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    for &x in &points {
        let ld = normal.log_density();
        let computed_value: f64 = ld.at(&x);

        // The computation should not panic and should return a finite value
        assert!(
            computed_value.is_finite(),
            "Log-density at {x} is not finite: {computed_value}"
        );

        // For standard normal, we can check some known values
        if x == 0.0 {
            let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
            assert!((computed_value - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_poisson_log_density_at_different_points() {
    let poisson = Poisson::new(2.0_f64);

    let points = vec![0u64, 1, 2, 3, 5, 10];

    for &k in &points {
        let ld = poisson.log_density();
        let computed_value: f64 = ld.at(&k);

        // The computation should not panic and should return a finite value
        assert!(
            computed_value.is_finite(),
            "Log-density at {k} is not finite: {computed_value}"
        );

        // Check that it's negative (since it's a log probability)
        assert!(
            computed_value <= 0.0,
            "Log-density should be non-positive, got {computed_value}"
        );
    }
}

#[test]
fn test_different_numeric_types() {
    // Test f32
    let normal_f32 = Normal::new(0.0_f32, 1.0_f32);
    let x_f32 = 1.0_f32;
    let ld_f32 = normal_f32.log_density();
    let result_f32: f32 = ld_f32.at(&x_f32);
    assert!(result_f32.is_finite());

    // Test f64
    let normal_f64 = Normal::new(0.0_f64, 1.0_f64);
    let x_f64 = 1.0_f64;
    let ld_f64 = normal_f64.log_density();
    let result_f64: f64 = ld_f64.at(&x_f64);
    assert!(result_f64.is_finite());

    // Results should be close (within f32 precision)
    assert!((f64::from(result_f32) - result_f64).abs() < 1e-6);
}

/// Test that measures with the same root can use automatic computation
#[test]
fn test_automatic_shared_root_computation() {
    let normal1 = Normal::new(0.0_f64, 1.0_f64);
    let normal2 = Normal::new(1.0_f64, 1.5_f64);
    let x = 0.5_f64;

    // This should use the automatic computation: normal1.log_density() - normal2.log_density()
    let ld_wrt = normal1.log_density().wrt(normal2.clone());
    let relative_density: f64 = ld_wrt.at(&x);

    // Manually compute the same thing
    let ld1: f64 = normal1.log_density().at(&x);
    let ld2: f64 = normal2.log_density().at(&x);
    let manual_relative = ld1 - ld2;

    assert!(
        (relative_density - manual_relative).abs() < 1e-10,
        "Automatic computation {relative_density} != manual computation {manual_relative}"
    );
}
