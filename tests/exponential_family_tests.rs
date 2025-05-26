//! Tests for exponential family functionality
//!
//! This module tests that exponential family distributions can actually perform
//! proper computations using their natural parameter representation.

use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};

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
    assert!((reconstructed.rate - poisson.rate).abs() < 1e-10);
}

#[test]
fn test_poisson_sufficient_statistics() {
    let poisson = Poisson::new(2.0_f64);
    let k = 5u64;

    let sufficient_stat = poisson.sufficient_statistic(&k);

    // For Poisson distribution, sufficient statistic is [k] as [F; 1]
    assert_eq!(sufficient_stat[0], k as f64);
}

#[test]
fn test_poisson_log_partition() {
    let poisson = Poisson::new(3.5_f64);

    let log_partition = poisson.log_partition();

    // For Poisson(λ), A(η) = e^η = λ
    assert!((log_partition - poisson.rate).abs() < 1e-10);
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
    let mut log_factorial = 0.0_f64;
    for i in 1..=k {
        log_factorial += (i as f64).ln();
    }

    // Using DotProduct for [F; 1] arrays
    use measures::traits::DotProduct;
    let exp_fam_log_density = natural_param.dot(&sufficient_stat) - log_partition - log_factorial;

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

#[test]
fn test_automatic_chain_rule_for_poisson() {
    println!("\n=== Testing Automatic Chain Rule Enhancement ===");

    let poisson = Poisson::new(2.5_f64);
    let k = 3u64;

    // The default exp_fam_log_density should now automatically include:
    // 1. Exponential family part: η·T(x) - A(η) = k·ln(λ) - λ
    // 2. Chain rule part: base_measure.log_density_wrt_root(x) = -log(k!)
    let automatic_result = poisson.exp_fam_log_density(&k);

    // Manual computation for comparison
    let manual_exp_fam = (k as f64) * poisson.rate.ln() - poisson.rate;
    let manual_factorial = if k == 0 {
        0.0
    } else {
        -(1..=k).map(|i| (i as f64).ln()).sum::<f64>()
    };
    let manual_result = manual_exp_fam + manual_factorial;

    println!("Automatic chain rule result: {automatic_result}");
    println!(
        "Manual computation: exp_fam({manual_exp_fam}) + factorial({manual_factorial}) = {manual_result}"
    );
    println!("Difference: {}", (automatic_result - manual_result).abs());

    assert!(
        (automatic_result - manual_result).abs() < 1e-10,
        "Automatic chain rule failed: auto={automatic_result}, manual={manual_result}"
    );

    // Also verify it matches the HasLogDensity implementation
    let log_density_result: f64 = poisson.log_density().at(&k);
    assert!(
        (automatic_result - log_density_result).abs() < 1e-10,
        "exp_fam_log_density doesn't match log_density: {automatic_result} vs {log_density_result}"
    );

    println!("✅ Automatic chain rule works perfectly!");
    println!("✅ No manual override needed for Poisson!");
}

#[test]
fn test_automatic_chain_rule_for_normal() {
    println!("\n=== Testing Normal with Automatic Chain Rule ===");

    let normal = Normal::new(1.0_f64, 2.0_f64);
    let x = 0.5_f64;

    // For Normal, base_measure == root_measure (both LebesgueMeasure)
    // So chain rule part should be zero: base_measure.log_density_wrt_root(x) = 0
    let automatic_result = normal.exp_fam_log_density(&x);

    // Manual computation: just the exponential family part (no chain rule needed)
    let natural_params = normal.to_natural();
    let sufficient_stats = normal.sufficient_statistic(&x);
    let log_partition = normal.log_partition();
    let manual_result = natural_params[0] * sufficient_stats[0]
        + natural_params[1] * sufficient_stats[1]
        - log_partition;

    println!("Normal automatic result: {automatic_result}");
    println!("Normal manual result: {manual_result}");
    println!("Difference: {}", (automatic_result - manual_result).abs());

    assert!(
        (automatic_result - manual_result).abs() < 1e-10,
        "Normal automatic chain rule failed: auto={automatic_result}, manual={manual_result}"
    );

    // Verify base measure log-density is indeed zero
    let base_measure = normal.base_measure();
    let chain_rule_part: f64 = base_measure.log_density_wrt_root(&x);
    println!("Normal chain rule part: {chain_rule_part} (should be 0)");

    assert!(
        chain_rule_part.abs() < 1e-10,
        "Normal chain rule part should be zero, got: {chain_rule_part}"
    );

    println!("✅ Normal distributions work correctly with automatic chain rule!");
    println!("✅ Chain rule part is zero as expected for base_measure == root_measure!");
}

#[test]
fn test_exponential_family_relative_density_optimization() {
    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(2.0, 1.5);
    let x = 1.0;

    // Standard computation
    let standard_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    {
        use measures::exponential_family::jit::ZeroOverheadOptimizer;
        let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
        let optimized_result: f64 = optimized_fn(&x);

        // Results should be identical
        assert!(
            (standard_result - optimized_result).abs() < 1e-10,
            "Standard: {:.15}, Optimized: {:.15}, Diff: {:.2e}",
            standard_result,
            optimized_result,
            (standard_result - optimized_result).abs()
        );
    }

    // Without JIT feature, just test standard computation
    #[cfg(not(feature = "jit"))]
    {
        // Just verify the standard computation works
        assert!(standard_result.is_finite());
    }
}
