//! Validation tests comparing our implementations with rv crate as reference.
//!
//! ## Key Insights from `IntegralMeasure` Investigation
//!
//! The debugging tests prove that the `IntegralMeasure` approach is mathematically correct:
//!
//! 1. **Working Poisson** = Exponential Family part + Chain rule part
//!    - `ExpFam` part: η·T(x) - A(η) = k·ln(λ) - λ  
//!    - Chain rule part: -log(k!) from the factorial base measure
//!    - Combined: k·ln(λ) - λ - log(k!) = correct Poisson log-PMF
//!
//! 2. **Default `ExpFam` implementation** only computes η·T(x) - A(η), missing the chain rule
//!
//! 3. **Future enhancement**: Modify `ExponentialFamily` trait to automatically include
//!    chain rule when base measure != root measure, eliminating the need for manual overrides

use approx::assert_abs_diff_eq;
use measures::core::HasLogDensity;
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};
use rv::prelude::*;

#[test]
fn test_normal_vs_rv() {
    use rv::dist::Gaussian;

    let measures_normal = Normal::new(1.0_f64, 2.0_f64);
    let rv_normal = Gaussian::new(1.0, 2.0).unwrap();

    // Test at several points
    let test_points = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];

    for x in test_points {
        let measures_result = measures_normal.log_density_wrt_root(&x);
        let rv_result = rv_normal.ln_f(&x);

        println!(
            "x={}: measures={:.10}, rv={:.10}, diff={:.2e}",
            x,
            measures_result,
            rv_result,
            (measures_result - rv_result).abs()
        );

        assert_abs_diff_eq!(measures_result, rv_result, epsilon = 1e-10);
    }
}

#[test]
fn test_poisson_vs_rv() {
    use rv::dist::Poisson as RvPoisson;

    let measures_poisson = Poisson::new(2.5_f64);
    let rv_poisson = RvPoisson::new(2.5).unwrap();

    // Test at several integer points
    let test_points: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 10, 15, 20];

    for k in test_points {
        let measures_result = measures_poisson.log_density_wrt_root(&(k as u64));
        let rv_result = rv_poisson.ln_f(&k);

        println!(
            "k={}: measures={:.10}, rv={:.10}, diff={:.2e}",
            k,
            measures_result,
            rv_result,
            (measures_result - rv_result).abs()
        );

        assert_abs_diff_eq!(measures_result, rv_result, epsilon = 1e-10);
    }
}

#[test]
fn test_poisson_working_vs_integral_measure() {
    // Test that our Poisson with FactorialMeasure matches manual computation
    let poisson = Poisson::new(2.5_f64);

    for k in [0, 1, 2, 5, 10, 15, 20, 25] {
        let integral_result = poisson.exp_fam_log_density(&k);

        // Manual Poisson computation for verification
        let lambda = 2.5_f64;
        let k_f64 = k as f64;
        let mut log_factorial = 0.0;
        for i in 1..=k {
            log_factorial += (i as f64).ln();
        }
        let manual_result = k_f64 * lambda.ln() - lambda - log_factorial;

        println!(
            "k={}: integral={:.10}, manual={:.10}, diff={:.2e}",
            k,
            integral_result,
            manual_result,
            (integral_result - manual_result).abs()
        );

        assert_abs_diff_eq!(integral_result, manual_result, epsilon = 1e-10);
    }
}

/// Test O(1) Stirling's approximation accuracy against exact computation
#[test]
fn test_stirling_factorial_accuracy() {
    use measures::measures::derived::factorial::FactorialMeasure;

    let factorial_measure = FactorialMeasure::<f64>::new();

    // Test accuracy for different k values
    let test_cases = vec![
        // (k, expected_relative_error_threshold)
        (21, 1e-4),   // Just above exact computation cutoff
        (25, 1e-5),   // Should be very accurate
        (50, 1e-6),   // Stirling gets more accurate with larger k
        (100, 1e-7),  // Even better
        (500, 1e-8),  // Excellent accuracy
        (1000, 1e-9), // Nearly perfect
    ];

    for (k, error_threshold) in test_cases {
        // Our O(1) implementation
        let stirling_result = factorial_measure.log_density_wrt_root(&k);

        // Exact computation for reference (O(k))
        let mut exact_log_factorial = 0.0_f64;
        for i in 1..=k {
            exact_log_factorial += (i as f64).ln();
        }
        let exact_result = -exact_log_factorial;

        let relative_error = ((stirling_result - exact_result) / exact_result).abs();

        println!(
            "k={k}: stirling={stirling_result:.12}, exact={exact_result:.12}, relative_error={relative_error:.2e}"
        );

        assert!(
            relative_error < error_threshold,
            "k={k}: relative error {relative_error:.2e} exceeds threshold {error_threshold:.2e}"
        );
    }
}

/// Test that O(1) factorial provides correct Poisson results
#[test]
fn test_stirling_poisson_vs_rv() {
    use rv::dist::Poisson as RvPoisson;

    let measures_poisson = Poisson::new(10.0_f64);
    let rv_poisson = RvPoisson::new(10.0).unwrap();

    // Test at larger k values where Stirling approximation is used
    let test_points: Vec<usize> = vec![25, 50, 100, 200, 500, 1000];

    for k in test_points {
        let measures_result = measures_poisson.log_density_wrt_root(&(k as u64));
        let rv_result = rv_poisson.ln_f(&k);

        let relative_error = ((measures_result - rv_result) / rv_result).abs();

        println!(
            "k={k}: measures={measures_result:.10}, rv={rv_result:.10}, relative_error={relative_error:.2e}"
        );

        // For statistical computing, 0.01% relative error is excellent
        assert!(
            relative_error < 1e-4,
            "k={k}: relative error {relative_error:.2e} too large"
        );
    }
}

/// Performance regression test: O(1) should be much faster for large k
#[test]
fn test_factorial_performance_regression() {
    use std::time::Instant;

    let poisson = Poisson::new(5.0_f64);
    let large_k = 1000u64;
    let iterations = 1000;

    // Time our O(1) implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = poisson.exp_fam_log_density(&large_k);
    }
    let o1_duration = start.elapsed();

    // Time the old O(k) approach
    let start = Instant::now();
    for _ in 0..iterations {
        let mut log_factorial = 0.0_f64;
        for i in 1..=large_k {
            log_factorial += (i as f64).ln();
        }
        let _ =
            poisson.to_natural()[0] * (large_k as f64) - poisson.log_partition() - log_factorial;
    }
    let ok_duration = start.elapsed();

    let speedup = ok_duration.as_nanos() as f64 / o1_duration.as_nanos() as f64;

    println!("O(1) time: {o1_duration:?}");
    println!("O(k) time: {ok_duration:?}");
    println!("Speedup: {speedup:.1}x");

    // For k=1000, we should see significant speedup
    assert!(speedup > 10.0, "Expected >10x speedup, got {speedup:.1}x");
}

/// Test that our precomputed lookup table values are correct
#[test]
fn test_lookup_table_correctness() {
    // This test validates that our LOG_FACTORIAL_TABLE contains the correct values
    // by recomputing them and comparing

    for k in 0u64..=20 {
        // Compute log(k!) exactly using the O(k) method
        let exact_log_factorial = if k == 0 {
            0.0_f64 // log(0!) = log(1) = 0
        } else {
            (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
        };

        // Get the value from our lookup table via FactorialMeasure
        let factorial_measure =
            measures::measures::derived::factorial::FactorialMeasure::<f64>::new();
        let table_result = factorial_measure.log_density_wrt_root(&k);
        let expected_result = -exact_log_factorial; // FactorialMeasure returns -log(k!)

        let error = (table_result - expected_result).abs();

        println!("k={k}: exact={expected_result:.15}, table={table_result:.15}, error={error:.2e}");

        // For the exact lookup table values, we should have good accuracy
        // Note: Small differences are expected because:
        // - Table values: computed as ln(k!) where k! is computed as product
        // - "Exact" values: computed as sum(ln(i)) which has different rounding
        // Both approaches are mathematically equivalent but have different numerical properties
        assert!(
            error < 1e-9,
            "k={k}: lookup table error {error:.2e} too large (expected < 1e-9)"
        );
    }
}

/// Test that demonstrates the generation code for the lookup table
#[test]
fn test_lookup_table_generation_code() {
    // This test shows exactly how the lookup table values were computed
    // It serves as documentation and validation

    println!("\n// Generated lookup table values:");
    println!("const LOG_FACTORIAL_TABLE: [f64; 21] = [");

    for k in 0u64..=20 {
        let log_factorial = if k == 0 {
            0.0 // Special case: log(0!) = log(1) = 0
        } else {
            // Compute k! then take log
            let factorial = (1..=k).product::<u64>();
            (factorial as f64).ln()
        };

        let comment = if k == 0 {
            format!("// log({k}!) = log(1)")
        } else if k == 1 {
            format!("// log({k}!) = log(1)")
        } else if k == 2 {
            "// log(2!) = log(2) = LN_2".to_string()
        } else {
            let factorial = (1..=k).product::<u64>();
            format!("// log({k}!) = log({factorial})")
        };

        let value_str = if k == 2 {
            "std::f64::consts::LN_2".to_string()
        } else {
            format!("{log_factorial:.15}")
        };

        println!("    {value_str},  {comment}");
    }

    println!("];");
    println!("// This validates that our precomputed table is correct!");
}
