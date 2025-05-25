//! IID Optimization Demo
//!
//! Demonstrates the `natural_and_log_partition()` optimization for exponential family
//! distributions, which computes both values efficiently in a single call.

use measures::exponential_family::ExponentialFamily;
use measures::{IIDExtension, Normal};

fn main() {
    let normal = Normal::new(1.5, 2.0);
    let iid_normal = normal.clone().iid();
    let samples = vec![1.0, 2.0, 1.5, 2.5, 1.2];

    // Compare separate vs combined calls
    let start = std::time::Instant::now();
    let natural_params_sep = normal.to_natural();
    let log_partition_sep: f64 = normal.log_partition();
    let separate_time = start.elapsed();

    let start = std::time::Instant::now();
    let (natural_params_combined, log_partition_combined): ([f64; 2], f64) =
        normal.natural_and_log_partition();
    let combined_time = start.elapsed();

    println!(
        "Separate calls: natural=[{:.6}, {:.6}], log_partition={:.6}, time={:?}",
        natural_params_sep[0], natural_params_sep[1], log_partition_sep, separate_time
    );

    println!(
        "Combined call:  natural=[{:.6}, {:.6}], log_partition={:.6}, time={:?}",
        natural_params_combined[0],
        natural_params_combined[1],
        log_partition_combined,
        combined_time
    );

    // Verify identical results
    let params_match = (natural_params_sep[0] - natural_params_combined[0]).abs() < 1e-10_f64
        && (natural_params_sep[1] - natural_params_combined[1]).abs() < 1e-10_f64;
    let partition_match = (log_partition_sep - log_partition_combined).abs() < 1e-10_f64;

    assert!(
        params_match && partition_match,
        "Results should be identical"
    );

    // Show IID computation using the optimization
    let start = std::time::Instant::now();
    let iid_result: f64 = iid_normal.iid_log_density(&samples);
    let iid_time = start.elapsed();

    println!("IID log-density: {iid_result:.8}, time: {iid_time:?}");
    println!("✓ Optimization verified");
}
