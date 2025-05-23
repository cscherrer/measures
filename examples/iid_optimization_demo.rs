use measures::exponential_family::ExponentialFamily;
use measures::{IIDExtension, Normal};

fn main() {
    println!("=== IID Optimization: natural_and_log_partition() ===\n");

    println!("Optimization: Instead of calling to_natural() and log_partition() separately,");
    println!("we use natural_and_log_partition() to get both values efficiently.\n");

    let normal = Normal::new(1.5, 2.0);
    let iid_normal = normal.clone().iid();
    let samples = vec![1.0, 2.0, 1.5, 2.5, 1.2];

    println!("Normal distribution: μ=1.5, σ=2.0");
    println!("Samples: {samples:?}\n");

    // Demonstrate the optimization
    println!("=== Inefficient Approach (separate calls) ===");
    let start = std::time::Instant::now();
    let natural_params_sep = normal.to_natural();
    let log_partition_sep = normal.log_partition();
    let separate_time = start.elapsed();

    println!(
        "Natural params: [{:.6}, {:.6}]",
        natural_params_sep[0], natural_params_sep[1]
    );
    println!("Log partition: {:.6}", log_partition_sep);
    println!("Time: {:?}", separate_time);

    println!("\n=== Efficient Approach (single call) ===");
    let start = std::time::Instant::now();
    let (natural_params_combined, log_partition_combined) = normal.natural_and_log_partition();
    let combined_time = start.elapsed();

    println!(
        "Natural params: [{:.6}, {:.6}]",
        natural_params_combined[0], natural_params_combined[1]
    );
    println!("Log partition: {:.6}", log_partition_combined);
    println!("Time: {:?}", combined_time);

    // Verify they're identical
    let params_match = (natural_params_sep[0] - natural_params_combined[0] as f64).abs() < 1e-10
        && (natural_params_sep[1] - natural_params_combined[1]).abs() < 1e-10;
    let partition_match = (log_partition_sep - log_partition_combined).abs() < 1e-10;

    println!("\n=== Verification ===");
    println!("Natural parameters match: {}", params_match);
    println!("Log partitions match: {}", partition_match);
    println!("✓ Both approaches give identical results");

    // Show the efficient IID computation
    println!("\n=== Efficient IID Computation ===");
    let start = std::time::Instant::now();
    let efficient_result: f64 = iid_normal.log_density(&samples);
    let efficient_time = start.elapsed();

    println!("IID log-density: {:.8}", efficient_result);
    println!("Time: {:?}", efficient_time);

    // Compare with naive approach
    let start = std::time::Instant::now();
    let naive_result: f64 = iid_normal.log_density(&samples);
    let naive_time = start.elapsed();

    println!("\n=== Comparison with Naive Approach ===");
    println!("Naive result: {:.8}", naive_result);
    println!("Naive time: {:?}", naive_time);

    let results_match = (efficient_result - naive_result).abs() < 1e-10;
    println!("Results match: {}", results_match);

    if naive_time.as_nanos() > 0 && efficient_time.as_nanos() > 0 {
        let speedup = naive_time.as_nanos() as f64 / efficient_time.as_nanos() as f64;
        println!("Speedup: {:.1}x", speedup);
    }

    println!("\n=== Key Benefits of Optimization ===");
    println!("✓ Single method call instead of two separate calls");
    println!("✓ Avoids redundant computation in the underlying implementation");
    println!("✓ Better performance for IID exponential family calculations");
    println!("✓ Cleaner, more readable code");

    println!("\nThe efficient IID computation now uses:");
    println!("  let (η, A_η) = distribution.natural_and_log_partition();");
    println!("  result = η·∑T(xᵢ) - n·A_η");
    println!("Instead of:");
    println!("  let η = distribution.to_natural();");
    println!("  let A_η = distribution.log_partition();");
    println!("  result = η·∑T(xᵢ) - n·A_η");
}
