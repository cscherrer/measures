use measures::exponential_family::ExponentialFamily;
use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== IID: Naive vs Efficient Exponential Family Computation ===\n");

    // Create a normal distribution and IID version
    let normal = Normal::new(0.5, 1.5);
    let iid_normal = normal.clone().iid();
    let samples = vec![0.2, 0.8, -0.1, 1.2, 0.6];

    println!("Normal distribution: μ=0.5, σ=1.5");
    println!("Samples: {samples:?}");
    println!("Sample size: n={}\n", samples.len());

    // Method 1: Naive approach (sum individual log-densities)
    println!("=== Method 1: Naive Summation ===");
    let naive_result: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();
    println!("Result: {naive_result:.10}");

    // Show the breakdown
    println!("Breakdown:");
    for (i, &x) in samples.iter().enumerate() {
        let log_density = normal.log_density().at(&x);
        println!("  log p(x{}) = log p({:.1}) = {:.6}", i + 1, x, log_density);
    }
    let manual_sum: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();
    println!("  Sum: {manual_sum:.10}");
    println!(
        "  ✓ Method matches: {}\n",
        (naive_result - manual_sum).abs() < 1e-10
    );

    // Method 2: Efficient exponential family approach
    println!("=== Method 2: Efficient Exponential Family ===");
    let efficient_result = iid_normal.log_density(&samples);
    println!("Result: {efficient_result:.10}");

    // Show the efficient computation breakdown
    println!("Breakdown:");

    // Get exponential family components
    let natural_params = normal.to_natural();
    let log_partition = normal.log_partition();
    println!(
        "  Natural parameters η = [{:.6}, {:.6}]",
        natural_params[0], natural_params[1]
    );
    println!("  Log partition A(η) = {log_partition:.6}");

    // Compute sufficient statistics efficiently
    let sum_x: f64 = samples.iter().sum();
    let sum_x2: f64 = samples.iter().map(|&x| x * x).sum();
    println!("  Sufficient statistics ∑T(xᵢ) = [{sum_x:.6}, {sum_x2:.6}]");

    // Exponential family computation
    let dot_product = natural_params[0] * sum_x + natural_params[1] * sum_x2;
    let n = samples.len() as f64;
    let scaled_log_partition = n * log_partition;

    println!("  Dot product η·∑T(xᵢ) = {dot_product:.6}");
    println!("  n·A(η) = {n:.1} × {log_partition:.6} = {scaled_log_partition:.6}");
    println!(
        "  Final: {:.6} - {:.6} = {:.10}",
        dot_product,
        scaled_log_partition,
        dot_product - scaled_log_partition
    );

    // Verify they match
    println!("\n=== Verification ===");
    println!("  Naive approach:     {naive_result:.10}");
    println!("  Efficient approach: {efficient_result:.10}");
    let difference = (naive_result - efficient_result).abs();
    println!("  Difference:         {difference:.2e}");
    println!("  ✓ Methods agree: {}", difference < 1e-10);

    println!("\n=== Efficiency Demonstration ===");
    let large_samples: Vec<f64> = (0..50000).map(|i| 0.5 + f64::from(i) * 0.0001).collect();
    println!("Testing with {} samples...", large_samples.len());

    // Naive approach timing
    let start = std::time::Instant::now();
    let naive_large: f64 = large_samples.iter().map(|&x| normal.log_density().at(&x)).sum();
    let naive_duration = start.elapsed();

    // Efficient approach timing  
    let start = std::time::Instant::now();
    let efficient_large = iid_normal.log_density(&large_samples);
    let efficient_duration = start.elapsed();

    println!("Naive time:     {naive_duration:?}");
    println!("Efficient time: {efficient_duration:?}");
    let speedup = naive_duration.as_nanos() as f64 / efficient_duration.as_nanos() as f64;
    println!("Speedup:        {speedup:.1}x");
    println!("Results match:  {}", (naive_large - efficient_large).abs() < 1e-8);

    println!("\n=== Summary ===");
    println!("✓ The efficient exponential family approach computes the same result");
    println!("✓ But uses the mathematical structure: η·∑ᵢT(xᵢ) - n·A(η)");
    println!("✓ Instead of summing n individual log-densities");
    println!("✓ This provides significant computational benefits for large samples");

    println!("\nKey insight: IID collections of exponential families");
    println!("are themselves exponential families with predictable structure!");
}
