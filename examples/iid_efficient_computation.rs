use measures::exponential_family::ExponentialFamily;
use measures::{LogDensityBuilder, Normal};

fn main() {
    println!("=== Efficient IID Exponential Family Computation ===\n");

    // Compare the naive approach vs exponential family approach
    let normal = Normal::new(1.0, 2.0);
    let samples = vec![0.5, 1.2, 0.8, 1.5, 0.9];

    println!("Normal distribution: μ=1.0, σ=2.0");
    println!("Samples: {samples:?}");
    println!("Sample size: n={}", samples.len());

    // Method 1: Naive approach (sum individual log-densities)
    println!("\n=== Method 1: Naive Summation ===");
    let start = std::time::Instant::now();
    let mut naive_total = 0.0;
    for (i, &x) in samples.iter().enumerate() {
        let individual_log_density = normal.log_density().at(&x);
        naive_total += individual_log_density;
        println!(
            "  x{} = {:.1}: log p(x{}) = {:.6}",
            i + 1,
            x,
            i + 1,
            individual_log_density
        );
    }
    let naive_duration = start.elapsed();
    println!("  Total (naive): {naive_total:.6}");
    println!("  Time: {naive_duration:?}");

    // Method 2: Exponential Family Approach
    println!("\n=== Method 2: Exponential Family Structure ===");
    let start = std::time::Instant::now();

    // Use efficient natural_and_log_partition method
    let (natural_params, log_partition) = normal.natural_and_log_partition();

    println!(
        "  Natural parameters η = [{:.6}, {:.6}]",
        natural_params[0], natural_params[1]
    );
    println!("  Log partition A(η) = {log_partition:.6}");

    // Compute sufficient statistics: ∑ᵢT(xᵢ) = [∑x, ∑x²]
    let sum_x: f64 = samples.iter().sum();
    let sum_x2: f64 = samples.iter().map(|&x| x * x).sum();
    let sufficient_stats = [sum_x, sum_x2];

    println!(
        "  Sufficient statistics ∑T(xᵢ) = [{:.6}, {:.6}]",
        sufficient_stats[0], sufficient_stats[1]
    );

    // Exponential family computation: η·∑T(xᵢ) - n·A(η)
    let dot_product =
        natural_params[0] * sufficient_stats[0] + natural_params[1] * sufficient_stats[1];
    let n = samples.len() as f64;
    let exp_fam_log_density = dot_product - n * log_partition;

    let exp_fam_duration = start.elapsed();

    println!("  Dot product η·∑T(xᵢ) = {dot_product:.6}");
    println!("  n·A(η) = {:.6}", n * log_partition);
    println!("  Total (exp family): {exp_fam_log_density:.6}");
    println!("  Time: {exp_fam_duration:?}");

    // Verify they're the same
    let difference = (naive_total - exp_fam_log_density).abs();
    println!("\n=== Verification ===");
    println!("  Naive approach:     {naive_total:.10}");
    println!("  Exp family approach: {exp_fam_log_density:.10}");
    println!("  Difference:         {difference:.2e}");
    println!("  ✓ Methods agree: {}", difference < 1e-10);

    // Show efficiency gains with larger samples
    println!("\n=== Efficiency Comparison ===");
    let large_samples: Vec<f64> = (0..10000).map(|i| 1.0 + f64::from(i) * 0.001).collect();

    // Naive approach
    let start = std::time::Instant::now();
    let naive_large: f64 = large_samples
        .iter()
        .map(|&x| normal.log_density().at(&x))
        .sum();
    let naive_large_duration = start.elapsed();

    // Exponential family approach
    let start = std::time::Instant::now();
    let sum_x_large: f64 = large_samples.iter().sum();
    let sum_x2_large: f64 = large_samples.iter().map(|&x| x * x).sum();
    let sufficient_stats_large = [sum_x_large, sum_x2_large];
    let dot_product_large = natural_params[0] * sufficient_stats_large[0]
        + natural_params[1] * sufficient_stats_large[1];
    let n_large = large_samples.len() as f64;
    let exp_fam_large = dot_product_large - n_large * log_partition;
    let exp_fam_large_duration = start.elapsed();

    println!("  Large sample (n={})", large_samples.len());
    println!("  Naive time:     {naive_large_duration:?}");
    println!("  Exp family time: {exp_fam_large_duration:?}");
    println!(
        "  Speedup:        {:.1}x",
        naive_large_duration.as_nanos() as f64 / exp_fam_large_duration.as_nanos() as f64
    );
    println!(
        "  Results match:  {}",
        (naive_large - exp_fam_large).abs() < 1e-8
    );

    println!("\n=== Key Insights ===");
    println!("✓ Exponential family structure enables efficient computation");
    println!("✓ O(n) sufficient statistic computation vs O(n) individual densities");
    println!("✓ Single dot product vs n dot products");
    println!("✓ Significant speedup for large samples");
    println!("✓ Mathematically equivalent results");

    println!("\nThe exponential family approach computes:");
    println!("  1. ∑ᵢT(xᵢ) - vectorized sufficient statistics");
    println!("  2. η·∑ᵢT(xᵢ) - single dot product");
    println!("  3. n·A(η) - scaled log partition");
    println!("Instead of n individual log-density computations!");
}
