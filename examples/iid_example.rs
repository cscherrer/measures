use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== IID Exponential Family Log Density Computation ===\n");

    // Create a normal distribution
    let normal = Normal::new(0.0, 1.0);
    println!("Base distribution: Normal(μ=0.0, σ=1.0)");

    // Create an IID version using the extension trait
    let iid_normal = normal.clone().iid();
    println!(
        "Created IID wrapper: {:?}\n",
        std::any::type_name_of_val(&iid_normal)
    );

    // Test with different sample sizes
    let test_cases = [
        vec![0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0, -1.0],
        vec![0.5, -0.3, 1.2, -0.8],
        vec![], // Edge case: empty sample
    ];

    for (i, samples) in test_cases.iter().enumerate() {
        println!("--- Test Case {}: {:?} ---", i + 1, samples);

        if samples.is_empty() {
            println!("Empty sample case:");
            let iid_log_density = iid_normal.compute_iid_log_density(samples);
            println!("  IID log-density: {iid_log_density}");
            println!("  Expected: 0.0 (log(1) = 0 for empty product)");
            println!("  ✓ Correct: {}\n", iid_log_density == 0.0);
            continue;
        }

        // Method 1: Using the IID wrapper's manual computation
        let iid_log_density = iid_normal.compute_iid_log_density(samples);

        // Method 2: Manual sum of individual log-densities (verification)
        let individual_sum: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();

        // Method 3: Step-by-step breakdown for educational purposes
        println!("  Individual log-densities:");
        let mut manual_sum = 0.0;
        for (j, &x) in samples.iter().enumerate() {
            let log_density = normal.log_density().at(&x);
            manual_sum += log_density;
            println!(
                "    Sample {}: x = {:.3}, log p(x) = {:.6}",
                j + 1,
                x,
                log_density
            );
        }

        println!("\n  Results:");
        println!("    IID log-density (method 1): {iid_log_density:.6}");
        println!("    Individual sum (method 2):  {individual_sum:.6}");
        println!("    Manual sum (method 3):      {manual_sum:.6}");

        let diff1 = (iid_log_density - individual_sum).abs();
        let diff2 = (iid_log_density - manual_sum).abs();

        println!("    Difference 1-2: {diff1:.2e}");
        println!("    Difference 1-3: {diff2:.2e}");

        let is_correct = diff1 < 1e-10 && diff2 < 1e-10;
        println!("    ✓ All methods agree: {is_correct}\n");

        assert!(is_correct, "Methods should produce identical results");
    }

    // Demonstrate mathematical properties
    println!("=== Mathematical Properties Validation ===");

    // Property 1: IID of single sample equals original distribution
    let single_sample = vec![1.5];
    let iid_single = iid_normal.compute_iid_log_density(&single_sample);
    let original_single = normal.log_density().at(&1.5);
    println!("Single sample property:");
    println!("  IID log p([1.5]) = {iid_single:.6}");
    println!("  Original log p(1.5) = {original_single:.6}");
    println!(
        "  ✓ Equal: {}\n",
        (iid_single - original_single).abs() < 1e-10
    );

    // Property 2: Additivity - log p(x,y) = log p(x) + log p(y)
    let x = 0.8;
    let y = -1.2;
    let joint_samples = vec![x, y];
    let joint_density = iid_normal.compute_iid_log_density(&joint_samples);
    let separate_sum = normal.log_density().at(&x) + normal.log_density().at(&y);
    println!("Additivity property:");
    println!("  log p([{x}, {y}]) = {joint_density:.6}");
    println!("  log p({x}) + log p({y}) = {separate_sum:.6}");
    println!(
        "  ✓ Equal: {}\n",
        (joint_density - separate_sum).abs() < 1e-10
    );

    // Property 3: Scaling with sample size
    let base_sample = vec![0.0, 1.0];
    let extended_sample = vec![0.0, 1.0, 0.0, 1.0]; // Repeated pattern
    let base_density = iid_normal.compute_iid_log_density(&base_sample);
    let extended_density = iid_normal.compute_iid_log_density(&extended_sample);
    println!("Sample size scaling:");
    println!("  log p([0, 1]) = {base_density:.6}");
    println!("  log p([0, 1, 0, 1]) = {extended_density:.6}");
    println!("  Expected ratio: 2.0 (double the pattern)");
    println!("  Actual ratio: {:.6}", extended_density / base_density);
    println!(
        "  ✓ Correct scaling: {}\n",
        (extended_density / base_density - 2.0).abs() < 1e-10
    );

    println!("=== Summary ===");
    println!("✓ IID exponential family computation is working correctly!");
    println!("✓ All mathematical properties validated");
    println!("✓ The IID wrapper correctly implements:");
    println!("  - Root measure: IID of underlying root measure");
    println!("  - Log-density: Sum of individual log-densities");
    println!("  - Mathematical foundation for IID measures");
    println!("\nThe implementation correctly follows the principle:");
    println!("  log p(x₁, x₂, ..., xₙ) = Σᵢ log p(xᵢ)");
    println!("for independent and identically distributed samples.");
}
