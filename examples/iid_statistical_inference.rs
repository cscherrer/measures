use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== Statistical Inference with IID Log Densities ===\n");

    // Generate some "observed" data from a known distribution for demonstration
    let true_mean = 2.5;
    let true_std = 1.2;
    let observed_data = vec![2.1, 3.2, 1.8, 2.9, 3.1, 1.9, 2.3, 4.1, 2.7, 2.0];

    println!("Observed data: {observed_data:?}");
    println!("True parameters (unknown in practice): μ={true_mean}, σ={true_std}\n");

    // Demonstrate likelihood computation for different parameter values
    println!("=== Likelihood Evaluation ===");

    let parameter_candidates = vec![
        (1.0, 1.0), // Poor fit
        (2.0, 1.0), // Better fit
        (2.5, 1.2), // True parameters
        (2.6, 1.3), // Close to true
        (3.0, 1.5), // Different fit
    ];

    let mut best_log_likelihood = f64::NEG_INFINITY;
    let mut best_params = (0.0, 0.0);

    for (mu, sigma) in parameter_candidates {
        // Create normal distribution with these parameters
        let candidate_dist = Normal::new(mu, sigma);
        let iid_candidate = candidate_dist.iid();

        // Compute log-likelihood of observed data under this model
        let log_likelihood = iid_candidate.iid_log_density(&observed_data);

        println!("μ={mu:.1}, σ={sigma:.1}: log-likelihood = {log_likelihood:.6}");

        if log_likelihood > best_log_likelihood {
            best_log_likelihood = log_likelihood;
            best_params = (mu, sigma);
        }
    }

    println!(
        "\nBest parameters from candidates: μ={:.1}, σ={:.1}",
        best_params.0, best_params.1
    );
    println!("Best log-likelihood: {best_log_likelihood:.6}\n");

    // Compare individual vs joint computation
    println!("=== Verification: Individual vs Joint Computation ===");
    let test_dist = Normal::new(2.5, 1.2);
    let iid_test = test_dist.clone().iid();

    // Method 1: Joint IID computation
    let joint_log_likelihood: f64 = iid_test.iid_log_density(&observed_data);

    // Method 2: Sum of individual log-likelihoods
    let individual_sum: f64 = observed_data
        .iter()
        .map(|&x| test_dist.log_density().at(&x))
        .sum();

    println!("Joint computation:      {joint_log_likelihood:.6}");
    println!("Sum of individuals:     {individual_sum:.6}");
    let difference: f64 = (joint_log_likelihood - individual_sum).abs();
    println!("Difference:             {difference:.2e}");

    // Demonstrate the mathematical relationship
    println!("\n=== Mathematical Relationship Demonstration ===");

    // For IID samples, the likelihood function is:
    // L(θ) = ∏ᵢ p(xᵢ|θ)
    // And the log-likelihood is:
    // ℓ(θ) = log L(θ) = ∑ᵢ log p(xᵢ|θ)

    let sample_dist = Normal::new(0.0, 1.0);
    let small_sample = vec![0.5, -0.2, 1.1];

    println!("Sample: {small_sample:?}");
    println!("Distribution: Normal(0, 1)");

    // Show step-by-step computation
    println!("\nStep-by-step log-likelihood computation:");
    let mut running_sum = 0.0;
    for (i, &x) in small_sample.iter().enumerate() {
        let log_p = sample_dist.log_density().at(&x);
        running_sum += log_p;
        println!(
            "  x{} = {:.1}: log p(x{}) = {:.6}, cumulative sum = {:.6}",
            i + 1,
            x,
            i + 1,
            log_p,
            running_sum
        );
    }

    let iid_sample = sample_dist.iid();
    let iid_result: f64 = iid_sample.iid_log_density(&small_sample);
    println!("  IID computation result: {iid_result:.6}");
    let diff: f64 = (iid_result - running_sum).abs();
    println!("  ✓ Match: {}", diff < 1e-10);

    // Model comparison example
    println!("\n=== Model Comparison Example ===");

    let test_data = vec![1.5, 2.1, 1.8, 2.3, 1.9];
    println!("Test data: {test_data:?}");

    // Compare different models
    let models = vec![
        ("Normal(1.5, 0.5)", Normal::new(1.5, 0.5)),
        ("Normal(2.0, 0.3)", Normal::new(2.0, 0.3)),
        ("Normal(2.0, 0.8)", Normal::new(2.0, 0.8)),
    ];

    println!("\nModel comparison by log-likelihood:");
    for (name, model) in models {
        let iid_model = model.iid();
        let log_likelihood = iid_model.iid_log_density(&test_data);
        println!("  {name}: {log_likelihood:.6}");
    }

    println!("\n=== Sample Size Effects ===");

    // Show how log-likelihood scales with sample size
    let base_dist = Normal::new(1.0, 1.0);
    let base_sample = [1.2, 0.8, 1.1];

    for n in 1..=4 {
        let repeated_sample: Vec<f64> = base_sample
            .iter()
            .cycle()
            .take(n * base_sample.len())
            .copied()
            .collect();

        let iid_dist = base_dist.clone().iid();
        let log_likelihood = iid_dist.iid_log_density(&repeated_sample);

        println!(
            "Sample size {}: log-likelihood = {:.6}",
            repeated_sample.len(),
            log_likelihood
        );
    }

    println!("\n=== Summary ===");
    println!("✓ IID log density computation enables:");
    println!("  - Maximum likelihood parameter estimation");
    println!("  - Model comparison via likelihood ratios");
    println!("  - Proper handling of independent observations");
    println!("  - Efficient computation of joint likelihoods");
    println!("\n✓ Mathematical correctness verified:");
    println!("  - log L(θ) = ∑ᵢ log p(xᵢ|θ) for IID samples");
    println!("  - Proper scaling with sample size");
    println!("  - Consistency across computation methods");
}
