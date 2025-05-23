use measures::{
    IIDExtension, LogDensityBuilder, Normal, distributions::discrete::poisson::Poisson,
};

fn main() {
    println!("=== IID Exponential Family: Theory and Practice ===\n");

    // Demonstrate the mathematical foundation described in the user's note:
    // For exponential family f(x|θ) = h(x) exp(η(θ)ᵀT(x) - A(η(θ)))
    // IID collection: f(x₁,...,xₙ|θ) = [∏ᵢh(xᵢ)]exp(η(θ)ᵀ∑ᵢT(xᵢ) - nA(η(θ)))

    println!("=== Theoretical Foundation ===");
    println!("For an exponential family f(x|θ) = h(x) exp(η(θ)ᵀT(x) - A(η(θ))),");
    println!("the IID collection of n samples has the structure:");
    println!("  f(x₁,...,xₙ|θ) = [∏ᵢh(xᵢ)] exp(η(θ)ᵀ∑ᵢT(xᵢ) - nA(η(θ)))");
    println!();
    println!("Key properties:");
    println!("  - Base measure: ∏ᵢh(xᵢ) (product of individual base measures)");
    println!("  - Sufficient statistic: ∑ᵢT(xᵢ) (sum of individual sufficient statistics)");
    println!("  - Natural parameter: η(θ) (same as original)");
    println!("  - Log-partition function: nA(η(θ)) (scaled by sample size)");
    println!();

    // Example 1: Normal distribution
    println!("=== Example 1: Normal Distribution ===");
    let normal = Normal::new(2.0, 1.5);
    let iid_normal = normal.clone().iid();

    println!("Base distribution: Normal(μ=2.0, σ=1.5)");
    println!("Natural parameters for Normal: η = [μ/σ², -1/(2σ²)]");
    println!("Sufficient statistics for Normal: T(x) = [x, x²]");

    let test_sample = vec![1.8, 2.3, 1.9, 2.1];
    println!("\nTest sample: {test_sample:?}");

    // Individual analysis
    println!("\nIndividual analysis:");
    let mut total_log_density = 0.0;
    for (i, &x) in test_sample.iter().enumerate() {
        let log_density = normal.log_density().at(&x);
        total_log_density += log_density;
        println!(
            "  x{} = {:.1}: log p(x{}) = {:.6}",
            i + 1,
            x,
            i + 1,
            log_density
        );
    }
    println!("  Total (sum): {total_log_density:.6}");

    // IID analysis
    let iid_log_density: f64 = iid_normal.log_density(&test_sample);
    println!("\nIID analysis:");
    println!("  Joint log-density: {iid_log_density:.6}");
    let difference: f64 = (iid_log_density - total_log_density).abs();
    println!("  Difference from sum: {difference:.2e}");
    println!("  ✓ Demonstrates: log p(x₁,...,xₙ) = ∑ᵢ log p(xᵢ)");

    // Example 2: Poisson distribution
    println!("\n=== Example 2: Poisson Distribution ===");
    let poisson = Poisson::new(3.2_f64);
    let iid_poisson = poisson.clone().iid();

    println!("Base distribution: Poisson(λ=3.2)");
    println!("Natural parameters for Poisson: η = [log(λ)]");
    println!("Sufficient statistics for Poisson: T(x) = [x]");
    println!("Log-partition function for Poisson: A(η) = eᶯ = λ");

    let poisson_sample = vec![2, 4, 3, 3, 5, 2];
    println!("\nTest sample: {poisson_sample:?}");

    // Individual analysis for Poisson
    println!("\nIndividual analysis:");
    let mut poisson_total = 0.0;
    for (i, &x) in poisson_sample.iter().enumerate() {
        let log_density = poisson.log_density().at(&x);
        poisson_total += log_density;
        println!(
            "  x{} = {}: log p(x{}) = {:.6}",
            i + 1,
            x,
            i + 1,
            log_density
        );
    }
    println!("  Total (sum): {poisson_total:.6}");

    // IID analysis for Poisson
    let poisson_iid_log_density = iid_poisson.log_density(&poisson_sample);
    println!("\nIID analysis:");
    println!("  Joint log-density: {poisson_iid_log_density:.6}");
    println!(
        "  Difference from sum: {:.2e}",
        (poisson_iid_log_density - poisson_total).abs()
    );

    // Demonstrate sufficient statistic property
    println!("\n=== Sufficient Statistics Property ===");
    println!("For IID exponential families, all information about θ comes through ∑ᵢT(xᵢ)");

    // Example with different samples that have same sufficient statistics
    let sample_a = vec![1.0, 3.0, 2.0];
    let sample_b = vec![2.0, 2.0, 2.0]; // Same mean, different individual values

    println!("\nNormal distribution example:");
    println!("Sample A: {sample_a:?}");
    println!("Sample B: {sample_b:?}");

    let mean_a: f64 = sample_a.iter().sum::<f64>() / sample_a.len() as f64;
    let mean_b: f64 = sample_b.iter().sum::<f64>() / sample_b.len() as f64;
    let sum_squares_a: f64 = sample_a.iter().map(|&x| x * x).sum();
    let sum_squares_b: f64 = sample_b.iter().map(|&x| x * x).sum();

    println!(
        "Sample A: mean = {:.3}, sum = {:.3}, sum_squares = {:.3}",
        mean_a,
        sample_a.iter().sum::<f64>(),
        sum_squares_a
    );
    println!(
        "Sample B: mean = {:.3}, sum = {:.3}, sum_squares = {:.3}",
        mean_b,
        sample_b.iter().sum::<f64>(),
        sum_squares_b
    );

    if (mean_a - mean_b).abs() < 1e-10 && (sum_squares_a - sum_squares_b).abs() < 1e-10 {
        println!("✓ Same sufficient statistics → same information about parameters");
    }

    // Demonstrate sample size scaling
    println!("\n=== Sample Size Scaling ===");
    println!("Log-partition function scales as n·A(η) for n samples");

    let base_sample = vec![2.0, 2.5];
    let base_log_likelihood = iid_normal.log_density(&base_sample);

    println!("\nBase sample: {base_sample:?}");
    println!("Log-likelihood: {base_log_likelihood:.6}");

    for multiplier in 2..=4 {
        let repeated_sample: Vec<f64> = base_sample
            .iter()
            .cycle()
            .take(multiplier * base_sample.len())
            .copied()
            .collect();

        let repeated_log_likelihood = iid_normal.log_density(&repeated_sample);
        let expected_scaling = base_log_likelihood * multiplier as f64;

        println!(
            "{}× repeated sample (n={}): log-likelihood = {:.6}",
            multiplier,
            repeated_sample.len(),
            repeated_log_likelihood
        );
        println!("  Expected ({multiplier}× base): {expected_scaling:.6}");
        println!(
            "  Ratio: {:.3}",
            repeated_log_likelihood / base_log_likelihood
        );
    }

    // Model comparison using likelihood ratios
    println!("\n=== Model Comparison via Likelihood Ratios ===");
    println!("IID structure enables efficient model comparison");

    let comparison_data = vec![2.1, 2.3, 1.9, 2.4, 2.0];
    println!("\nComparison data: {comparison_data:?}");

    let models = vec![
        ("Normal(2.0, 1.0)", Normal::new(2.0, 1.0)),
        ("Normal(2.2, 0.8)", Normal::new(2.2, 0.8)),
        ("Normal(2.5, 1.2)", Normal::new(2.5, 1.2)),
    ];

    let mut log_likelihoods = Vec::new();

    for (name, model) in &models {
        let iid_model = model.clone().iid();
        let log_likelihood = iid_model.log_density(&comparison_data);
        log_likelihoods.push(log_likelihood);
        println!("{name}: log L = {log_likelihood:.6}");
    }

    // Find best model
    let best_idx = log_likelihoods
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!("\nBest model: {}", models[best_idx].0);

    // Compute likelihood ratios
    let best_log_likelihood = log_likelihoods[best_idx];
    println!("\nLikelihood ratios (vs best):");
    for (i, (name, _)) in models.iter().enumerate() {
        let log_diff: f64 = log_likelihoods[i] - best_log_likelihood;
        let ratio: f64 = log_diff.exp();
        println!("  {name}: ratio = {ratio:.6}");
    }

    println!("\n=== Summary ===");
    println!("✓ IID exponential families maintain the exponential family structure");
    println!("✓ Sufficient statistics become sums: ∑ᵢT(xᵢ)");
    println!("✓ Natural parameters remain unchanged: η(θ)");
    println!("✓ Log-partition scales with sample size: n·A(η)");
    println!("✓ Enables efficient likelihood-based inference");
    println!("✓ Supports model comparison via likelihood ratios");
    println!("\nThis mathematical foundation is crucial for:");
    println!("  - Maximum likelihood estimation");
    println!("  - Bayesian inference with conjugate priors");
    println!("  - Efficient computation with large datasets");
    println!("  - Statistical inference theory");
}
