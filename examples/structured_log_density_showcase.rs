//! Comprehensive showcase of structured log-density decomposition.
//!
//! This example demonstrates how the unified log-density decomposition framework
//! works for both exponential families and non-exponential families, enabling
//! efficient computation for IID samples and parameter optimization.

use measures::distributions::continuous::cauchy::Cauchy;
use measures::distributions::continuous::student_t::StudentT;
use measures::{HasLogDensityDecomposition, LogDensityBuilder, Normal};
use std::time::Instant;

fn main() {
    println!("=== Structured Log-Density Decomposition Showcase ===\n");

    // 1. Compare exponential family vs non-exponential family
    demonstrate_decomposition_structure();

    // 2. Show efficiency gains for IID samples
    demonstrate_iid_efficiency();

    // 3. Parameter optimization scenario
    demonstrate_parameter_optimization();

    // 4. Caching strategies
    demonstrate_caching_strategies();
}

fn demonstrate_decomposition_structure() {
    println!("1. Log-Density Decomposition Structure");
    println!("=====================================");

    // Normal distribution (exponential family)
    let normal = Normal::new(1.0, 2.0);
    println!("Normal(μ=1.0, σ=2.0):");
    println!("  log p(x|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²");
    println!("  Structure: constant + param_term + mixed_term");

    // Cauchy distribution (non-exponential family)
    let cauchy = Cauchy::new(0.0, 1.0);
    println!("\nCauchy(x₀=0.0, γ=1.0):");
    println!("  log p(x|x₀,γ) = -log(π) - log(γ) - log(1 + ((x-x₀)/γ)²)");
    println!("  Structure: constant + param_term + mixed_term");

    // Student's t distribution (non-exponential family)
    let student_t = StudentT::new(3.0);
    println!("\nStudent's t(ν=3.0):");
    println!("  log p(x|ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - (ν+1)/2*log(1 + x²/ν)");
    println!("  Structure: param_term + mixed_term");

    // Test decomposition accuracy
    let x = 1.5;

    let normal_direct: f64 = normal.log_density().at(&x);
    let normal_decomp = normal.log_density_decomposition();
    let normal_via_decomp = normal_decomp.evaluate(&x, &(1.0, 2.0));
    println!(
        "\nNormal - Direct: {:.6}, Decomposed: {:.6}, Diff: {:.2e}",
        normal_direct,
        normal_via_decomp,
        (normal_direct - normal_via_decomp).abs()
    );

    let cauchy_direct: f64 = cauchy.log_density().at(&x);
    let cauchy_decomp = cauchy.log_density_decomposition();
    let cauchy_via_decomp = cauchy_decomp.evaluate(&x, &(0.0, 1.0));
    println!(
        "Cauchy - Direct: {:.6}, Decomposed: {:.6}, Diff: {:.2e}",
        cauchy_direct,
        cauchy_via_decomp,
        (cauchy_direct - cauchy_via_decomp).abs()
    );

    let t_direct: f64 = student_t.log_density().at(&x);
    let t_decomp = student_t.log_density_decomposition();
    let t_via_decomp = t_decomp.evaluate(&x, &3.0);
    println!(
        "Student's t - Direct: {:.6}, Decomposed: {:.6}, Diff: {:.2e}",
        t_direct,
        t_via_decomp,
        (t_direct - t_via_decomp).abs()
    );

    println!();
}

fn demonstrate_iid_efficiency() {
    println!("2. IID Sample Efficiency");
    println!("========================");

    // Generate sample data
    let samples: Vec<f64> = (0..10000).map(|i| f64::from(i) * 0.001 - 5.0).collect();
    println!("Testing with {} samples", samples.len());

    // Test Cauchy distribution
    let cauchy = Cauchy::new(0.0, 1.0);

    // Method 1: Individual log-density computations
    let start = Instant::now();
    let individual_sum: f64 = samples.iter().map(|&x| cauchy.log_density().at(&x)).sum();
    let individual_time = start.elapsed();

    // Method 2: IID decomposition
    let start = Instant::now();
    let iid_result = cauchy.log_density_iid(&samples);
    let iid_time = start.elapsed();

    println!("\nCauchy IID computation:");
    println!("  Individual sum: {individual_sum:.6} (took {individual_time:?})");
    println!("  IID method:     {iid_result:.6} (took {iid_time:?})");
    println!(
        "  Difference:     {:.2e}",
        (individual_sum - iid_result).abs()
    );
    println!(
        "  Speedup:        {:.1}x",
        individual_time.as_nanos() as f64 / iid_time.as_nanos() as f64
    );

    // Test Student's t distribution
    let student_t = StudentT::new(5.0);

    let start = Instant::now();
    let t_individual_sum: f64 = samples
        .iter()
        .map(|&x| student_t.log_density().at(&x))
        .sum();
    let t_individual_time = start.elapsed();

    let start = Instant::now();
    let t_iid_result = student_t.log_density_iid(&samples);
    let t_iid_time = start.elapsed();

    println!("\nStudent's t IID computation:");
    println!("  Individual sum: {t_individual_sum:.6} (took {t_individual_time:?})");
    println!("  IID method:     {t_iid_result:.6} (took {t_iid_time:?})");
    println!(
        "  Difference:     {:.2e}",
        (t_individual_sum - t_iid_result).abs()
    );
    println!(
        "  Speedup:        {:.1}x",
        t_individual_time.as_nanos() as f64 / t_iid_time.as_nanos() as f64
    );

    println!();
}

fn demonstrate_parameter_optimization() {
    println!("3. Parameter Optimization Scenario");
    println!("==================================");

    // Simulate parameter optimization for Cauchy distribution
    let samples = vec![0.1, -0.5, 1.2, -0.8, 0.3, 2.1, -1.1, 0.7];
    println!("Optimizing Cauchy parameters for {} samples", samples.len());

    // Test different parameter values
    let location_values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let scale_values = vec![0.5, 1.0, 1.5, 2.0];

    println!("\nParameter grid search results:");
    println!("Location  Scale    Log-Likelihood");
    println!("--------------------------------");

    let mut best_ll = f64::NEG_INFINITY;
    let mut best_params = (0.0, 1.0);

    for &location in &location_values {
        for &scale in &scale_values {
            let cauchy = Cauchy::new(location, scale);
            let log_likelihood = cauchy.log_density_iid(&samples);

            if log_likelihood > best_ll {
                best_ll = log_likelihood;
                best_params = (location, scale);
            }

            println!("{location:8.1}  {scale:5.1}    {log_likelihood:12.6}");
        }
    }

    println!(
        "\nBest parameters: location={:.1}, scale={:.1}",
        best_params.0, best_params.1
    );
    println!("Best log-likelihood: {best_ll:.6}");

    // Show how decomposition enables efficient parameter optimization
    println!("\nDecomposition-based optimization:");
    let cauchy = Cauchy::new(best_params.0, best_params.1);
    let decomp = cauchy.log_density_decomposition();

    // In real optimization, we'd cache data-dependent terms
    let data_terms_sum: f64 = samples
        .iter()
        .map(|&x| decomp.evaluate_data_terms(&x))
        .sum();

    println!("  Data-dependent terms (cached): {data_terms_sum:.6}");
    println!(
        "  Parameter-dependent terms: {:.6}",
        decomp.evaluate_param_terms(&best_params)
    );
    println!("  Constants: {:.6}", decomp.constant_sum());

    println!();
}

fn demonstrate_caching_strategies() {
    println!("4. Caching Strategies");
    println!("====================");

    let samples = [0.5, -0.3, 1.1, -0.7, 0.2, 1.8, -1.2, 0.9];
    let student_t = StudentT::new(4.0);
    let decomp = student_t.log_density_decomposition();

    println!("Student's t distribution with {} samples", samples.len());

    // Strategy 1: Cache parameter-dependent terms when data changes
    println!("\nStrategy 1: Fixed parameters, changing data");
    let param_terms = decomp.evaluate_param_terms(&4.0);
    println!("  Cached parameter terms: {param_terms:.6}");

    // Simulate adding new data points
    let new_samples = vec![0.1, -0.4, 0.8];
    for &x in &new_samples {
        let mixed_terms: f64 = decomp
            .mixed_terms
            .iter()
            .map(|term| (term.compute)(&x, &4.0))
            .sum();
        println!("  New sample x={x:.1}: mixed_terms={mixed_terms:.6}");
    }

    // Strategy 2: Cache data-dependent terms when parameters change
    println!("\nStrategy 2: Fixed data, changing parameters");

    // For Student's t, there are no pure data terms, only mixed terms
    // But we can still demonstrate the concept
    let test_nus = vec![2.0, 3.0, 5.0, 10.0];

    println!("  Testing different degrees of freedom:");
    for &nu in &test_nus {
        let param_terms = decomp.evaluate_param_terms(&nu);
        let mixed_sum: f64 = samples
            .iter()
            .map(|&x| {
                decomp
                    .mixed_terms
                    .iter()
                    .map(|term| (term.compute)(&x, &nu))
                    .sum::<f64>()
            })
            .sum();

        let total = param_terms * (samples.len() as f64) + mixed_sum;
        println!(
            "    ν={nu:4.1}: param_terms={param_terms:8.4}, mixed_sum={mixed_sum:8.4}, total={total:8.4}"
        );
    }

    // Strategy 3: Incremental updates for streaming data
    println!("\nStrategy 3: Incremental updates");
    let mut running_mixed_sum = 0.0;
    let nu = 3.0;

    println!("  Adding samples incrementally:");
    for (i, &x) in samples.iter().enumerate() {
        let mixed_term: f64 = decomp
            .mixed_terms
            .iter()
            .map(|term| (term.compute)(&x, &nu))
            .sum();
        running_mixed_sum += mixed_term;

        let param_contribution = decomp.evaluate_param_terms(&nu) * ((i + 1) as f64);
        let total = param_contribution + running_mixed_sum;

        println!(
            "    Sample {}: x={:5.1}, mixed={:8.4}, total={:8.4}",
            i + 1,
            x,
            mixed_term,
            total
        );
    }

    println!("\n=== Framework Benefits ===");
    println!("✓ Unified decomposition for all distributions");
    println!("✓ Efficient IID sample computation");
    println!("✓ Optimized parameter optimization");
    println!("✓ Flexible caching strategies");
    println!("✓ Incremental computation support");
    println!("✓ Same API for exponential and non-exponential families");
}
