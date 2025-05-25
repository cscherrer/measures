//! Test for categorical distribution with proper exponential family computation

use measures::distributions::discrete::categorical::Categorical;
use measures::exponential_family::traits::ExponentialFamily;

#[test]
fn test_categorical_exponential_family_computation() {
    // Test a simple 3-category distribution
    let probs = vec![0.2_f64, 0.3, 0.5];
    let categorical = Categorical::new(probs.clone());

    // Test that we can compute log-density for all categories
    for (i, &expected_prob) in probs.iter().enumerate() {
        let log_density: f64 = categorical.exp_fam_log_density(&i);
        let expected_log_density = expected_prob.ln();

        // Allow for small numerical differences
        let diff = (log_density - expected_log_density).abs();
        assert!(
            diff < 1e-10,
            "Category {i}: Expected {expected_log_density}, got {log_density}, diff = {diff}"
        );
    }

    // Test that out-of-bounds gives -infinity
    let log_density: f64 = categorical.exp_fam_log_density(&3);
    assert!(
        log_density.is_infinite() && log_density.is_sign_negative(),
        "Out-of-bounds category should give -infinity"
    );
}

#[test]
fn test_categorical_uniform_distribution() {
    // Test uniform distribution with 4 categories
    let categorical = Categorical::uniform(4);
    let expected_log_prob = (0.25_f64).ln();

    for i in 0..4 {
        let log_density: f64 = categorical.exp_fam_log_density(&i);
        let diff = (log_density - expected_log_prob).abs();
        assert!(
            diff < 1e-10,
            "Uniform category {i}: Expected {expected_log_prob}, got {log_density}, diff = {diff}"
        );
    }
}

#[test]
fn test_categorical_exponential_family_components() {
    let probs = vec![0.1_f64, 0.4, 0.5];
    let categorical = Categorical::new(probs);

    // Test natural parameters (log-odds relative to last category)
    let (natural_params, log_partition) = categorical.natural_and_log_partition();

    // η₁ = log(p₁/p₃) = log(0.1/0.5) = log(0.2)
    // η₂ = log(p₂/p₃) = log(0.4/0.5) = log(0.8)
    let expected_eta1 = (0.1_f64 / 0.5_f64).ln();
    let expected_eta2 = (0.4_f64 / 0.5_f64).ln();

    assert!((natural_params[0] - expected_eta1).abs() < 1e-10);
    assert!((natural_params[1] - expected_eta2).abs() < 1e-10);

    // Test sufficient statistics
    let suff_stat_0 = categorical.sufficient_statistic(&0);
    let suff_stat_1 = categorical.sufficient_statistic(&1);
    let suff_stat_2 = categorical.sufficient_statistic(&2);

    // For category 0: [1, 0]
    assert_eq!(suff_stat_0, vec![1.0, 0.0]);
    // For category 1: [0, 1]
    assert_eq!(suff_stat_1, vec![0.0, 1.0]);
    // For category 2 (reference): [0, 0]
    assert_eq!(suff_stat_2, vec![0.0, 0.0]);

    // Test log partition: A(η) = log(1 + exp(η₁) + exp(η₂))
    let expected_log_partition = (1.0_f64 + expected_eta1.exp() + expected_eta2.exp()).ln();
    assert!((log_partition - expected_log_partition).abs() < 1e-10);
}
