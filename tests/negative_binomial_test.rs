//! Test for negative binomial distribution with `NegativeBinomialCoefficientMeasure`

use measures::HasLogDensity;
use measures::distributions::discrete::negative_binomial::NegativeBinomial;
use measures::exponential_family::traits::ExponentialFamily;

#[test]
fn test_negative_binomial_with_base_measure() {
    let neg_binomial = NegativeBinomial::new(5, 0.6_f64);

    // Test that we can compute log-density
    let log_density: f64 = neg_binomial.exp_fam_log_density(&3);

    // The result should be finite (not NaN or infinity)
    assert!(log_density.is_finite());

    // Test a few more values
    for x in 0..=10 {
        let log_density: f64 = neg_binomial.exp_fam_log_density(&x);
        assert!(
            log_density.is_finite(),
            "Log density should be finite for x={x}"
        );
    }
}

#[test]
fn test_negative_binomial_base_measure() {
    let neg_binomial = NegativeBinomial::new(3, 0.4_f64);
    let base_measure = neg_binomial.base_measure();

    // Test that base measure gives correct log-density
    for x in 0..=10 {
        let log_density: f64 = base_measure.log_density_wrt_root(&x);
        assert!(
            log_density.is_finite(),
            "Base measure log density should be finite for x={x}"
        );
        // The base measure should give positive values (since it's log(C(x+r-1,x)))
        assert!(
            log_density >= 0.0,
            "Base measure log density should be >= 0 for x={x}"
        );
    }
}

#[test]
fn test_negative_binomial_manual_computation() {
    // Test against manual computation for a simple case
    let neg_binomial = NegativeBinomial::new(2, 0.5_f64);

    // For NegativeBinomial(2, 0.5) with exponential family parameterization:
    // η = log(p/(1-p)) = log(0.5/0.5) = 0
    // A(η) = -r * log(1 - sigmoid(η)) = -2 * log(1 - 0.5) = -2 * log(0.5) = 2 * log(2)
    // T(1) = 1
    // h(1) = C(1+2-1, 1) = C(2,1) = 2, so log h(1) = log(2)
    //
    // Exponential family log density = η·T(x) - A(η) + log h(x)
    //                                = 0*1 - 2*log(2) + log(2)
    //                                = -log(2)
    let expected = -(2.0_f64.ln());
    let actual: f64 = neg_binomial.exp_fam_log_density(&1);

    // Allow for small numerical differences
    let diff = (actual - expected).abs();
    assert!(
        diff < 1e-10,
        "Expected {expected}, got {actual}, diff = {diff}"
    );
}

#[test]
fn test_negative_binomial_exponential_family_components() {
    let neg_binomial = NegativeBinomial::new(3, 0.6_f64);

    // Test natural parameters (log-odds)
    let (natural_params, log_partition) = neg_binomial.natural_and_log_partition();

    // η = log(p/(1-p)) = log(0.6/0.4) = log(1.5)
    let expected_eta = (0.6_f64 / 0.4_f64).ln();
    assert!((natural_params[0] - expected_eta).abs() < 1e-10);

    // Test sufficient statistics
    let suff_stat_0 = neg_binomial.sufficient_statistic(&0);
    let suff_stat_3 = neg_binomial.sufficient_statistic(&3);
    let suff_stat_5 = neg_binomial.sufficient_statistic(&5);

    // For x=0: [0]
    assert_eq!(suff_stat_0, [0.0]);
    // For x=3: [3]
    assert_eq!(suff_stat_3, [3.0]);
    // For x=5: [5]
    assert_eq!(suff_stat_5, [5.0]);

    // Test log partition: A(η) = -r * log(1 - sigmoid(η))
    let exp_eta = expected_eta.exp();
    let sigmoid_eta = exp_eta / (1.0 + exp_eta);
    let expected_log_partition = -3.0 * (1.0 - sigmoid_eta).ln();
    assert!((log_partition - expected_log_partition).abs() < 1e-10);
}
