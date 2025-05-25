//! Test for binomial distribution with `BinomialCoefficientMeasure`

use measures::core::HasLogDensity;
use measures::distributions::discrete::binomial::Binomial;
use measures::exponential_family::traits::ExponentialFamily;

#[test]
fn test_binomial_with_base_measure() {
    let binomial = Binomial::new(10, 0.3_f64);

    // Test that we can compute log-density
    let log_density: f64 = binomial.exp_fam_log_density(&3);

    // The result should be finite (not NaN or infinity)
    assert!(log_density.is_finite());

    // Test a few more values
    for k in 0..=10 {
        let log_density: f64 = binomial.exp_fam_log_density(&k);
        assert!(
            log_density.is_finite(),
            "Log density should be finite for k={k}"
        );
    }

    // Test that k > n gives -infinity (outside support)
    let log_density: f64 = binomial.exp_fam_log_density(&15);
    assert!(
        log_density.is_infinite() && log_density.is_sign_negative(),
        "Log density should be -infinity for k > n"
    );
}

#[test]
fn test_binomial_base_measure() {
    let binomial = Binomial::new(5, 0.4_f64);
    let base_measure = binomial.base_measure();

    // Test that base measure gives correct log-density
    for k in 0..=5 {
        let log_density: f64 = base_measure.log_density_wrt_root(&k);
        assert!(
            log_density.is_finite(),
            "Base measure log density should be finite for k={k}"
        );
        // The base measure should give positive values (since it's log(C(n,k)))
        assert!(
            log_density >= 0.0,
            "Base measure log density should be >= 0 for k={k}"
        );
    }

    // Test that k > n gives -infinity
    let log_density: f64 = base_measure.log_density_wrt_root(&10);
    assert!(
        log_density.is_infinite() && log_density.is_sign_negative(),
        "Base measure should give -infinity for k > n"
    );
}

#[test]
fn test_binomial_manual_computation() {
    // Test against manual computation for a simple case
    let binomial = Binomial::new(3, 0.5_f64);

    // For Binomial(3, 0.5), P(X=1) = C(3,1) * 0.5^1 * 0.5^2 = 3 * 0.5^3 = 3/8
    // So log P(X=1) = log(3) + 3*log(0.5) = log(3) - 3*log(2)
    let expected = 3.0_f64.ln() - 3.0 * 2.0_f64.ln();
    let actual: f64 = binomial.exp_fam_log_density(&1);

    // Allow for small numerical differences
    let diff = (actual - expected).abs();
    assert!(
        diff < 1e-10,
        "Expected {expected}, got {actual}, diff = {diff}"
    );
}
