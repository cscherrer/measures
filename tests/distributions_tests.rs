//! Comprehensive tests for distributions with low test coverage
//!
//! These tests focus on semantic invariants and mathematical properties
//! that should hold for probability distributions, using property-based testing.

use measures::Measure;
use measures::distributions::continuous::beta::Beta;
use measures::distributions::continuous::chi_squared::ChiSquared;
use measures::distributions::continuous::exponential::Exponential;
use measures::distributions::continuous::gamma::Gamma;
use measures::exponential_family::traits::ExponentialFamily;
use proptest::prelude::*;

// Constants for testing - no magic numbers
const SMALL_POSITIVE: f64 = 1e-6;
const LARGE_VALUE: f64 = 1000.0;
const UNIT_VALUE: f64 = 1.0;
const ZERO_VALUE: f64 = 0.0;
const HALF_VALUE: f64 = 0.5;
const TWO_VALUE: f64 = 2.0;
const THREE_VALUE: f64 = 3.0;
const TEN_VALUE: f64 = 10.0;

/// Tests for Beta distribution
mod beta_tests {
    use super::*;

    #[test]
    fn test_beta_creation_valid_parameters() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);
        assert_eq!(beta.alpha, TWO_VALUE);
        assert_eq!(beta.beta, THREE_VALUE);
    }

    #[test]
    #[should_panic(expected = "Alpha parameter must be positive")]
    fn test_beta_creation_invalid_alpha() {
        Beta::new(ZERO_VALUE, UNIT_VALUE);
    }

    #[test]
    #[should_panic(expected = "Beta parameter must be positive")]
    fn test_beta_creation_invalid_beta() {
        Beta::new(UNIT_VALUE, ZERO_VALUE);
    }

    #[test]
    fn test_beta_default() {
        let beta: Beta<f64> = Beta::default();
        assert_eq!(beta.alpha, UNIT_VALUE);
        assert_eq!(beta.beta, UNIT_VALUE);
    }

    #[test]
    fn test_beta_mean_calculation() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);
        let expected_mean = TWO_VALUE / (TWO_VALUE + THREE_VALUE); // 2/5 = 0.4
        assert!((beta.mean() - expected_mean).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_beta_variance_calculation() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);
        let alpha_plus_beta = TWO_VALUE + THREE_VALUE; // 5
        let expected_variance = (TWO_VALUE * THREE_VALUE)
            / (alpha_plus_beta * alpha_plus_beta * (alpha_plus_beta + UNIT_VALUE)); // 6/(25*6) = 0.04
        assert!((beta.variance() - expected_variance).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_beta_support() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);

        // Values in (0,1) should be in support
        assert!(beta.in_support(HALF_VALUE));
        assert!(beta.in_support(SMALL_POSITIVE));
        assert!(beta.in_support(UNIT_VALUE - SMALL_POSITIVE));

        // Boundary and outside values should not be in support
        assert!(!beta.in_support(ZERO_VALUE));
        assert!(!beta.in_support(UNIT_VALUE));
        assert!(!beta.in_support(-SMALL_POSITIVE));
        assert!(!beta.in_support(UNIT_VALUE + SMALL_POSITIVE));
    }

    #[test]
    fn test_beta_exponential_family_properties() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);

        // Test natural parameters
        let (natural_params, _log_partition) = beta.natural_and_log_partition();
        assert_eq!(natural_params[0], UNIT_VALUE); // α - 1 = 2 - 1 = 1
        assert_eq!(natural_params[1], TWO_VALUE); // β - 1 = 3 - 1 = 2

        // Test sufficient statistics
        let x = HALF_VALUE;
        let suff_stats = beta.sufficient_statistic(&x);
        assert!((suff_stats[0] - x.ln()).abs() < SMALL_POSITIVE);
        assert!((suff_stats[1] - (UNIT_VALUE - x).ln()).abs() < SMALL_POSITIVE);
    }

    proptest! {
        #[test]
        fn prop_beta_mean_in_unit_interval(
            alpha in SMALL_POSITIVE..TEN_VALUE,
            beta_param in SMALL_POSITIVE..TEN_VALUE
        ) {
            let beta = Beta::new(alpha, beta_param);
            let mean = beta.mean();
            assert!(mean > ZERO_VALUE && mean < UNIT_VALUE);
        }

        #[test]
        fn prop_beta_variance_positive(
            alpha in SMALL_POSITIVE..TEN_VALUE,
            beta_param in SMALL_POSITIVE..TEN_VALUE
        ) {
            let beta = Beta::new(alpha, beta_param);
            let variance = beta.variance();
            assert!(variance > ZERO_VALUE);
        }

        #[test]
        fn prop_beta_symmetric_when_equal_params(
            param in SMALL_POSITIVE..TEN_VALUE
        ) {
            let beta = Beta::new(param, param);
            let mean = beta.mean();
            // When α = β, mean should be 0.5
            assert!((mean - HALF_VALUE).abs() < SMALL_POSITIVE);
        }
    }
}

/// Tests for Chi-squared distribution
mod chi_squared_tests {
    use super::*;

    #[test]
    fn test_chi_squared_creation() {
        let chi_sq = ChiSquared::new(THREE_VALUE);
        assert_eq!(chi_sq.degrees_of_freedom, THREE_VALUE);
    }

    #[test]
    fn test_chi_squared_default() {
        let chi_sq: ChiSquared<f64> = ChiSquared::default();
        assert_eq!(chi_sq.degrees_of_freedom, UNIT_VALUE);
    }

    #[test]
    fn test_chi_squared_mean() {
        let chi_sq = ChiSquared::new(THREE_VALUE);
        assert_eq!(chi_sq.mean(), THREE_VALUE); // Mean = k
    }

    #[test]
    fn test_chi_squared_variance() {
        let chi_sq = ChiSquared::new(THREE_VALUE);
        assert_eq!(chi_sq.variance(), TWO_VALUE * THREE_VALUE); // Variance = 2k
    }

    #[test]
    fn test_chi_squared_shape_and_rate() {
        let chi_sq = ChiSquared::new(THREE_VALUE);
        assert_eq!(chi_sq.shape(), THREE_VALUE / TWO_VALUE); // k/2
        assert_eq!(chi_sq.rate(), HALF_VALUE); // 1/2
    }

    #[test]
    fn test_chi_squared_support() {
        let chi_sq = ChiSquared::new(THREE_VALUE);

        // Positive values should be in support
        assert!(chi_sq.in_support(SMALL_POSITIVE));
        assert!(chi_sq.in_support(UNIT_VALUE));
        assert!(chi_sq.in_support(LARGE_VALUE));

        // Zero and negative values should not be in support
        assert!(!chi_sq.in_support(ZERO_VALUE));
        assert!(!chi_sq.in_support(-SMALL_POSITIVE));
    }

    proptest! {
        #[test]
        fn prop_chi_squared_mean_equals_dof(dof in SMALL_POSITIVE..TEN_VALUE) {
            let chi_sq = ChiSquared::new(dof);
            assert!((chi_sq.mean() - dof).abs() < SMALL_POSITIVE);
        }

        #[test]
        fn prop_chi_squared_variance_twice_dof(dof in SMALL_POSITIVE..TEN_VALUE) {
            let chi_sq = ChiSquared::new(dof);
            assert!((chi_sq.variance() - TWO_VALUE * dof).abs() < SMALL_POSITIVE);
        }
    }
}

/// Tests for Exponential distribution
mod exponential_tests {
    use super::*;

    #[test]
    fn test_exponential_creation() {
        let exp = Exponential::new(TWO_VALUE);
        assert_eq!(exp.rate, TWO_VALUE);
    }

    #[test]
    fn test_exponential_default() {
        let exp: Exponential<f64> = Exponential::default();
        assert_eq!(exp.rate, UNIT_VALUE);
    }

    #[test]
    fn test_exponential_mean() {
        let exp = Exponential::new(TWO_VALUE);
        assert_eq!(exp.mean(), HALF_VALUE); // Mean = 1/λ = 1/2
    }

    #[test]
    fn test_exponential_variance() {
        let exp = Exponential::new(TWO_VALUE);
        let expected_variance = UNIT_VALUE / (TWO_VALUE * TWO_VALUE); // 1/λ² = 1/4
        assert!((exp.variance() - expected_variance).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_exponential_support() {
        let exp = Exponential::new(TWO_VALUE);

        // Non-negative values should be in support
        assert!(exp.in_support(ZERO_VALUE));
        assert!(exp.in_support(SMALL_POSITIVE));
        assert!(exp.in_support(LARGE_VALUE));

        // Negative values should not be in support
        assert!(!exp.in_support(-SMALL_POSITIVE));
    }

    proptest! {
        #[test]
        fn prop_exponential_mean_inverse_rate(rate in SMALL_POSITIVE..TEN_VALUE) {
            let exp = Exponential::new(rate);
            let expected_mean = UNIT_VALUE / rate;
            assert!((exp.mean() - expected_mean).abs() < SMALL_POSITIVE);
        }

        #[test]
        fn prop_exponential_variance_inverse_rate_squared(rate in SMALL_POSITIVE..TEN_VALUE) {
            let exp = Exponential::new(rate);
            let expected_variance = UNIT_VALUE / (rate * rate);
            assert!((exp.variance() - expected_variance).abs() < SMALL_POSITIVE);
        }
    }
}

/// Tests for Gamma distribution
mod gamma_tests {
    use super::*;

    #[test]
    fn test_gamma_creation() {
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);
        assert_eq!(gamma.shape, TWO_VALUE);
        assert_eq!(gamma.rate, THREE_VALUE);
    }

    #[test]
    fn test_gamma_from_shape_scale() {
        let shape = TWO_VALUE;
        let scale = HALF_VALUE;
        let gamma = Gamma::from_shape_scale(shape, scale);
        assert_eq!(gamma.shape, shape);
        assert_eq!(gamma.scale(), scale);
        assert_eq!(gamma.rate, UNIT_VALUE / scale); // rate = 1/scale
    }

    #[test]
    fn test_gamma_default() {
        let gamma: Gamma<f64> = Gamma::default();
        assert_eq!(gamma.shape, UNIT_VALUE);
        assert_eq!(gamma.rate, UNIT_VALUE);
    }

    #[test]
    fn test_gamma_mean() {
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);
        let expected_mean = TWO_VALUE / THREE_VALUE; // k/θ = 2/3
        assert!((gamma.mean() - expected_mean).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_gamma_variance() {
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);
        let expected_variance = TWO_VALUE / (THREE_VALUE * THREE_VALUE); // k/θ² = 2/9
        assert!((gamma.variance() - expected_variance).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_gamma_support() {
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);

        // Positive values should be in support
        assert!(gamma.in_support(SMALL_POSITIVE));
        assert!(gamma.in_support(UNIT_VALUE));
        assert!(gamma.in_support(LARGE_VALUE));

        // Zero and negative values should not be in support
        assert!(!gamma.in_support(ZERO_VALUE));
        assert!(!gamma.in_support(-SMALL_POSITIVE));
    }

    proptest! {
        #[test]
        fn prop_gamma_mean_shape_over_rate(
            shape in SMALL_POSITIVE..TEN_VALUE,
            rate in SMALL_POSITIVE..TEN_VALUE
        ) {
            let gamma = Gamma::new(shape, rate);
            let expected_mean = shape / rate;
            assert!((gamma.mean() - expected_mean).abs() < SMALL_POSITIVE);
        }

        #[test]
        fn prop_gamma_variance_shape_over_rate_squared(
            shape in SMALL_POSITIVE..TEN_VALUE,
            rate in SMALL_POSITIVE..TEN_VALUE
        ) {
            let gamma = Gamma::new(shape, rate);
            let expected_variance = shape / (rate * rate);
            assert!((gamma.variance() - expected_variance).abs() < SMALL_POSITIVE);
        }

        #[test]
        fn prop_gamma_exponential_special_case(rate in SMALL_POSITIVE..TEN_VALUE) {
            // Gamma(1, λ) should be equivalent to Exponential(λ)
            let gamma = Gamma::new(UNIT_VALUE, rate);
            let exponential = Exponential::new(rate);

            assert!((gamma.mean() - exponential.mean()).abs() < SMALL_POSITIVE);
            assert!((gamma.variance() - exponential.variance()).abs() < SMALL_POSITIVE);
        }
    }
}

/// Cross-distribution tests for semantic invariants
mod cross_distribution_tests {
    use super::*;

    #[test]
    fn test_all_distributions_positive_variance() {
        let beta = Beta::new(TWO_VALUE, THREE_VALUE);
        let chi_sq = ChiSquared::new(THREE_VALUE);
        let exp = Exponential::new(TWO_VALUE);
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);

        assert!(beta.variance() > ZERO_VALUE);
        assert!(chi_sq.variance() > ZERO_VALUE);
        assert!(exp.variance() > ZERO_VALUE);
        assert!(gamma.variance() > ZERO_VALUE);
    }

    #[test]
    fn test_support_consistency() {
        let chi_sq = ChiSquared::new(THREE_VALUE);
        let exp = Exponential::new(TWO_VALUE);
        let gamma = Gamma::new(TWO_VALUE, THREE_VALUE);

        // All should support positive values
        assert!(chi_sq.in_support(UNIT_VALUE));
        assert!(exp.in_support(UNIT_VALUE));
        assert!(gamma.in_support(UNIT_VALUE));

        // None should support negative values
        assert!(!chi_sq.in_support(-UNIT_VALUE));
        assert!(!exp.in_support(-UNIT_VALUE));
        assert!(!gamma.in_support(-UNIT_VALUE));
    }

    proptest! {
        #[test]
        fn prop_exponential_family_natural_params_well_defined(
            alpha in SMALL_POSITIVE..TEN_VALUE,
            beta_param in SMALL_POSITIVE..TEN_VALUE
        ) {
            let beta = Beta::new(alpha, beta_param);
            let (natural_params, log_partition) = beta.natural_and_log_partition();

            // Natural parameters should be finite
            assert!(natural_params[0].is_finite());
            assert!(natural_params[1].is_finite());

            // Log partition should be finite
            assert!(log_partition.is_finite());
        }
    }
}

/// Test edge cases and boundary conditions
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_beta_extreme_parameters() {
        // Very small parameters
        let beta_small = Beta::new(SMALL_POSITIVE, SMALL_POSITIVE);
        assert!(beta_small.mean().is_finite());
        assert!(beta_small.variance().is_finite());

        // Very large parameters
        let beta_large = Beta::new(LARGE_VALUE, LARGE_VALUE);
        assert!(beta_large.mean().is_finite());
        assert!(beta_large.variance().is_finite());

        // Asymmetric parameters
        let beta_asym = Beta::new(SMALL_POSITIVE, LARGE_VALUE);
        assert!(beta_asym.mean() < HALF_VALUE); // Should be skewed towards 0
    }

    #[test]
    fn test_gamma_chi_squared_relationship() {
        // Chi-squared(k) should be equivalent to Gamma(k/2, 1/2)
        let dof = THREE_VALUE;
        let chi_sq = ChiSquared::new(dof);
        let gamma_equiv = Gamma::new(dof / TWO_VALUE, HALF_VALUE);

        assert!((chi_sq.mean() - gamma_equiv.mean()).abs() < SMALL_POSITIVE);
        assert!((chi_sq.variance() - gamma_equiv.variance()).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_distribution_scaling_properties() {
        // Test that scaling rate parameters affects mean and variance correctly
        let base_rate = UNIT_VALUE;
        let scaled_rate = TWO_VALUE;

        let exp_base = Exponential::new(base_rate);
        let exp_scaled = Exponential::new(scaled_rate);

        // Mean should be inversely proportional to rate
        assert!(
            (exp_scaled.mean() * scaled_rate - exp_base.mean() * base_rate).abs() < SMALL_POSITIVE
        );

        // Variance should be inversely proportional to rate squared
        let variance_ratio = exp_base.variance() / exp_scaled.variance();
        let expected_ratio = (scaled_rate / base_rate).powi(2);
        assert!((variance_ratio - expected_ratio).abs() < SMALL_POSITIVE);
    }
}
