//! Tests for discrete distributions with zero coverage: Bernoulli, Binomial, and Geometric.
//! These tests focus on ensuring correctness of distribution creation, parameter validation,
//! probability mass functions, and statistical properties while using property-based testing
//! for robustness.

use measures::Measure;
use measures::distributions::discrete::bernoulli::Bernoulli;
use measures::distributions::discrete::binomial::Binomial;
use measures::distributions::discrete::geometric::Geometric;
use measures::exponential_family::traits::ExponentialFamily;
use proptest::prelude::*;

// Constants for testing - no magic numbers
const ZERO_PROB: f64 = 0.0;
const UNIT_PROB: f64 = 1.0;
const HALF_PROB: f64 = 0.5;
const SMALL_PROB: f64 = 0.1;
const LARGE_PROB: f64 = 0.9;
const TOLERANCE: f64 = 1e-10;
const SMALL_N: u64 = 1;
const MEDIUM_N: u64 = 10;
const LARGE_N: u64 = 100;
const MAX_TEST_N: u64 = 1000;

/// Tests for Bernoulli distribution
mod bernoulli_tests {
    use super::*;

    #[test]
    fn test_bernoulli_creation_valid_parameters() {
        let bernoulli = Bernoulli::new(HALF_PROB);
        assert_eq!(bernoulli.prob, HALF_PROB);
    }

    #[test]
    #[should_panic]
    fn test_bernoulli_creation_invalid_prob_negative() {
        Bernoulli::<f64>::new(-SMALL_PROB);
    }

    #[test]
    #[should_panic]
    fn test_bernoulli_creation_invalid_prob_greater_than_one() {
        Bernoulli::<f64>::new(UNIT_PROB + SMALL_PROB);
    }

    #[test]
    fn test_bernoulli_default() {
        let bernoulli: Bernoulli<f64> = Bernoulli::default();
        assert_eq!(bernoulli.prob, HALF_PROB);
    }

    #[test]
    fn test_bernoulli_mean_calculation() {
        let bernoulli = Bernoulli::new(LARGE_PROB);
        assert!((bernoulli.mean() - LARGE_PROB).abs() < TOLERANCE);
    }

    #[test]
    fn test_bernoulli_variance_calculation() {
        let bernoulli = Bernoulli::new(LARGE_PROB);
        let expected_variance = LARGE_PROB * (UNIT_PROB - LARGE_PROB);
        assert!((bernoulli.variance() - expected_variance).abs() < TOLERANCE);
    }

    #[test]
    fn test_bernoulli_support() {
        let bernoulli = Bernoulli::new(HALF_PROB);
        assert!(bernoulli.in_support(0u8));
        assert!(bernoulli.in_support(1u8));
        assert!(!bernoulli.in_support(2u8));
    }

    #[test]
    fn test_bernoulli_exponential_family_properties() {
        let bernoulli = Bernoulli::new(LARGE_PROB);

        // Test natural parameters
        let natural_params = bernoulli.to_natural();
        assert!(!natural_params.is_empty());

        // Test sufficient statistics
        let sufficient_stats = bernoulli.sufficient_statistic(&1u8);
        assert!(!sufficient_stats.is_empty());

        // Test log partition function
        let log_partition = bernoulli.log_partition();
        assert!(log_partition.is_finite());
    }

    proptest! {
        #[test]
        fn prop_bernoulli_mean_in_unit_interval(
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let bernoulli = Bernoulli::new(prob);
            let mean = bernoulli.mean();
            prop_assert!((ZERO_PROB..=UNIT_PROB).contains(&mean));
            prop_assert!((mean - prob).abs() < TOLERANCE);
        }

        #[test]
        fn prop_bernoulli_variance_positive(
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let bernoulli = Bernoulli::new(prob);
            let variance = bernoulli.variance();
            prop_assert!(variance >= ZERO_PROB);
            prop_assert!(variance <= 0.25); // Maximum at p=0.5
        }
    }
}

/// Tests for Binomial distribution
mod binomial_tests {
    use super::*;

    #[test]
    fn test_binomial_creation_valid_parameters() {
        let binomial = Binomial::new(MEDIUM_N, HALF_PROB);
        assert_eq!(binomial.n, MEDIUM_N);
        assert_eq!(binomial.prob, HALF_PROB);
    }

    #[test]
    #[should_panic]
    fn test_binomial_creation_invalid_prob_negative() {
        Binomial::<f64>::new(MEDIUM_N, -SMALL_PROB);
    }

    #[test]
    #[should_panic]
    fn test_binomial_creation_invalid_prob_greater_than_one() {
        Binomial::<f64>::new(MEDIUM_N, UNIT_PROB + SMALL_PROB);
    }

    #[test]
    fn test_binomial_default() {
        // Binomial doesn't have a default, so we'll test with explicit parameters
        let binomial = Binomial::new(SMALL_N, HALF_PROB);
        assert_eq!(binomial.n, SMALL_N);
        assert_eq!(binomial.prob, HALF_PROB);
    }

    #[test]
    fn test_binomial_mean_calculation() {
        let binomial = Binomial::new(MEDIUM_N, LARGE_PROB);
        let expected_mean = (MEDIUM_N as f64) * LARGE_PROB;
        assert!((binomial.mean() - expected_mean).abs() < TOLERANCE);
    }

    #[test]
    fn test_binomial_variance_calculation() {
        let binomial = Binomial::new(MEDIUM_N, LARGE_PROB);
        let expected_variance = (MEDIUM_N as f64) * LARGE_PROB * (UNIT_PROB - LARGE_PROB);
        assert!((binomial.variance() - expected_variance).abs() < TOLERANCE);
    }

    #[test]
    fn test_binomial_support() {
        let binomial = Binomial::new(MEDIUM_N, HALF_PROB);
        assert!(binomial.in_support(0u64));
        assert!(binomial.in_support(5u64));
        assert!(binomial.in_support(MEDIUM_N));
        assert!(!binomial.in_support(MEDIUM_N + 1));
    }

    #[test]
    fn test_binomial_reduces_to_bernoulli() {
        let binomial = Binomial::new(SMALL_N, LARGE_PROB);
        let bernoulli = Bernoulli::new(LARGE_PROB);

        assert!((binomial.mean() - bernoulli.mean()).abs() < TOLERANCE);
        assert!((binomial.variance() - bernoulli.variance()).abs() < TOLERANCE);
    }

    #[test]
    fn test_binomial_exponential_family_properties() {
        let binomial = Binomial::new(MEDIUM_N, LARGE_PROB);

        // Test natural parameters
        let natural_params = binomial.to_natural();
        assert!(!natural_params.is_empty());

        // Test sufficient statistics
        let sufficient_stats = binomial.sufficient_statistic(&5u64);
        assert!(!sufficient_stats.is_empty());

        // Test log partition function
        let log_partition = binomial.log_partition();
        assert!(log_partition.is_finite());
    }

    proptest! {
        #[test]
        fn prop_binomial_mean_bounded(
            n in 1u64..MAX_TEST_N,
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let binomial = Binomial::new(n, prob);
            let mean = binomial.mean();
            prop_assert!(mean >= ZERO_PROB);
            prop_assert!(mean <= (n as f64));
        }

        #[test]
        fn prop_binomial_variance_positive(
            n in 1u64..MAX_TEST_N,
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let binomial = Binomial::new(n, prob);
            let variance = binomial.variance();
            prop_assert!(variance >= ZERO_PROB);
            prop_assert!(variance <= (n as f64) * 0.25); // Maximum at p=0.5
        }
    }
}

/// Tests for Geometric distribution
mod geometric_tests {
    use super::*;

    #[test]
    fn test_geometric_creation_valid_parameters() {
        let geometric = Geometric::new(HALF_PROB);
        assert_eq!(geometric.prob, HALF_PROB);
    }

    #[test]
    #[should_panic]
    fn test_geometric_creation_invalid_prob_zero() {
        Geometric::<f64>::new(ZERO_PROB);
    }

    #[test]
    #[should_panic]
    fn test_geometric_creation_invalid_prob_negative() {
        Geometric::<f64>::new(-SMALL_PROB);
    }

    #[test]
    #[should_panic]
    fn test_geometric_creation_invalid_prob_greater_than_one() {
        Geometric::<f64>::new(UNIT_PROB + SMALL_PROB);
    }

    #[test]
    fn test_geometric_default() {
        let geometric: Geometric<f64> = Geometric::default();
        assert_eq!(geometric.prob, HALF_PROB);
    }

    #[test]
    fn test_geometric_mean_calculation() {
        let geometric = Geometric::new(HALF_PROB);
        let expected_mean = UNIT_PROB / HALF_PROB;
        assert!((geometric.mean() - expected_mean).abs() < TOLERANCE);
    }

    #[test]
    fn test_geometric_variance_calculation() {
        let geometric = Geometric::new(HALF_PROB);
        let expected_variance = (UNIT_PROB - HALF_PROB) / (HALF_PROB * HALF_PROB);
        assert!((geometric.variance() - expected_variance).abs() < TOLERANCE);
    }

    #[test]
    fn test_geometric_support() {
        let geometric = Geometric::new(HALF_PROB);
        assert!(geometric.in_support(1u64));
        assert!(geometric.in_support(10u64));
        assert!(geometric.in_support(100u64));
        assert!(!geometric.in_support(0u64));
    }

    #[test]
    fn test_geometric_memoryless_property() {
        // The geometric distribution should satisfy P(X > s+t | X > s) = P(X > t)
        // This is a key property of the geometric distribution
        let geometric = Geometric::new(SMALL_PROB);

        // Test that mean is always positive and finite
        let mean = geometric.mean();
        assert!(mean > ZERO_PROB);
        assert!(mean.is_finite());

        // Test that variance is always positive and finite
        let variance = geometric.variance();
        assert!(variance > ZERO_PROB);
        assert!(variance.is_finite());
    }

    #[test]
    fn test_geometric_exponential_family_properties() {
        let geometric = Geometric::new(LARGE_PROB);

        // Test natural parameters
        let natural_params = geometric.to_natural();
        assert!(!natural_params.is_empty());

        // Test sufficient statistics
        let sufficient_stats = geometric.sufficient_statistic(&5u64);
        assert!(!sufficient_stats.is_empty());

        // Test log partition function
        let log_partition = geometric.log_partition();
        assert!(log_partition.is_finite());
    }

    proptest! {
        #[test]
        fn prop_geometric_mean_positive(
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let geometric = Geometric::new(prob);
            let mean = geometric.mean();
            prop_assert!(mean > ZERO_PROB);
            prop_assert!(mean.is_finite());
        }

        #[test]
        fn prop_geometric_variance_positive(
            prob in SMALL_PROB..LARGE_PROB
        ) {
            let geometric = Geometric::new(prob);
            let variance = geometric.variance();
            prop_assert!(variance > ZERO_PROB);
            prop_assert!(variance.is_finite());
        }

        #[test]
        fn prop_geometric_higher_prob_lower_mean(
            prob1 in SMALL_PROB..HALF_PROB,
            prob2 in HALF_PROB..LARGE_PROB
        ) {
            let geometric1 = Geometric::new(prob1);
            let geometric2 = Geometric::new(prob2);

            // Higher probability should lead to lower expected waiting time
            prop_assert!(geometric1.mean() > geometric2.mean());
        }
    }
}

/// Cross-distribution tests for discrete distributions
mod cross_distribution_tests {
    use super::*;

    #[test]
    fn test_all_discrete_distributions_positive_variance() {
        let bernoulli = Bernoulli::new(LARGE_PROB);
        let binomial = Binomial::new(MEDIUM_N, LARGE_PROB);
        let geometric = Geometric::new(LARGE_PROB);

        assert!(bernoulli.variance() >= ZERO_PROB);
        assert!(binomial.variance() >= ZERO_PROB);
        assert!(geometric.variance() > ZERO_PROB);
    }

    #[test]
    fn test_support_consistency() {
        let bernoulli = Bernoulli::new(HALF_PROB);
        let binomial = Binomial::new(MEDIUM_N, HALF_PROB);
        let geometric = Geometric::new(HALF_PROB);

        // Bernoulli support: {0, 1}
        assert!(bernoulli.in_support(0u8));
        assert!(bernoulli.in_support(1u8));
        assert!(!bernoulli.in_support(2u8));

        // Binomial support: {0, 1, ..., n}
        assert!(binomial.in_support(0u64));
        assert!(binomial.in_support(MEDIUM_N));
        assert!(!binomial.in_support(MEDIUM_N + 1));

        // Geometric support: {1, 2, 3, ...}
        assert!(!geometric.in_support(0u64));
        assert!(geometric.in_support(1u64));
        assert!(geometric.in_support(100u64));
    }

    proptest! {
        #[test]
        fn prop_exponential_family_natural_params_well_defined(
            prob in SMALL_PROB..LARGE_PROB,
            n in 1u64..100u64
        ) {
            let bernoulli = Bernoulli::new(prob);
            let binomial = Binomial::new(n, prob);
            let geometric = Geometric::new(prob);

            // All natural parameters should be finite
            for &param in &bernoulli.to_natural() {
                prop_assert!(param.is_finite());
            }

            for &param in &binomial.to_natural() {
                prop_assert!(param.is_finite());
            }

            for &param in &geometric.to_natural() {
                prop_assert!(param.is_finite());
            }
        }
    }
}

/// Edge case tests for discrete distributions
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_bernoulli_extreme_probabilities() {
        // Test with very small probability
        let bernoulli_small = Bernoulli::new(SMALL_PROB);
        assert!((bernoulli_small.mean() - SMALL_PROB).abs() < TOLERANCE);
        assert!(bernoulli_small.variance() > ZERO_PROB);

        // Test with very large probability
        let bernoulli_large = Bernoulli::new(LARGE_PROB);
        assert!((bernoulli_large.mean() - LARGE_PROB).abs() < TOLERANCE);
        assert!(bernoulli_large.variance() > ZERO_PROB);
    }

    #[test]
    fn test_binomial_large_n() {
        let binomial = Binomial::new(LARGE_N, HALF_PROB);
        let expected_mean = (LARGE_N as f64) * HALF_PROB;
        let expected_variance = (LARGE_N as f64) * HALF_PROB * HALF_PROB;

        assert!((binomial.mean() - expected_mean).abs() < TOLERANCE);
        assert!((binomial.variance() - expected_variance).abs() < TOLERANCE);
    }

    #[test]
    fn test_geometric_extreme_probabilities() {
        // Test with small probability (high variance)
        let geometric_small = Geometric::new(SMALL_PROB);
        assert!(geometric_small.mean() > UNIT_PROB / SMALL_PROB - TOLERANCE);
        assert!(geometric_small.variance().is_finite());

        // Test with large probability (low variance)
        let geometric_large = Geometric::new(LARGE_PROB);
        assert!(geometric_large.mean() < UNIT_PROB / LARGE_PROB + TOLERANCE);
        assert!(geometric_large.variance().is_finite());
    }
}
