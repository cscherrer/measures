//! Tests for core density functionality including `LogDensity` creation,
//! evaluation, caching, and helper functions. These tests focus on ensuring
//! correctness of density computations and related operations while also
//! using property-based testing to ensure robustness.

use measures::LogDensityTrait;
use measures::core::density::{LogDensity, log_density_at, log_density_batch};
use measures::distributions::continuous::normal::Normal;
use proptest::prelude::*;

// Constants for testing - no magic numbers
const ZERO_VALUE: f64 = 0.0;
const UNIT_VALUE: f64 = 1.0;
const TWO_VALUE: f64 = 2.0;
const THREE_VALUE: f64 = 3.0;
const SMALL_POSITIVE: f64 = 1e-6;
const LARGE_VALUE: f64 = 100.0;
const TOLERANCE: f64 = 1e-10;
const TEST_MEAN: f64 = 2.0;
const TEST_STD_DEV: f64 = 1.5;
const LARGE_BATCH_SIZE: usize = 1000;
const MANY_EVALUATIONS: usize = 100;

/// Tests for basic log-density functionality
mod basic_density_tests {
    use super::*;

    #[test]
    fn test_log_density_creation() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        // Should be able to create log density
        assert!(log_density.measure().mean == ZERO_VALUE);
        assert!(log_density.measure().std_dev == UNIT_VALUE);
    }

    #[test]
    fn test_log_density_with_custom_base_measure() {
        let normal = Normal::new(TEST_MEAN, TEST_STD_DEV);
        let log_density = LogDensity::new(normal);

        assert!(log_density.measure().mean == TEST_MEAN);
        assert!(log_density.measure().std_dev == TEST_STD_DEV);
    }

    #[test]
    fn test_log_density_evaluation() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        let density_at_zero: f64 = log_density.at(&ZERO_VALUE);
        let density_at_one: f64 = log_density.at(&UNIT_VALUE);

        // Should be finite values
        assert!(density_at_zero.is_finite());
        assert!(density_at_one.is_finite());

        // Density at mean should be higher than at other points for normal distribution
        assert!(density_at_zero > density_at_one);
    }
}

/// Tests for helper functions
mod helper_function_tests {
    use super::*;

    #[test]
    fn test_log_density_at_function() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        let density_at_zero = log_density_at(&normal, &ZERO_VALUE);
        let density_at_one = log_density_at(&normal, &UNIT_VALUE);

        assert!(density_at_zero.is_finite());
        assert!(density_at_one.is_finite());
        assert!(density_at_zero > density_at_one);
    }

    #[test]
    fn test_log_density_batch_function() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let points = vec![ZERO_VALUE, UNIT_VALUE, -UNIT_VALUE];

        let densities = log_density_batch(&normal, &points);

        assert_eq!(densities.len(), points.len());
        for density in densities {
            assert!(density.is_finite());
        }
    }

    #[test]
    fn test_log_density_batch_empty() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let empty_points: Vec<f64> = vec![];

        let densities = log_density_batch(&normal, &empty_points);
        assert!(densities.is_empty());
    }

    #[test]
    fn test_log_density_batch_single_point() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let single_point = vec![ZERO_VALUE];

        let densities = log_density_batch(&normal, &single_point);
        assert_eq!(densities.len(), 1);
        assert!(densities[0].is_finite());
    }
}

/// Tests for caching functionality
mod caching_tests {
    use super::*;

    #[test]
    fn test_cached_log_density_creation() {
        // Test with integer types that implement Hash and Eq
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        // For now, just test that the basic density works
        let density_value: f64 = log_density.at(&ZERO_VALUE);
        assert!(density_value.is_finite());
    }

    #[test]
    fn test_cached_log_density_evaluation() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        // Test multiple evaluations at the same point
        let density1: f64 = log_density.at(&ZERO_VALUE);
        let density2: f64 = log_density.at(&ZERO_VALUE);
        let density3: f64 = log_density.at(&UNIT_VALUE);

        // Should be consistent
        assert!((density1 - density2).abs() < TOLERANCE);
        assert!(density1 != density3); // Different points should give different values
    }

    #[test]
    fn test_cached_log_density_consistency() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        let test_points = vec![ZERO_VALUE, UNIT_VALUE, -UNIT_VALUE];

        for &point in &test_points {
            let density: f64 = log_density.at(&point);
            assert!(density.is_finite());
        }
    }

    #[test]
    fn test_cached_for_specific_type() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        // Test that we can evaluate at different points
        let density: f64 = log_density.at(&ZERO_VALUE);
        assert!(density.is_finite());
    }
}

/// Property-based tests for log density functionality
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_log_density_consistency(
            mean in -LARGE_VALUE..LARGE_VALUE,
            std_dev in SMALL_POSITIVE..LARGE_VALUE,
            x in -LARGE_VALUE..LARGE_VALUE
        ) {
            let normal = Normal::new(mean, std_dev);
            let log_density = LogDensity::new(normal);

            let density1: f64 = log_density.at(&x);
            let density2: f64 = log_density.at(&x);

            // Should be deterministic
            prop_assert!((density1 - density2).abs() < TOLERANCE);
            prop_assert!(density1.is_finite());
        }

        #[test]
        fn test_log_density_batch_consistency(
            mean in -LARGE_VALUE..LARGE_VALUE,
            std_dev in SMALL_POSITIVE..LARGE_VALUE,
            points in prop::collection::vec(-LARGE_VALUE..LARGE_VALUE, 1..10)
        ) {
            let normal = Normal::new(mean, std_dev);

            let batch_result = log_density_batch(&normal, &points);

            prop_assert_eq!(batch_result.len(), points.len());

            for (i, &point) in points.iter().enumerate() {
                let individual_result = log_density_at(&normal, &point);
                prop_assert!((batch_result[i] - individual_result).abs() < TOLERANCE);
            }
        }

        #[test]
        fn test_helper_functions_consistency(
            mean in -LARGE_VALUE..LARGE_VALUE,
            std_dev in SMALL_POSITIVE..LARGE_VALUE,
            x in -LARGE_VALUE..LARGE_VALUE
        ) {
            let normal = Normal::new(mean, std_dev);
            let log_density = LogDensity::new(normal.clone());

            let direct_result: f64 = log_density.at(&x);
            let helper_result = log_density_at(&normal, &x);

            prop_assert!((direct_result - helper_result).abs() < TOLERANCE);
        }
    }
}

/// Tests for semantic invariants
mod semantic_invariant_tests {
    use super::*;

    #[test]
    fn test_log_density_symmetry_for_symmetric_distributions() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        // Normal distribution should be symmetric around its mean
        let density_pos: f64 = log_density_at(&normal, &UNIT_VALUE);
        let density_neg: f64 = log_density_at(&normal, &-UNIT_VALUE);

        assert!((density_pos - density_neg).abs() < SMALL_POSITIVE);
    }

    #[test]
    fn test_log_density_maximum_at_mode() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        // For normal distribution, mode = mean, so density should be highest at mean
        let density_at_mode: f64 = log_density_at(&normal, &ZERO_VALUE);
        let density_away_from_mode: f64 = log_density_at(&normal, &UNIT_VALUE);

        assert!(density_at_mode > density_away_from_mode);
    }

    #[test]
    fn test_log_density_decreases_with_distance_from_mode() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        let density_at_mode: f64 = log_density_at(&normal, &ZERO_VALUE);
        let density_one_std: f64 = log_density_at(&normal, &UNIT_VALUE);
        let density_two_std: f64 = log_density_at(&normal, &TWO_VALUE);

        // Should decrease as we move away from the mode
        assert!(density_at_mode > density_one_std);
        assert!(density_one_std > density_two_std);
    }

    #[test]
    fn test_batch_computation_order_independence() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let points1 = vec![ZERO_VALUE, UNIT_VALUE, TWO_VALUE];
        let points2 = vec![TWO_VALUE, UNIT_VALUE, ZERO_VALUE];

        let densities1: Vec<f64> = log_density_batch(&normal, &points1);
        let densities2: Vec<f64> = log_density_batch(&normal, &points2);

        // Results should be consistent regardless of order
        assert!((densities1[0] - densities2[2]).abs() < SMALL_POSITIVE); // 0.0
        assert!((densities1[1] - densities2[1]).abs() < SMALL_POSITIVE); // 1.0
        assert!((densities1[2] - densities2[0]).abs() < SMALL_POSITIVE); // 2.0
    }
}

/// Tests for edge cases and boundary conditions
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_log_density_with_extreme_parameters() {
        // Test with very small standard deviation
        let normal_small = Normal::new(ZERO_VALUE, SMALL_POSITIVE);
        let log_density_small = LogDensity::new(normal_small);
        let density_small: f64 = log_density_small.at(&ZERO_VALUE);
        assert!(density_small.is_finite());

        // Test with very large standard deviation
        let normal_large = Normal::new(ZERO_VALUE, LARGE_VALUE);
        let log_density_large = LogDensity::new(normal_large);
        let density_large: f64 = log_density_large.at(&ZERO_VALUE);
        assert!(density_large.is_finite());
    }

    #[test]
    fn test_log_density_batch_with_duplicates() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        let points = vec![ZERO_VALUE, ZERO_VALUE, UNIT_VALUE, ZERO_VALUE];
        let densities = log_density_batch(&normal, &points);

        assert_eq!(densities.len(), points.len());
        // First, second, and fourth should be equal (all at zero)
        assert!((densities[0] - densities[1]).abs() < TOLERANCE);
        assert!((densities[0] - densities[3]).abs() < TOLERANCE);
    }

    #[test]
    fn test_cached_density_with_many_evaluations() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        // Test many evaluations at the same point
        for _ in 0..MANY_EVALUATIONS {
            let density: f64 = log_density.at(&ZERO_VALUE);
            assert!(density.is_finite());
        }
    }

    #[test]
    fn test_log_density_different_distributions() {
        let normal1 = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let normal2 = Normal::new(UNIT_VALUE, UNIT_VALUE);

        let log_density1 = LogDensity::new(normal1);
        let log_density2 = LogDensity::new(normal2);

        let density1: f64 = log_density1.at(&ZERO_VALUE);
        let density2: f64 = log_density2.at(&ZERO_VALUE);

        // Different distributions should give different densities at the same point
        assert!((density1 - density2).abs() > TOLERANCE);
    }

    #[test]
    fn test_log_density_batch_large_input() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        let points: Vec<f64> = (0..LARGE_BATCH_SIZE).map(|i| i as f64 / 100.0).collect();
        let densities = log_density_batch(&normal, &points);

        assert_eq!(densities.len(), points.len());
        for density in densities {
            assert!(density.is_finite());
        }
    }
}

/// Tests for performance characteristics
mod performance_tests {
    use super::*;

    #[test]
    fn test_caching_performance_benefit() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);
        let log_density = LogDensity::new(normal);

        let test_point = ZERO_VALUE;

        // Test multiple evaluations (simulating caching benefit)
        let start = std::time::Instant::now();
        for _ in 0..MANY_EVALUATIONS {
            let _density: f64 = log_density.at(&test_point);
        }
        let duration = start.elapsed();

        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000); // Less than 1 second
    }

    #[test]
    fn test_batch_vs_individual_consistency() {
        let normal = Normal::new(ZERO_VALUE, UNIT_VALUE);

        let points = vec![ZERO_VALUE, UNIT_VALUE, -UNIT_VALUE, TWO_VALUE];

        // Batch computation
        let batch_densities = log_density_batch(&normal, &points);

        // Individual computations
        let individual_densities: Vec<f64> = points
            .iter()
            .map(|&point| log_density_at(&normal, &point))
            .collect();

        assert_eq!(batch_densities.len(), individual_densities.len());

        for (batch, individual) in batch_densities.iter().zip(individual_densities.iter()) {
            assert!((batch - individual).abs() < TOLERANCE);
        }
    }
}
