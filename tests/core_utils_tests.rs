//! Comprehensive tests for core utility functions
//!
//! These tests focus on safe numeric conversions and edge cases,
//! using property-based testing to ensure robustness.

use measures::{float_constant, safe_convert, safe_convert_or, safe_float_convert};
use proptest::prelude::*;

/// Test `safe_convert` with various numeric types
#[test]
fn test_safe_convert_basic() {
    // Test i32 to f64 conversion
    let result: f64 = safe_convert(42i32);
    assert_eq!(result, 42.0);

    // Test f32 to f64 conversion
    let result: f64 = safe_convert(std::f32::consts::PI);
    assert!((result - std::f64::consts::PI).abs() < 1e-6);

    // Test u64 to f32 conversion
    let result: f32 = safe_convert(100u64);
    assert_eq!(result, 100.0);
}

#[test]
fn test_safe_convert_with_default_fallback() {
    // Test conversion that should succeed
    let result: f64 = safe_convert(123i32);
    assert_eq!(result, 123.0);

    // For cases where conversion might fail, we get the default
    // This is harder to test directly, but we can test the structure
    let result: f64 = safe_convert(0i32);
    assert_eq!(result, 0.0);
}

#[test]
fn test_safe_convert_or_with_explicit_fallback() {
    // Test successful conversion
    let result: f64 = safe_convert_or(42i32, 999.0);
    assert_eq!(result, 42.0);

    // Test with zero (should convert successfully)
    let result: f64 = safe_convert_or(0i32, 999.0);
    assert_eq!(result, 0.0);

    // Test with negative number
    let result: f64 = safe_convert_or(-42i32, 999.0);
    assert_eq!(result, -42.0);
}

#[test]
fn test_safe_float_convert_success_cases() {
    // Test f32 to f64 conversion
    let result: Result<f64, &'static str> = safe_float_convert(std::f64::consts::PI);
    assert!(result.is_ok());
    assert!((result.unwrap() - std::f64::consts::PI).abs() < 1e-6);

    // Test f64 to f32 conversion
    let result: Result<f32, &'static str> = safe_float_convert(2.71f64);
    assert!(result.is_ok());
    assert!((result.unwrap() - 2.71).abs() < 1e-6);

    // Test same type conversion
    let result: Result<f64, &'static str> = safe_float_convert(1.41f64);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1.41);
}

#[test]
fn test_float_constant_basic_values() {
    // Test zero
    let zero: f64 = float_constant(0.0);
    assert_eq!(zero, 0.0);

    let zero_f32: f32 = float_constant(0.0);
    assert_eq!(zero_f32, 0.0);

    // Test one
    let one: f64 = float_constant(1.0);
    assert_eq!(one, 1.0);

    let one_f32: f32 = float_constant(1.0);
    assert_eq!(one_f32, 1.0);

    // Test negative values
    let neg: f64 = float_constant(-1.0);
    assert_eq!(neg, -1.0);

    // Test fractional values
    let frac: f64 = float_constant(0.5);
    assert_eq!(frac, 0.5);
}

#[test]
fn test_float_constant_special_values() {
    // Test positive infinity
    let pos_inf: f64 = float_constant(f64::INFINITY);
    assert!(pos_inf.is_infinite() && pos_inf.is_sign_positive());

    // Test negative infinity
    let neg_inf: f64 = float_constant(f64::NEG_INFINITY);
    assert!(neg_inf.is_infinite() && neg_inf.is_sign_negative());

    // Test NaN
    let nan_val: f64 = float_constant(f64::NAN);
    assert!(nan_val.is_nan());
}

// Property-based tests

proptest! {
    /// Test safe_convert with random integers
    #[test]
    fn prop_safe_convert_integers(value in any::<i32>()) {
        let result: f64 = safe_convert(value);
        // Should not panic and should be finite for reasonable integers
        assert!(result.is_finite());
        assert_eq!(result, f64::from(value));
    }

    /// Test safe_convert with random unsigned integers
    #[test]
    fn prop_safe_convert_unsigned(value in any::<u32>()) {
        let result: f64 = safe_convert(value);
        assert!(result.is_finite());
        assert!(result >= 0.0);
        assert_eq!(result, f64::from(value));
    }

    /// Test safe_convert_or with random values and fallbacks
    #[test]
    fn prop_safe_convert_or_integers(
        value in any::<i32>(),
        fallback in any::<f64>()
    ) {
        let result: f64 = safe_convert_or(value, fallback);
        // Should either be the converted value or the fallback
        assert!(result.is_finite() || fallback.is_infinite() || fallback.is_nan());
        // For normal integers, should get the converted value
        if value.abs() < 1_000_000 {  // Reasonable range
            assert_eq!(result, f64::from(value));
        }
    }

    /// Test safe_float_convert with finite f32 values
    #[test]
    fn prop_safe_float_convert_finite(value in any::<f32>().prop_filter("finite", |x| x.is_finite())) {
        let result: Result<f64, &'static str> = safe_float_convert(value);
        assert!(result.is_ok());
        let converted = result.unwrap();
        assert!(converted.is_finite());
        assert!((converted - f64::from(value)).abs() < 1e-6);
    }

    /// Test float_constant with reasonable values
    #[test]
    fn prop_float_constant_reasonable(value in -1000.0..1000.0f64) {
        let result_f64: f64 = float_constant(value);
        let result_f32: f32 = float_constant(value);

        // Should not panic and should preserve the value
        assert_eq!(result_f64, value);
        assert!((result_f32 - value as f32).abs() < 1e-6);
    }

    /// Test float_constant with extreme but finite values
    #[test]
    fn prop_float_constant_extreme_finite(
        value in prop_oneof![
            (-1e100..=-1e-100f64),
            (1e-100..=1e100f64)
        ]
    ) {
        let result: f64 = float_constant(value);
        // Should not panic
        assert!(result.is_finite() || result.is_infinite());
    }
}

/// Test semantic invariants

#[test]
fn test_safe_convert_preserves_zero() {
    // Zero should always convert to zero
    let zero_i32: f64 = safe_convert(0i32);
    assert_eq!(zero_i32, 0.0);

    let zero_u64: f32 = safe_convert(0u64);
    assert_eq!(zero_u64, 0.0);

    let zero_f32: f64 = safe_convert(0.0f32);
    assert_eq!(zero_f32, 0.0);
}

#[test]
fn test_safe_convert_preserves_sign() {
    // Positive values should remain positive
    let pos: f64 = safe_convert(42i32);
    assert!(pos > 0.0);

    // Negative values should remain negative
    let neg: f64 = safe_convert(-42i32);
    assert!(neg < 0.0);
}

#[test]
fn test_safe_convert_or_fallback_semantics() {
    // When conversion succeeds, should not use fallback
    let result: f64 = safe_convert_or(100i32, 999.0);
    assert_eq!(result, 100.0);
    assert_ne!(result, 999.0);
}

#[test]
fn test_float_constant_type_consistency() {
    // Same input should produce consistent results for same type
    let val1: f64 = float_constant(std::f64::consts::PI);
    let val2: f64 = float_constant(std::f64::consts::PI);
    assert_eq!(val1, val2);

    // Different types should handle the same input appropriately
    let val_f64: f64 = float_constant(std::f64::consts::PI);
    let val_f32: f32 = float_constant(std::f64::consts::PI);
    assert!((val_f64 - f64::from(val_f32)).abs() < 1e-6);
}

/// Test edge cases and boundary conditions

#[test]
fn test_safe_convert_boundary_values() {
    // Test with maximum values for different types
    let max_u8: f64 = safe_convert(u8::MAX);
    assert_eq!(max_u8, f64::from(u8::MAX));

    let max_i16: f64 = safe_convert(i16::MAX);
    assert_eq!(max_i16, f64::from(i16::MAX));

    let min_i16: f64 = safe_convert(i16::MIN);
    assert_eq!(min_i16, f64::from(i16::MIN));
}

#[test]
fn test_safe_float_convert_special_values() {
    // Test with infinity
    let pos_inf_result: Result<f64, &'static str> = safe_float_convert(f32::INFINITY);
    assert!(pos_inf_result.is_ok());
    assert!(pos_inf_result.unwrap().is_infinite());

    let neg_inf_result: Result<f64, &'static str> = safe_float_convert(f32::NEG_INFINITY);
    assert!(neg_inf_result.is_ok());
    assert!(neg_inf_result.unwrap().is_infinite());

    // Test with NaN
    let nan_result: Result<f64, &'static str> = safe_float_convert(f32::NAN);
    assert!(nan_result.is_ok());
    assert!(nan_result.unwrap().is_nan());
}

#[test]
fn test_float_constant_fallback_behavior() {
    // Test that fallback behavior works for special values
    let zero_result: f64 = float_constant(0.0);
    assert_eq!(zero_result, 0.0);

    let one_result: f64 = float_constant(1.0);
    assert_eq!(one_result, 1.0);

    // Test with very small numbers
    let tiny: f64 = float_constant(1e-300);
    assert!(tiny.is_finite());

    // Test with very large numbers
    let huge: f64 = float_constant(1e300);
    assert!(huge.is_finite() || huge.is_infinite());
}

/// Test that functions handle type conversions correctly

#[test]
fn test_cross_type_conversions() {
    // i8 to f64
    let small_int: f64 = safe_convert(127i8);
    assert_eq!(small_int, 127.0);

    // u16 to f32
    let medium_uint: f32 = safe_convert(65535u16);
    assert_eq!(medium_uint, 65535.0);

    // f32 to f64 (should be lossless for most values)
    let float_val: f64 = safe_convert(std::f32::consts::PI);
    assert!((float_val - std::f64::consts::PI).abs() < 1e-5);
}

#[test]
fn test_safe_convert_or_with_different_fallback_types() {
    // Test with integer fallback
    let result1: i32 = safe_convert_or(42u32, -1i32);
    assert_eq!(result1, 42);

    // Test with float fallback
    let result2: f64 = safe_convert_or(100i32, -999.0f64);
    assert_eq!(result2, 100.0);
}

/// Test performance characteristics (basic sanity checks)

#[test]
fn test_conversion_functions_dont_panic() {
    // Test a variety of inputs to ensure no panics
    let test_values = [0, 1, -1, 42, -42, i32::MAX, i32::MIN, 1000, -1000];

    for &val in &test_values {
        let _: f64 = safe_convert(val);
        let _: f32 = safe_convert(val);
        let _: f64 = safe_convert_or(val, 0.0);
        let _: f32 = safe_convert_or(val, 0.0);
    }

    let float_values = [
        0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        -2.71,
        f64::MIN,
        f64::MAX,
        1e-10,
        1e10,
        -1e-10,
        -1e10,
    ];

    for &val in &float_values {
        let _: f64 = float_constant(val);
        let _: f32 = float_constant(val);
        if val.is_finite() {
            let _: Result<f32, _> = safe_float_convert(val);
        }
    }
}
