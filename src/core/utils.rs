//! Utility functions for safe numeric conversions and common operations.

use num_traits::Float;

/// Safe conversion from one numeric type to another.
///
/// This provides a safer alternative to `.unwrap()` calls throughout the codebase.
/// Returns a default value if conversion fails.
pub fn safe_convert<T, U>(value: T) -> U
where
    T: num_traits::ToPrimitive,
    U: num_traits::FromPrimitive + Default,
{
    if let Some(f64_val) = value.to_f64() {
        U::from_f64(f64_val).unwrap_or_default()
    } else {
        U::default()
    }
}

/// Safe conversion with explicit fallback value.
pub fn safe_convert_or<T, U>(value: T, fallback: U) -> U
where
    T: num_traits::ToPrimitive,
    U: num_traits::FromPrimitive + Copy,
{
    if let Some(f64_val) = value.to_f64() {
        U::from_f64(f64_val).unwrap_or(fallback)
    } else {
        fallback
    }
}

/// Safe conversion for Float types with better error handling.
pub fn safe_float_convert<T, U>(value: T) -> Result<U, &'static str>
where
    T: Float,
    U: Float,
{
    if let Some(f64_val) = value.to_f64() {
        if let Some(result) = U::from(f64_val) {
            Ok(result)
        } else {
            Err("Failed to convert to target float type")
        }
    } else {
        Err("Failed to convert to f64 intermediate")
    }
}

/// Create a constant of the target Float type from a literal.
/// This is safer than `T::from(literal).unwrap()`.
#[must_use]
pub fn float_constant<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(|| {
        // If conversion fails, try to provide a reasonable fallback
        if value == 0.0 {
            T::zero()
        } else if value == 1.0 {
            T::one()
        } else if value.is_infinite() {
            if value > 0.0 {
                T::infinity()
            } else {
                T::neg_infinity()
            }
        } else if value.is_nan() {
            T::nan()
        } else {
            // Last resort: return zero
            T::zero()
        }
    })
}
