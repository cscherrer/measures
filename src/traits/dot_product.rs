//! Dot product trait and implementations.
//!
//! This module provides a trait for dot product operations between types.
//!
//! # Note for Distribution Implementers
//!
//! For exponential family distributions with scalar natural parameters,
//! use `[T; 1]` as the natural parameter type instead of just `T`.
//! This ensures compatibility with the `DotProduct` trait without needing
//! additional implementations for scalar types.

use num_traits::Float;

/// Extension trait for dot product operations
pub trait DotProduct<Rhs = Self> {
    type Output;

    fn dot(&self, other: &Rhs) -> Self::Output;
}

// Implementation for fixed-size arrays
impl<T: Float, const N: usize> DotProduct for [T; N] {
    type Output = T;

    fn dot(&self, other: &Self) -> Self::Output {
        let mut result = T::zero();
        for (a, b) in self.iter().zip(other.iter()) {
            result = result + (*a) * (*b);
        }
        result
    }
}

// Implementation for Vec<T> (for distributions like Categorical)
impl<T: Float> DotProduct for Vec<T> {
    type Output = T;

    fn dot(&self, other: &Self) -> Self::Output {
        let mut result = T::zero();
        for (a, b) in self.iter().zip(other.iter()) {
            result = result + (*a) * (*b);
        }
        result
    }
}

/// Helper function for element-wise array subtraction
///
/// This is needed for exponential family relative density computation where
/// we compute (η₁ - η₂)·T(x) - (A(η₁) - A(η₂)).
pub fn array_sub<T: Float, const N: usize>(a: [T; N], b: [T; N]) -> [T; N] {
    let mut result = [T::zero(); N];
    for i in 0..N {
        result[i] = a[i] - b[i];
    }
    result
}
