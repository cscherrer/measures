//! Dot product trait and implementations.
//!
//! This module provides a trait for dot product operations between types.

use num_traits::Float;

/// Extension trait for dot product operations
pub trait DotProduct {
    type Output;

    fn dot(&self, other: &Self) -> Self::Output;
}

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
