//! Utility functions for working with arrays in exponential families.
//!
//! This module provides helper functions to work around Rust's orphan rules
//! when dealing with array operations like addition.

/// Add two arrays element-wise.
pub fn add_arrays<T, const N: usize>(lhs: [T; N], rhs: [T; N]) -> [T; N]
where
    T: std::ops::Add<Output = T> + Default + Copy,
{
    let mut result = [T::default(); N];
    for i in 0..N {
        result[i] = lhs[i] + rhs[i];
    }
    result
}

/// Create a default array.
#[must_use]
pub fn default_array<T, const N: usize>() -> [T; N]
where
    T: Default,
{
    std::array::from_fn(|_| T::default())
}

/// Sum an iterator of arrays.
pub fn sum_arrays<T, const N: usize, I>(iter: I) -> [T; N]
where
    T: std::ops::Add<Output = T> + Default + Copy,
    I: Iterator<Item = [T; N]>,
{
    iter.fold(default_array(), |acc, item| add_arrays(acc, item))
}
