//! Standard Normal distribution implementation.
//!
//! This module provides the Standard Normal distribution, which is a continuous probability
//! distribution with mean 0 and standard deviation 1. It's a special case of the Normal
//! distribution that is particularly optimized for computations.

use crate::core::{False, Measure, MeasureMarker, True};
use crate::exponential_family::{ExponentialFamily, ExponentialFamilyCache};
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

/// Precomputed cache for standard normal distribution.
///
/// This struct caches the exponential family components for standard normal:
/// - Natural parameters η = [0, -1/2]
/// - Log partition function A(η) = ½log(2π)
/// - Base measure (Lebesgue measure)
#[derive(Clone)]
pub struct StdNormalCache<T: Float> {
    /// Cached natural parameters η = [0, -1/2]
    pub natural_params: [T; 2],
    /// Cached log partition function A(η) = ½log(2π)
    pub log_partition: T,
    /// Cached base measure
    pub base_measure: LebesgueMeasure<T>,
}

impl<T: Float + FloatConst> StdNormalCache<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            natural_params: [T::zero(), -T::from(0.5).unwrap()],
            log_partition: T::PI().ln() / T::from(2.0).unwrap() + T::from(0.5).unwrap().ln(),
            base_measure: LebesgueMeasure::new(),
        }
    }
}

impl<T: Float + FloatConst> Default for StdNormalCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement the `ExponentialFamilyCache` trait - boilerplate eliminated!
impl<T: Float + FloatConst> ExponentialFamilyCache<T, T> for StdNormalCache<T> {
    type Distribution = StdNormal<T>;

    fn from_distribution(_distribution: &Self::Distribution) -> Self {
        Self::new()
    }

    fn log_partition(&self) -> T {
        self.log_partition
    }
    fn natural_params(&self) -> &[T; 2] {
        &self.natural_params
    }
    fn base_measure(&self) -> &LebesgueMeasure<T> {
        &self.base_measure
    }
}

/// A standard normal (Gaussian) distribution with mean 0 and standard deviation 1.
///
/// This specialized implementation is optimized for the standard case.
#[derive(Clone, Debug, Default)]
pub struct StdNormal<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> StdNormal<T> {
    /// Create a new standard normal distribution.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> MeasureMarker for StdNormal<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Measure<T> for StdNormal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> <Self as Measure<T>>::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

impl<T: Float + FloatConst> ExponentialFamily<T, T> for StdNormal<T> {
    type NaturalParam = [T; 2]; // (η₁, η₂) = (0, -1/2)
    type SufficientStat = [T; 2]; // (x, x²)
    type BaseMeasure = LebesgueMeasure<T>;
    type Cache = StdNormalCache<T>; // Simple cache for standard normal

    fn from_natural(_param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        // For StdNormal, natural parameters are always (0, -1/2)
        // This implementation is included for completeness
        Self::new()
    }

    fn to_natural(&self) -> <Self as ExponentialFamily<T, T>>::NaturalParam {
        // For StdNormal, natural parameters are (0, -1/2)
        [T::zero(), -T::from(0.5).unwrap()]
    }

    fn log_partition(&self) -> T {
        // Use cached computation for efficiency
        let cache = StdNormalCache::from_distribution(self);
        cache.log_partition()
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn precompute_cache(&self) -> Self::Cache {
        StdNormalCache::new()
    }

    fn cached_log_density(&self, cache: &Self::Cache, x: &T) -> T {
        // Use the new ExponentialFamilyCache trait for cleaner implementation
        cache.log_density(x)
    }
}
