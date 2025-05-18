//! Standard Normal distribution implementation.
//!
//! This module provides the Standard Normal distribution, which is a continuous probability
//! distribution with mean 0 and standard deviation 1. It's a special case of the Normal
//! distribution that is particularly optimized for computations.

use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{
    ExpFamDensity, ExponentialFamily, ExponentialFamilyMeasure, compute_stdnormal_log_density,
};
use crate::measures::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

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

impl<T: Float + FloatConst> ExpFamDensity<T, T> for StdNormal<T> {}

impl<T: Float + FloatConst> ExponentialFamilyMeasure<T, T> for StdNormal<T> {}

impl<T: Float> Measure<T> for StdNormal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> <Self as Measure<T>>::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Implement HasDensity directly using our optimized calculation
impl<T: Float + FloatConst> HasDensity<T> for StdNormal<T> {
    fn log_density<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        // Create the LogDensity normally, then modify the implementation of From trait
        LogDensity::new(self, x)
    }
}

impl<T: Float + FloatConst> ExponentialFamily<T, T> for StdNormal<T> {
    type NaturalParam = [T; 2]; // (η₁, η₂) = (0, -1/2)
    type SufficientStat = [T; 2]; // (x, x²)

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
        // For StdNormal, log partition is log(sqrt(2π)) = log(2π)/2
        T::PI().ln() / T::from(2.0).unwrap() + T::from(0.5).unwrap().ln()
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [*x, *x * *x]
    }

    fn carrier_measure(&self, _x: &T) -> T {
        T::one()
    }
}

// Implement From for LogDensity to f64 - optimized for StdNormal
impl<T: Float + FloatConst> From<LogDensity<'_, T, StdNormal<T>>> for f64 {
    fn from(val: LogDensity<'_, T, StdNormal<T>>) -> Self {
        compute_stdnormal_log_density(*val.x)
    }
}

// Similarly for Lebesgue measure
impl<T: Float + FloatConst> From<LogDensity<'_, T, StdNormal<T>, LebesgueMeasure<T>>> for f64 {
    fn from(val: LogDensity<'_, T, StdNormal<T>, LebesgueMeasure<T>>) -> Self {
        // For StdNormal, the log-density with respect to Lebesgue measure
        // is the same as the log-density with respect to itself
        compute_stdnormal_log_density(*val.x)
    }
}
