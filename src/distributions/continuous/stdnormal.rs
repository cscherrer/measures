//! Standard Normal distribution implementation.
//!
//! This module provides the Standard Normal distribution, which is a continuous probability
//! distribution with mean 0 and standard deviation 1. It's a special case of the Normal
//! distribution that is particularly optimized for computations.

use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use measures_core::float_constant;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use num_traits::{Float, FloatConst};

/// Standard normal distribution N(0, 1).
///
/// This is a special case of the normal distribution with mean 0 and standard deviation 1.
/// It's a member of the exponential family with:
/// - Natural parameters: η = [0, -1/2]
/// - Sufficient statistics: T(x) = [x, x²]
/// - Log partition: A(η) = -½log(2π)
/// - Base measure: Lebesgue measure (dx)
#[derive(Clone, Debug)]
pub struct StdNormal<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + FloatConst> Default for StdNormal<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> MeasureMarker for StdNormal<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float + FloatConst> StdNormal<T> {
    /// Create a new standard normal distribution.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Measure<T> for StdNormal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for StdNormal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        // For standard normal, we ignore the parameters and always return N(0,1)
        let _ = param;
        Self::new()
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let natural_params = [
            T::zero(),                 // μ/σ² = 0/1 = 0
            float_constant::<T>(-0.5), // -1/(2σ²) = -1/2
        ];

        let log_partition = (float_constant::<T>(2.0) * T::PI()).ln() * float_constant::<T>(0.5);

        (natural_params, log_partition)
    }
}
