//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::types::{False, True};
use crate::core::utils::float_constant;
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::derived::factorial::FactorialMeasure;
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::{Float, FloatConst};

/// Poisson distribution with rate parameter λ.
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [log(λ)]
/// - Sufficient statistics: T(x) = [x]
/// - Log partition: A(η) = exp(η)
/// - Base measure: Factorial measure (factorial term)
#[derive(Clone, Debug)]
pub struct Poisson<F> {
    pub rate: F,
}

impl<F: Float> Default for Poisson<F> {
    fn default() -> Self {
        Self { rate: F::one() }
    }
}

impl<F: Float> MeasureMarker for Poisson<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<F: Float + FloatConst> Poisson<F> {
    /// Create a new Poisson distribution with the given rate parameter.
    pub fn new(rate: F) -> Self {
        assert!(rate > F::zero(), "Rate parameter must be positive");
        Self { rate }
    }
}

impl<F: Float> Measure<u64> for Poisson<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, _x: u64) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u64>::new()
    }
}

// Exponential family implementation
impl<F> ExponentialFamily<u64, F> for Poisson<F>
where
    F: Float + FloatConst + std::fmt::Debug + 'static,
{
    type NaturalParam = [F; 1];
    type SufficientStat = [F; 1];
    type BaseMeasure = FactorialMeasure<F>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta] = param;
        Self::new(eta.exp())
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [float_constant::<F>(*x as f64)]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        FactorialMeasure::<F>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        let natural_params = [self.rate.ln()];
        let log_partition = self.rate;
        (natural_params, log_partition)
    }
}
