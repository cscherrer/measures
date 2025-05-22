//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::{False, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::ExponentialFamily;
use crate::measures::counting::CountingMeasure;
use crate::measures::weighted::WeightedMeasure;
use num_traits::{Float, FloatConst};

/// A Poisson distribution.
///
/// The Poisson distribution has a single parameter lambda (rate) and
/// is defined over non-negative integers.
#[derive(Clone)]
pub struct Poisson<F: Float> {
    /// The rate parameter (expected number of occurrences)
    pub lambda: F,
}

impl<F: Float> Poisson<F> {
    /// Create a new Poisson distribution with the given rate.
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate parameter (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not positive.
    #[must_use]
    pub fn new(lambda: F) -> Self {
        assert!(lambda > F::zero(), "Rate parameter must be positive");
        Self { lambda }
    }
}

impl<F: Float> MeasureMarker for Poisson<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<F: Float> Measure<u64> for Poisson<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, _x: u64) -> bool {
        true // All non-negative integers are in the support
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::new()
    }
}

// Natural parameter for Poisson is log(lambda), sufficient statistic is k
impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    type NaturalParam = F; // η = log(λ)
    type SufficientStat = u64; // T(x) = x
    type BaseMeasure = WeightedMeasure<CountingMeasure<u64>, F>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self::new(param.exp())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        self.lambda.ln()
    }

    fn log_partition(&self) -> F {
        self.lambda
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        *x
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        // The base measure is a weighted counting measure
        // We use a fixed log-weight of 0 as required by measure theory
        // The factorial term will be handled in the log-density calculation
        WeightedMeasure::new(CountingMeasure::<u64>::new(), F::zero())
    }
}

// Implement From for LogDensity to f64
impl<F: Float + FloatConst> From<LogDensity<'_, u64, Poisson<F>>> for f64 {
    fn from(val: LogDensity<'_, u64, Poisson<F>>) -> Self {
        let k = *val.x;
        let lambda = val.measure.lambda;

        let k_f = F::from(k).unwrap();

        // PMF: P(X = k) = (e^-λ * λ^k) / k!
        // Log-PMF: -λ + k*log(λ) - log(k!)
        let mut log_factorial = F::zero();
        for i in 1..=k {
            log_factorial = log_factorial + F::from(i).unwrap().ln();
        }

        let result = -lambda + k_f * lambda.ln() - log_factorial;
        result.to_f64().unwrap()
    }
}
