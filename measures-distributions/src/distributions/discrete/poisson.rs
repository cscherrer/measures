//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use measures_core::float_constant;
use measures_core::primitive::counting::CountingMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
use num_traits::{Float, FloatConst};
use measures_combinators::measures::derived::factorial::FactorialMeasure;

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

    fn from_natural(param: <Self as ExponentialFamily<u64, F>>::NaturalParam) -> Self {
        let [eta] = param;
        let lambda = eta.exp();
        Self::new(lambda)
    }

    fn sufficient_statistic(&self, x: &u64) -> <Self as ExponentialFamily<u64, F>>::SufficientStat {
        [float_constant::<F>(*x as f64)]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<u64, F>>::BaseMeasure {
        FactorialMeasure::<F>::new()
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<u64, F>>::NaturalParam, F) {
        let natural_param = [self.rate.ln()];
        let log_partition = self.rate;
        (natural_param, log_partition)
    }
}

// Implementation of HasLogDensity for Poisson distribution
impl<F: Float + FloatConst> measures_core::HasLogDensity<u64, F> for Poisson<F> {
    fn log_density_wrt_root(&self, x: &u64) -> F {
        // Poisson PMF: P(X = k) = (λ^k * e^(-λ)) / k!
        // log P(X = k) = k * log(λ) - λ - log(k!)
        let k_f = float_constant::<F>(*x as f64);
        let log_factorial =
            (1..=*x).fold(F::zero(), |acc, i| acc + float_constant::<F>(i as f64).ln());
        k_f * self.rate.ln() - self.rate - log_factorial
    }
}
