//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::{False, HasLogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::ExponentialFamily;
use crate::measures::derived::weighted::WeightedMeasure;
use crate::measures::primitive::counting::CountingMeasure;
use crate::traits::DotProduct;
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

// Natural parameter for Poisson is log(lambda), sufficient statistic is k as F
impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    type NaturalParam = [F; 1]; // η = [log(λ)]
    type SufficientStat = [F; 1]; // T(x) = [x] (as Float)
    type BaseMeasure = WeightedMeasure<CountingMeasure<u64>, F>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self::new(param[0].exp())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        [self.lambda.ln()]
    }

    fn log_partition(&self) -> F {
        self.lambda
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [F::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        // The base measure is a weighted counting measure
        // We use a fixed log-weight of 0 as required by measure theory
        // The factorial term will be handled in the log-density calculation
        WeightedMeasure::new(CountingMeasure::<u64>::new(), F::zero())
    }

    // Override for the factorial term that's not in the generic implementation
    fn exp_fam_log_density(&self, x: &u64) -> F {
        let k = *x;
        let natural_param = self.to_natural();
        let sufficient_stat = self.sufficient_statistic(x);
        let log_partition = self.log_partition();

        // Compute log(k!)
        let mut log_factorial = F::zero();
        for i in 1..=k {
            log_factorial = log_factorial + F::from(i).unwrap().ln();
        }

        // η·T(x) - A(η) - log(k!)
        // Now using DotProduct for [F; 1] arrays
        natural_param.dot(&sufficient_stat) - log_partition - log_factorial
    }
}

/// Implement `HasLogDensity` for automatic shared-root computation  
impl<F: Float + FloatConst> HasLogDensity<u64, F> for Poisson<F> {
    fn log_density_wrt_root(&self, x: &u64) -> F {
        self.exp_fam_log_density(x)
    }
}
