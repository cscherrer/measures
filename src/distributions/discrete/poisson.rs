//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::{False, Measure, MeasureMarker, True};
use crate::exponential_family::{ExponentialFamily, GenericExpFamCache};
use crate::measures::derived::factorial::FactorialMeasure;
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::{Float, FloatConst};

/// A Poisson distribution.
///
/// The Poisson distribution has a single parameter lambda (rate) and
/// is defined over non-negative integers.
///
/// Uses the generic exponential family cache - no need for distribution-specific cache!
#[derive(Clone)]
pub struct Poisson<F: Float> {
    /// The rate parameter (expected number of occurrences)
    pub lambda: F,
}

impl<F: Float + FloatConst> Poisson<F> {
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

// Note: HasLogDensity implementation is now automatic via the blanket impl
// for exponential families in density.rs! No manual implementation needed.

impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    // Types specified once - no redundancy!
    type NaturalParam = [F; 1]; // η = [log(λ)]
    type SufficientStat = [F; 1]; // T(x) = [x] (as Float)
    type BaseMeasure = FactorialMeasure<F>;
    type Cache = GenericExpFamCache<Self, u64, F>; // Generic cache!

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self::new(param[0].exp())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        [self.lambda.ln()]
    }

    fn log_partition(&self) -> F {
        // For Poisson(λ), A(η) = e^η = λ
        self.lambda
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [F::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        FactorialMeasure::new()
    }

    fn precompute_cache(&self) -> Self::Cache {
        GenericExpFamCache::new(self)
    }

    fn cached_log_density(&self, cache: &Self::Cache, x: &u64) -> F {
        cache.log_density(x)
    }
}
