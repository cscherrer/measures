//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::{False, HasLogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{ExponentialFamily, ExponentialFamilyCache};
use crate::measures::derived::factorial::FactorialMeasure;
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::{Float, FloatConst};

/// Precomputed cache for optimal Poisson distribution density computation.
///
/// This struct caches the expensive exponential family components:
/// - Natural parameter η = [ln(λ)]
/// - Log partition function A(η) = λ
/// - Base measure (for chain rule computation)
#[derive(Clone)]
pub struct PoissonCache<F: Float> {
    /// Cached natural parameter η = [ln(λ)]
    pub natural_param: [F; 1],
    /// Cached log partition function A(η) = λ
    pub log_partition: F,
    /// Cached factorial measure for chain rule computation
    pub base_measure: FactorialMeasure<F>,
}

impl<F: Float + FloatConst> PoissonCache<F> {
    #[must_use]
    pub fn new(lambda: F) -> Self {
        assert!(lambda > F::zero(), "Rate parameter must be positive");

        Self {
            natural_param: [lambda.ln()], // η = [ln(λ)]
            log_partition: lambda,        // A(η) = λ
            base_measure: FactorialMeasure::new(),
        }
    }
}

/// Implement the `ExponentialFamilyCache` trait - boilerplate eliminated!
impl<F: Float + FloatConst> ExponentialFamilyCache<u64, F> for PoissonCache<F> {
    type Distribution = Poisson<F>;

    fn from_distribution(distribution: &Self::Distribution) -> Self {
        Self::new(distribution.lambda)
    }

    fn log_partition(&self) -> F { self.log_partition }
    fn natural_params(&self) -> &[F; 1] { &self.natural_param }
    fn base_measure(&self) -> &FactorialMeasure<F> { &self.base_measure }
}

/// A Poisson distribution.
///
/// The Poisson distribution has a single parameter lambda (rate) and
/// is defined over non-negative integers.
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

// Natural parameter for Poisson is log(lambda), sufficient statistic is k as F
impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    type NaturalParam = [F; 1]; // η = [log(λ)]
    type SufficientStat = [F; 1]; // T(x) = [x] (as Float)
    type BaseMeasure = FactorialMeasure<F>;
    type Cache = PoissonCache<F>; // Use our PoissonCache as the cached type

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self::new(param[0].exp())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        [self.lambda.ln()]
    }

    fn log_partition(&self) -> F {
        // Use cached computation for efficiency
        let cache = PoissonCache::from_distribution(self);
        cache.log_partition()
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        [F::from(*x).unwrap()]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        FactorialMeasure::new()
    }

    fn precompute_cache(&self) -> Self::Cache {
        PoissonCache::new(self.lambda)
    }

    fn cached_log_density(&self, cache: &Self::Cache, x: &u64) -> F {
        // Use the new ExponentialFamilyCache trait for cleaner implementation
        cache.log_density(x)
    }
}

/// Implement `HasLogDensity` for automatic shared-root computation
impl<F: Float + FloatConst> HasLogDensity<u64, F> for Poisson<F> {
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        // Use optimized cached computation from exponential family framework
        let cache = self.precompute_cache();
        self.cached_log_density(&cache, x)
    }
}
