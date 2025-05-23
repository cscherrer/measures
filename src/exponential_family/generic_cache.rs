//! Generic cache implementation for exponential family distributions.
//!
//! This module provides a single generic cache that works for any exponential family,
//! eliminating the need for distribution-specific cache implementations.

use crate::core::HasLogDensity;
use crate::exponential_family::ExponentialFamily;
use crate::traits::DotProduct;
use num_traits::Float;
use std::marker::PhantomData;

/// A generic cache for any exponential family distribution.
///
/// This eliminates the need for distribution-specific cache types by using
/// the associated types directly from the `ExponentialFamily` implementation.
/// 
/// Instead of each distribution defining its own cache struct, they can all use:
/// `type Cache = GenericExpFamCache<Self, X, F>;`
#[derive(Clone)]
pub struct GenericExpFamCache<D, X, F>
where
    D: ExponentialFamily<X, F> + Clone,
    X: Clone,
    F: Float,
    D::NaturalParam: Clone,
    D::BaseMeasure: Clone,
{
    /// Cached natural parameters η - type comes from D::NaturalParam
    pub natural_params: D::NaturalParam,
    /// Cached log partition function A(η)
    pub log_partition: F,
    /// Cached base measure - type comes from D::BaseMeasure
    pub base_measure: D::BaseMeasure,
    /// Phantom data to bind the distribution and point types
    _phantom: PhantomData<(D, X)>,
}

impl<D, X, F> GenericExpFamCache<D, X, F>
where
    D: ExponentialFamily<X, F> + Clone,
    X: Clone,
    F: Float,
    D::NaturalParam: Clone,
    D::BaseMeasure: Clone,
{
    /// Create a new generic cache from any exponential family distribution.
    #[must_use]
    pub fn new(distribution: &D) -> Self {
        Self {
            natural_params: distribution.to_natural(),
            log_partition: distribution.log_partition(),
            base_measure: distribution.base_measure(),
            _phantom: PhantomData,
        }
    }

    /// Get the cached natural parameters η.
    pub fn natural_params(&self) -> &D::NaturalParam {
        &self.natural_params
    }

    /// Get the cached log partition function A(η).
    pub fn log_partition(&self) -> F {
        self.log_partition
    }

    /// Get the cached base measure.
    pub fn base_measure(&self) -> &D::BaseMeasure {
        &self.base_measure
    }

    /// Compute log-density at a point using cached values.
    ///
    /// Uses the standard exponential family formula: η·T(x) - A(η) + log h(x)
    pub fn log_density(&self, x: &X) -> F
    where
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::BaseMeasure: HasLogDensity<X, F>,
    {
        // Create a temporary distribution instance to get sufficient statistics
        let distribution = D::from_natural(self.natural_params.clone());
        
        // Sufficient statistics: T(x)
        let sufficient_stats = distribution.sufficient_statistic(x);

        // Exponential family part: η·T(x) - A(η)
        let exp_fam_part = self.natural_params.dot(&sufficient_stats) - self.log_partition;

        // Chain rule part: log h(x) = log(d(base_measure)/d(root_measure))(x)
        let chain_rule_part = self.base_measure.log_density_wrt_root(x);

        // Complete log-density
        exp_fam_part + chain_rule_part
    }

    /// Compute log-density for multiple points efficiently.
    pub fn log_density_batch(&self, points: &[X]) -> Vec<F>
    where
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::BaseMeasure: HasLogDensity<X, F>,
    {
        points.iter().map(|x| self.log_density(x)).collect()
    }

    /// Create an optimized closure for computing log-density.
    pub fn log_density_fn(&self) -> impl Fn(&X) -> F + Clone
    where
        Self: Clone,
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::BaseMeasure: HasLogDensity<X, F>,
    {
        let cache = self.clone();
        move |x: &X| cache.log_density(x)
    }
}

/// Implement ExponentialFamilyCache for the generic cache.
///
/// This provides the standard interface that the existing code expects.
impl<D, X, F> crate::exponential_family::ExponentialFamilyCache<X, F> for GenericExpFamCache<D, X, F>
where
    D: ExponentialFamily<X, F> + Clone,
    X: Clone,
    F: Float,
    D::NaturalParam: Clone,
    D::BaseMeasure: Clone,
{
    type Distribution = D;

    fn from_distribution(distribution: &Self::Distribution) -> Self {
        Self::new(distribution)
    }

    fn log_partition(&self) -> F {
        self.log_partition
    }

    fn natural_params(&self) -> &D::NaturalParam {
        &self.natural_params
    }

    fn base_measure(&self) -> &D::BaseMeasure {
        &self.base_measure
    }
} 