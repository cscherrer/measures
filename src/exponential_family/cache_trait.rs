//! Cache trait for exponential family distributions.
//!
//! This module provides a trait for creating and managing cached computations
//! for exponential family distributions, enabling efficient batch operations
//! and eliminating redundant calculations.

use crate::core::HasLogDensity;
use crate::exponential_family::ExponentialFamily;
use crate::traits::DotProduct;
use num_traits::Float;

/// A simple trait for accessing cached exponential family fields.
///
/// This trait abstracts over the field access pattern common to all
/// exponential family caches, enabling a single blanket implementation
/// of `ExponentialFamilyCache` for all distributions.
pub trait CacheFields<X: Clone, F: Float> {
    /// The distribution type this cache is for
    type Distribution: ExponentialFamily<X, F>;

    /// Create a new cache from a distribution
    fn from_distribution(distribution: &Self::Distribution) -> Self;

    /// Access the cached log partition value
    fn log_partition(&self) -> F;

    /// Access the cached natural parameters  
    fn natural_params(&self) -> &<Self::Distribution as ExponentialFamily<X, F>>::NaturalParam;

    /// Access the cached base measure
    fn base_measure(&self) -> &<Self::Distribution as ExponentialFamily<X, F>>::BaseMeasure;
}

/// A trait for cached computations of exponential family distributions.
///
/// This trait separates cache management from the distribution itself,
/// allowing for:
/// - Efficient batch operations (create cache once, use many times)
/// - Cleaner API (no need for `precompute_cache()` + `cached_log_density()`)
/// - Reusable cached values (`log_partition`, `natural_params`, etc.)
/// - Generic implementation that works for all exponential families
///
/// The trait provides a default implementation of `log_density` that uses the
/// standard exponential family formula: η·T(x) - A(η) + log h(x)
///
/// # Example
///
/// ```rust
/// use measures::distributions::continuous::normal::Normal;
/// use measures::exponential_family::{ExponentialFamilyCache, GenericExpFamCache};
///
/// let normal = Normal::new(0.0, 1.0);
/// let cache: GenericExpFamCache<Normal<f64>, f64, f64> = GenericExpFamCache::from_distribution(&normal);
///
/// // Reuse cache for multiple computations
/// let density1 = cache.log_density(&0.5);
/// let density2 = cache.log_density(&1.0);
/// let log_partition = cache.log_partition(); // cached value
/// ```
pub trait ExponentialFamilyCache<X: Clone, F: Float>: Clone {
    /// The distribution type this cache is for
    type Distribution: ExponentialFamily<X, F>;

    /// Create a new cache from a distribution.
    ///
    /// This method should compute and store all expensive operations
    /// that are independent of the data point x:
    /// - Natural parameters η
    /// - Log partition function A(η)
    /// - Any intermediate values for efficient computation
    fn from_distribution(distribution: &Self::Distribution) -> Self;

    /// Get the cached log partition function A(η).
    ///
    /// This returns the precomputed value without any recomputation.
    fn log_partition(&self) -> F;

    /// Get the cached natural parameters η.
    ///
    /// This returns the precomputed natural parameters.
    fn natural_params(&self) -> &<Self::Distribution as ExponentialFamily<X, F>>::NaturalParam;

    /// Get the base measure for chain rule computation.
    ///
    /// This should return the cached base measure if needed for efficient
    /// chain rule computation.
    fn base_measure(&self) -> &<Self::Distribution as ExponentialFamily<X, F>>::BaseMeasure;

    /// Compute log-density at a point using cached values.
    ///
    /// This provides a default implementation using the exponential family formula:
    /// log p(x|θ) = η·T(x) - A(η) + log h(x)
    ///
    /// All components η and A(η) come from cached values. Most implementations
    /// should not need to override this method.
    fn log_density(&self, x: &X) -> F
    where
        <Self::Distribution as ExponentialFamily<X, F>>::NaturalParam: DotProduct<<Self::Distribution as ExponentialFamily<X, F>>::SufficientStat, Output = F>
            + Clone,
        <Self::Distribution as ExponentialFamily<X, F>>::BaseMeasure: HasLogDensity<X, F>,
    {
        // Create a temporary distribution instance to get sufficient statistics
        // This is a design trade-off: we could cache the distribution, but that
        // would make the cache heavier. For most use cases, this delegation is fine.
        let distribution = Self::Distribution::from_natural(self.natural_params().clone());

        // Sufficient statistics: T(x)
        let sufficient_stats = distribution.sufficient_statistic(x);

        // Exponential family part: η·T(x) - A(η)
        let exp_fam_part = self.natural_params().dot(&sufficient_stats) - self.log_partition();

        // Chain rule part: log h(x) = log(d(base_measure)/d(root_measure))(x)
        let chain_rule_part = self.base_measure().log_density_wrt_root(x);

        // Complete log-density
        exp_fam_part + chain_rule_part
    }

    /// Compute log-density for multiple points efficiently.
    ///
    /// Default implementation applies cached computation to each point,
    /// but implementations can override for further optimizations.
    fn log_density_batch(&self, points: &[X]) -> Vec<F>
    where
        X: Clone,
        <Self::Distribution as ExponentialFamily<X, F>>::NaturalParam: DotProduct<<Self::Distribution as ExponentialFamily<X, F>>::SufficientStat, Output = F>
            + Clone,
        <Self::Distribution as ExponentialFamily<X, F>>::BaseMeasure: HasLogDensity<X, F>,
    {
        points.iter().map(|x| self.log_density(x)).collect()
    }

    /// Create an optimized closure for computing log-density.
    ///
    /// Returns a closure that can be applied to individual points efficiently.
    /// Useful for mapping over iterators or in functional programming contexts.
    fn log_density_fn(&self) -> impl Fn(&X) -> F + Clone
    where
        Self: Clone,
        <Self::Distribution as ExponentialFamily<X, F>>::NaturalParam: DotProduct<<Self::Distribution as ExponentialFamily<X, F>>::SufficientStat, Output = F>
            + Clone,
        <Self::Distribution as ExponentialFamily<X, F>>::BaseMeasure: HasLogDensity<X, F>,
    {
        let cache = self.clone();
        move |x: &X| cache.log_density(x)
    }
}
