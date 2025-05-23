//! Exponential family trait definitions.
//!
//! This module provides the core traits for exponential family distributions.
//! The framework is designed to support exponential families over different spaces:
//!
//! - `X`: The space where random variables live (could be reals, integers, vectors, etc.)
//! - `F`: The field used for numerical computations (always some Float type)
//!
//! The exponential family is represented by the standard form:
//!
//! p(x|θ) = h(x) exp(η(θ)·T(x) - A(η(θ)))
//!
//! where:
//! - η(θ) are the natural parameters
//! - T(x) are the sufficient statistics
//! - A(η) is the log-partition function
//! - h(x) is the carrier measure

use crate::core::{HasLogDensity, Measure, True};
use crate::traits::DotProduct;
use num_traits::Float;

/// A trait for exponential family distributions over space X with computations in field F.
///
/// This trait generalizes exponential families to work with:
/// - Different spaces for the random variable (X)
/// - Different spaces for the sufficient statistics
/// - A common field (F) for numerical computations
pub trait ExponentialFamily<X: Clone, F: Float>: Clone {
    /// The natural parameter type
    type NaturalParam: Clone;

    /// The sufficient statistic type
    type SufficientStat;

    /// The base measure type
    type BaseMeasure: Measure<X> + Clone;

    /// Cached computation type for optimized density evaluation.
    ///
    /// Each exponential family can define what should be cached to avoid
    /// redundant computations. This typically includes:
    /// - Natural parameters η
    /// - Log partition function A(η)
    /// - Any intermediate values needed for efficient density computation
    /// - Distribution-specific optimized values
    type Cache: Clone;

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    fn to_natural(&self) -> Self::NaturalParam;

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> F;

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;

    /// Get the base measure for this exponential family
    fn base_measure(&self) -> Self::BaseMeasure;

    /// Precompute and cache values for optimized density computation.
    ///
    /// This method should compute and store all expensive operations that are
    /// independent of the data point x, such as:
    /// - Natural parameters η
    /// - Log partition function A(η)
    /// - Any intermediate values used in density computation
    /// - Distribution-specific optimizations
    ///
    /// The cached values can then be reused across multiple density evaluations,
    /// eliminating redundant computation for batch operations.
    fn precompute_cache(&self) -> Self::Cache;

    /// Compute log-density using precomputed cached values.
    ///
    /// This method should use the cached values to compute the log-density
    /// efficiently, avoiding any redundant computations. It should
    /// implement the exponential family formula using the cached values:
    ///
    /// log p(x|θ) = η·T(x) - A(η) + log h(x)
    ///
    /// where all components of η and A(η) come from the cached values.
    fn cached_log_density(&self, cache: &Self::Cache, x: &X) -> F;

    /// Compute log-density at multiple points efficiently using cached values.
    ///
    /// This provides a default implementation that precomputes cache once
    /// and applies it to all points, but distributions can override this
    /// for further optimizations (e.g., SIMD operations).
    fn cached_log_density_batch(&self, points: &[X]) -> Vec<F> {
        let cache = self.precompute_cache();
        points
            .iter()
            .map(|x| self.cached_log_density(&cache, x))
            .collect()
    }

    /// Create an optimized closure for computing log-density.
    ///
    /// Returns a closure that captures precomputed cache and can be
    /// applied to individual points efficiently. Useful for mapping over
    /// iterators or in functional programming contexts.
    fn cached_log_density_fn(&self) -> impl Fn(&X) -> F + Clone {
        let cache = self.precompute_cache();
        let distribution = (*self).clone();
        move |x: &X| distribution.cached_log_density(&cache, x)
    }

    /// Exponential family log-density computation with automatic chain rule.
    ///
    /// Computes: η·T(x) - A(η) + log(d(base_measure)/d(root_measure))(x)
    ///
    /// The chain rule term is automatically added when the base measure differs
    /// from the root measure. This eliminates the need for manual overrides in
    /// distributions like Poisson that have non-trivial base measures.
    ///
    /// Default implementation:
    /// - Pure exponential family: η·T(x) - A(η)
    /// - Chain rule: + `base_measure.log_density_wrt_root(x)`
    /// - Combined: Complete log-density with respect to root measure
    #[cfg(feature = "profiling")]
    #[profiling::function]
    #[inline]
    fn exp_fam_log_density(&self, x: &X) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        profiling::scope!("exp_fam_computation");

        let natural_params = self.to_natural();
        let sufficient_stats = self.sufficient_statistic(x);
        let log_partition = self.log_partition();

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = {
            profiling::scope!("dot_product");
            natural_params.dot(&sufficient_stats) - log_partition
        };

        // Chain rule part: log-density of base measure with respect to root measure
        let chain_rule_part = {
            profiling::scope!("chain_rule");
            let base_measure = self.base_measure();
            base_measure.log_density_wrt_root(x)
        };

        // Complete log-density: exponential family + chain rule
        exp_fam_part + chain_rule_part
    }

    /// Exponential family log-density computation with automatic chain rule (optimized, no profiling).
    ///
    /// This version uses cached computation when available for better performance.
    #[cfg(not(feature = "profiling"))]
    #[inline]
    fn exp_fam_log_density(&self, x: &X) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        // Use cached computation if possible for better performance
        let cache = self.precompute_cache();
        self.cached_log_density(&cache, x)
    }
}

/// Helper trait for distributions that use `GenericExpFamCache`.
///
/// This trait provides default implementations for `precompute_cache` and `cached_log_density`
/// that work with `GenericExpFamCache`. Distributions using the generic cache can simply call
/// these methods in their implementations.
pub trait GenericExpFamImpl<X: Clone, F: Float>: ExponentialFamily<X, F> {
    /// Create a `GenericExpFamCache` from this distribution.
    ///
    /// Use this in your `precompute_cache` implementation:
    /// ```ignore
    /// fn precompute_cache(&self) -> Self::Cache {
    ///     self.precompute_generic_cache()
    /// }
    /// ```
    fn precompute_generic_cache(
        &self,
    ) -> crate::exponential_family::GenericExpFamCache<Self, X, F> {
        crate::exponential_family::GenericExpFamCache::new(self)
    }

    /// Compute log-density using a `GenericExpFamCache`.
    ///
    /// Use this in your `cached_log_density` implementation:
    /// ```ignore  
    /// fn cached_log_density(&self, cache: &Self::Cache, x: &X) -> F {
    ///     self.cached_log_density_generic(cache, x)
    /// }
    /// ```
    fn cached_log_density_generic(
        &self,
        cache: &crate::exponential_family::GenericExpFamCache<Self, X, F>,
        x: &X,
    ) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        cache.log_density(x)
    }
}

// Blanket implementation: any exponential family can use the generic implementations
impl<T, X, F> GenericExpFamImpl<X, F> for T
where
    T: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
{
}

/// A marker trait for measures that are exponential families
///
/// This trait serves as a marker to identify exponential family distributions
/// and enables specialized implementations for density calculations.
pub trait ExponentialFamilyMeasure<X: Clone, F: Float>:
    Measure<X, IsExponentialFamily = True> + ExponentialFamily<X, F>
{
}
