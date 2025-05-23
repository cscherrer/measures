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

/// Helper trait for summing sufficient statistics.
///
/// This trait enables efficient computation of sufficient statistics for IID samples
/// and other aggregate operations in exponential families.
pub trait SumSufficientStats: Sized {
    /// Sum a collection of sufficient statistics
    fn sum_stats(stats: &[Self]) -> Self;
}

/// Implementation for array sufficient statistics [F; N]
impl<F: Float, const N: usize> SumSufficientStats for [F; N] {
    fn sum_stats(stats: &[Self]) -> Self {
        if stats.is_empty() {
            return [F::zero(); N];
        }

        let mut result = [F::zero(); N];
        for stat in stats {
            for (i, &val) in stat.iter().enumerate() {
                result[i] = result[i] + val;
            }
        }
        result
    }
}

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
    fn to_natural(&self) -> Self::NaturalParam {
        self.natural_and_log_partition().0
    }

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> F {
        self.natural_and_log_partition().1
    }

    /// Compute both natural parameters and log partition function efficiently.
    ///
    /// Many exponential families share expensive computations between η(θ) and A(η).
    /// This method allows computing both together to avoid duplication.
    ///
    /// Default implementation calls the separate methods, but distributions should
    /// override this for better performance when there are shared computations.
    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        (self.to_natural(), self.log_partition())
    }

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;

    /// Get the base measure for this exponential family
    fn base_measure(&self) -> Self::BaseMeasure;

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

        let (natural_params, log_partition) = self.natural_and_log_partition();
        let sufficient_stats = self.sufficient_statistic(x);

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
        Self: PrecomputeCache<X, F>,
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

/// A marker trait for measures that are exponential families
///
/// This trait serves as a marker to identify exponential family distributions
/// and enables specialized implementations for density calculations.
pub trait ExponentialFamilyMeasure<X: Clone, F: Float>:
    Measure<X, IsExponentialFamily = True> + ExponentialFamily<X, F>
{
}

/// Extension trait for exponential families that support cache precomputation.
///
/// This trait provides a default implementation for distributions using `GenericExpFamCache`.
/// Distributions can opt into this behavior by implementing this trait, or provide their own
/// custom cache precomputation logic.
pub trait PrecomputeCache<X: Clone, F: Float>: ExponentialFamily<X, F> {
    /// Precompute and cache expensive operations for efficient density evaluation.
    ///
    /// This method should compute and store values that are used repeatedly
    /// in density calculations, such as:
    /// - Natural parameters η
    /// - Log partition function A(η)
    /// - Any intermediate computations
    fn precompute_cache(&self) -> Self::Cache;

    /// Default implementation using `GenericExpFamCache`.
    ///
    /// Most distributions can use this default implementation, which provides
    /// a generic cache that stores natural parameters and log partition.
    fn precompute_cache_default(&self) -> crate::exponential_family::GenericExpFamCache<Self, X, F>
    where
        Self: Sized,
    {
        crate::exponential_family::GenericExpFamCache::new(self)
    }

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
}

/// Central helper function for computing exponential family log-density.
///
/// This function implements the standard exponential family formula:
/// log p(x|θ) = η·T(x) - A(η) + log h(x)
///
/// By centralizing this computation, we ensure consistency across all
/// exponential family implementations and reduce code duplication.
pub fn compute_exp_fam_log_density<D, X, F>(distribution: &D, x: &X) -> F
where
    D: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::BaseMeasure: HasLogDensity<X, F>,
{
    // Get exponential family components
    let (natural_params, log_partition) = distribution.natural_and_log_partition();
    let sufficient_stats = distribution.sufficient_statistic(x);
    let base_measure = distribution.base_measure();

    // Exponential family part: η·T(x) - A(η)
    let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

    // Chain rule part: log h(x) = log(d(base_measure)/d(root_measure))(x)
    let chain_rule_part = base_measure.log_density_wrt_root(x);

    // Complete log-density
    exp_fam_part + chain_rule_part
}

/// Central helper function for computing IID exponential family log-density.
///
/// This function implements the efficient IID computation:
/// log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
///
/// This is more efficient than summing individual log-densities because it:
/// - Computes natural parameters and log partition only once
/// - Computes sufficient statistics sum directly
/// - Scales the log partition by sample size
pub fn compute_iid_exp_fam_log_density<D, X, F>(distribution: &D, xs: &[X]) -> F
where
    D: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::SufficientStat: SumSufficientStats,
    D::BaseMeasure: HasLogDensity<X, F>,
{
    let n = F::from(xs.len()).unwrap();

    // Handle empty case
    if xs.is_empty() {
        return F::zero();
    }

    // 1. Compute sufficient statistics: ∑ᵢT(xᵢ)
    let individual_stats: Vec<D::SufficientStat> = xs
        .iter()
        .map(|x| distribution.sufficient_statistic(x))
        .collect();
    let sum_sufficient_stats = D::SufficientStat::sum_stats(&individual_stats);

    // 2. Get natural parameters and log partition efficiently: (η, A(η))
    let (natural_params, log_partition) = distribution.natural_and_log_partition();

    // 3. Exponential family computation: η·∑ᵢT(xᵢ) - n·A(η)
    let exp_fam_part = natural_params.dot(&sum_sufficient_stats) - n * log_partition;

    // 4. Base measure part: ∑ᵢlog h(xᵢ)
    let base_measure = distribution.base_measure();
    let base_measure_part: F = xs
        .iter()
        .map(|x| base_measure.log_density_wrt_root(x))
        .fold(F::zero(), |acc, x| acc + x);

    // Complete log-density
    exp_fam_part + base_measure_part
}

// Blanket implementation: any exponential family can use the generic implementations
impl<T, X, F> GenericExpFamImpl<X, F> for T
where
    T: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
{
}
