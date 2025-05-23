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
pub trait ExponentialFamily<X, F: Float> {
    /// The natural parameter type
    type NaturalParam;

    /// The sufficient statistic type
    type SufficientStat;

    /// The base measure type
    type BaseMeasure: Measure<X>;

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
    #[cfg(not(feature = "profiling"))]
    #[inline]
    fn exp_fam_log_density(&self, x: &X) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        let natural_params = self.to_natural();
        let sufficient_stats = self.sufficient_statistic(x);
        let log_partition = self.log_partition();

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

        // Chain rule part: log-density of base measure with respect to root measure
        let base_measure = self.base_measure();
        let chain_rule_part = base_measure.log_density_wrt_root(x);

        // Complete log-density: exponential family + chain rule
        exp_fam_part + chain_rule_part
    }
}

/// A marker trait for measures that are exponential families
///
/// This trait serves as a marker to identify exponential family distributions
/// and enables specialized implementations for density calculations.
pub trait ExponentialFamilyMeasure<X, F: Float>:
    Measure<X, IsExponentialFamily = True> + ExponentialFamily<X, F>
{
}
