//! Core traits for exponential family distributions.
//!
//! This module defines the fundamental traits for working with exponential family distributions,
//! which have the form:
//!
//! p(x|θ) = h(x) exp(η(θ)·T(x) - A(η(θ)))
//!
//! where:
//! - η(θ) are the natural parameters
//! - T(x) are the sufficient statistics
//! - A(η) is the log-partition function
//! - h(x) is the carrier measure
//!
//! The framework supports exponential families over different spaces:
//! - The space X where the random variable lives (could be ints, vectors, etc.)
//! - The field F for numerical computations (always some Float type)

use crate::core::types::True;
use crate::core::utils::float_constant;
use crate::core::{HasLogDensity, Measure};
use crate::traits::DotProduct;
use num_traits::Float;

/// The core trait for exponential family distributions.
///
/// An exponential family distribution has the canonical form:
/// p(x|θ) = h(x) exp(η(θ)·T(x) - A(η(θ)))
///
/// This trait captures the mathematical structure with minimal complexity.
/// For performance optimization, use symbolic optimization via the `SymbolicOptimizer` trait.
pub trait ExponentialFamily<X: Clone, F: Float>: Clone {
    /// The natural parameter type η(θ)
    type NaturalParam: Clone;

    /// The sufficient statistic type T(x)
    type SufficientStat;

    /// The base measure type h(x)
    type BaseMeasure: Measure<X> + Clone;

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    #[inline]
    fn to_natural(&self) -> Self::NaturalParam {
        self.natural_and_log_partition().0
    }

    /// Compute the log partition function A(η)
    #[inline]
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
    #[inline]
    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        (self.to_natural(), self.log_partition())
    }

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;

    /// Get the base measure for this exponential family
    fn base_measure(&self) -> Self::BaseMeasure;

    /// Exponential family log-density computation with automatic chain rule.
    ///
    /// Computes: η·T(x) - A(η) + `log(d(base_measure)/d(root_measure))(x)`
    ///
    /// The chain rule term is automatically added when the base measure differs
    /// from the root measure. This eliminates the need for manual overrides in
    /// distributions like Poisson that have non-trivial base measures.
    ///
    /// This is the standard evaluation method. For high-performance scenarios,
    /// use symbolic optimization instead.
    #[inline]
    fn exp_fam_log_density(&self, x: &X) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        let (natural_params, log_partition) = self.natural_and_log_partition();
        let sufficient_stats = self.sufficient_statistic(x);

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

        // Chain rule part: log-density of base measure with respect to root measure
        let chain_rule_part = self.base_measure().log_density_wrt_root(x);

        // Complete log-density: exponential family + chain rule
        exp_fam_part + chain_rule_part
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

/// Extension trait for working with IID samples of exponential family distributions.
///
/// This trait provides efficient computation for independent and identically
/// distributed samples using the exponential family structure.
pub trait SumSufficientStats<S> {
    /// Sum sufficient statistics from multiple observations.
    ///
    /// For IID samples x₁, x₂, ..., xₙ, computes ∑ᵢT(xᵢ) efficiently.
    /// This enables the IID log-density formula: η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
    fn sum_sufficient_stats<I>(stats: I) -> S
    where
        I: Iterator<Item = S>;
}

/// Implementation for arrays of sufficient statistics.
impl<F: Float, const N: usize> SumSufficientStats<[F; N]> for [F; N] {
    #[inline]
    fn sum_sufficient_stats<I>(stats: I) -> [F; N]
    where
        I: Iterator<Item = [F; N]>,
    {
        let mut sum = [F::zero(); N];
        for stat in stats {
            for (i, &value) in stat.iter().enumerate() {
                sum[i] = sum[i] + value;
            }
        }
        sum
    }
}

/// Central helper function for computing exponential family log-density.
///
/// This function implements the complete exponential family formula with automatic
/// chain rule handling. It's used by the blanket implementation of `HasLogDensity`
/// for exponential families.
#[inline]
pub fn compute_exp_fam_log_density<X, F, D>(distribution: &D, x: &X) -> F
where
    X: Clone,
    F: Float,
    D: ExponentialFamily<X, F>,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::BaseMeasure: HasLogDensity<X, F>,
{
    distribution.exp_fam_log_density(x)
}

/// Central helper function for computing IID exponential family log-density.
///
/// For IID samples x₁, x₂, ..., xₙ from an exponential family, computes:
/// log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
///
/// This is much more efficient than computing individual log-densities and summing.
pub fn compute_iid_exp_fam_log_density<X, F, D>(distribution: &D, samples: &[X]) -> F
where
    X: Clone,
    F: Float + std::iter::Sum,
    D: ExponentialFamily<X, F>,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::SufficientStat: SumSufficientStats<D::SufficientStat>,
    D::BaseMeasure: HasLogDensity<X, F>,
{
    if samples.is_empty() {
        return F::zero();
    }

    let (natural_params, log_partition) = distribution.natural_and_log_partition();

    // Compute sum of sufficient statistics: ∑ᵢT(xᵢ)
    let sum_sufficient_stats = D::SufficientStat::sum_sufficient_stats(
        samples.iter().map(|x| distribution.sufficient_statistic(x)),
    );

    // Exponential family part: η·∑ᵢT(xᵢ) - n·A(η)
    let n = float_constant::<F>(samples.len() as f64);
    let exp_fam_part = natural_params.dot(&sum_sufficient_stats) - n * log_partition;

    // Base measure part: ∑ᵢlog h(xᵢ)
    let base_measure = distribution.base_measure();
    let base_measure_part: F = samples
        .iter()
        .map(|x| base_measure.log_density_wrt_root(x))
        .sum();

    exp_fam_part + base_measure_part
}
