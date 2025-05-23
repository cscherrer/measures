//! IID (Independent and Identically Distributed) exponential family implementations.
//!
//! This module provides a wrapper that takes any exponential family and creates a new
//! exponential family representing n independent and identically distributed samples.
//!
//! For an exponential family p(x|θ) = h(x) exp(η·T(x) - A(η)), the joint distribution
//! of n iid samples X₁, X₂, ..., Xₙ is also an exponential family:
//!
//! p(x₁,...,xₙ|θ) = ∏ᵢ h(xᵢ) exp(η·∑ᵢT(xᵢ) - n·A(η))
//!
//! This has:
//! - Same natural parameter η
//! - Sufficient statistic: sum of individual sufficient statistics ∑ᵢT(xᵢ)
//! - Log partition function: n times the original A(η)

use crate::core::{False, HasLogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::ExponentialFamily;
use crate::measures::primitive::counting::CountingMeasure;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use crate::traits::DotProduct;
use num_traits::Float;

/// An IID (Independent and Identically Distributed) wrapper for exponential families.
///
/// This represents the joint distribution of multiple independent samples from the same
/// exponential family distribution.
#[derive(Clone)]
pub struct IID<D> {
    /// The underlying distribution
    pub distribution: D,
}

impl<D> IID<D> {
    /// Create a new IID wrapper for the given distribution.
    pub fn new(distribution: D) -> Self {
        Self { distribution }
    }
}

impl<D> MeasureMarker for IID<D> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

/// Helper trait to determine the IID root measure type
pub trait IIDRootMeasure<X: Clone> {
    type IIDRoot: Measure<Vec<X>>;
    fn iid_root_measure() -> Self::IIDRoot;
}

/// Implementation for `LebesgueMeasure`
impl<X: Clone> IIDRootMeasure<X> for LebesgueMeasure<X> {
    type IIDRoot = LebesgueMeasure<Vec<X>>;

    fn iid_root_measure() -> Self::IIDRoot {
        LebesgueMeasure::new()
    }
}

/// Implementation for `CountingMeasure`
impl<X: Clone> IIDRootMeasure<X> for CountingMeasure<X> {
    type IIDRoot = CountingMeasure<Vec<X>>;

    fn iid_root_measure() -> Self::IIDRoot {
        CountingMeasure::new()
    }
}

/// Single implementation for IID distributions
impl<D, X> Measure<Vec<X>> for IID<D>
where
    D: Measure<X>,
    D::RootMeasure: IIDRootMeasure<X>,
    X: Clone,
{
    type RootMeasure = <D::RootMeasure as IIDRootMeasure<X>>::IIDRoot;

    fn in_support(&self, xs: Vec<X>) -> bool {
        xs.iter().all(|x| self.distribution.in_support(x.clone()))
    }

    fn root_measure(&self) -> Self::RootMeasure {
        D::RootMeasure::iid_root_measure()
    }
}

/// Helper trait for summing sufficient statistics
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

/// Helper trait for computing IID log-density manually (backward compatibility)
impl<D> IID<D>
where
    D: Clone,
{
    /// Compute IID log-density by summing individual log-densities.
    ///
    /// NOTE: This is inefficient and kept only for comparison.
    /// The goal is to implement the efficient exponential family approach.
    pub fn compute_iid_log_density<X, F>(&self, xs: &Vec<X>) -> F
    where
        D: HasLogDensity<X, F>,
        X: Clone,
        F: Float,
    {
        xs.iter()
            .map(|x| self.distribution.log_density_wrt_root(x))
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Compute efficient IID log-density using exponential family structure.
    ///
    /// This demonstrates the proper approach: η·∑ᵢT(xᵢ) - n·A(η)
    pub fn efficient_iid_log_density<X, F>(&self, xs: &Vec<X>) -> F
    where
        D: ExponentialFamily<X, F>,
        F: Float,
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::SufficientStat: SumSufficientStats,
        X: Clone,
    {
        let n = F::from(xs.len()).unwrap();

        // 1. Compute sufficient statistics: ∑ᵢT(xᵢ)
        let individual_stats: Vec<D::SufficientStat> = xs
            .iter()
            .map(|x| self.distribution.sufficient_statistic(x))
            .collect();
        let sum_sufficient_stats = D::SufficientStat::sum_stats(&individual_stats);

        // 2. Get natural parameters and log partition efficiently: (η, A(η))
        let (natural_params, log_partition) = self.distribution.natural_and_log_partition();

        // 3. Exponential family computation: η·∑ᵢT(xᵢ) - n·A(η)
        natural_params.dot(&sum_sufficient_stats) - n * log_partition
    }
}

/// Extension trait that adds the `iid()` method to exponential families.
pub trait IIDExtension<X: Clone, F: Float>: ExponentialFamily<X, F> + Sized {
    /// Create an IID version of this exponential family.
    ///
    /// The returned distribution uses the efficient exponential family structure
    /// for computation: η·∑ᵢT(xᵢ) - n·A(η).
    fn iid(self) -> IID<Self> {
        IID::new(self)
    }
}

// Blanket implementation for all exponential families
impl<T, X, F> IIDExtension<X, F> for T
where
    T: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
{
}
