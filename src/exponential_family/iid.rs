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
use crate::exponential_family::{ExponentialFamily, SumSufficientStats};
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

/// Helper trait for computing IID log-density manually (backward compatibility)
impl<D> IID<D>
where
    D: Clone,
{
    /// Compute efficient IID log-density using exponential family structure.
    ///
    /// This uses the proper exponential family approach: η·∑ᵢT(xᵢ) - n·A(η)
    /// This is both mathematically correct and computationally efficient.
    pub fn iid_log_density<X, F>(&self, xs: &[X]) -> F
    where
        D: ExponentialFamily<X, F>,
        F: Float,
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::SufficientStat: SumSufficientStats,
        D::BaseMeasure: HasLogDensity<X, F>,
        X: Clone,
    {
        // Use the centralized computation function
        crate::exponential_family::traits::compute_iid_exp_fam_log_density(&self.distribution, xs)
    }

    /// Compute IID log-density using fallback summation for non-exponential families.
    ///
    /// This method provides backward compatibility for distributions that don't
    /// implement the exponential family trait.
    pub fn iid_log_density_fallback<X, F>(&self, xs: &[X]) -> F
    where
        D: HasLogDensity<X, F>,
        X: Clone,
        F: Float,
    {
        xs.iter()
            .map(|x| self.distribution.log_density_wrt_root(x))
            .fold(F::zero(), |acc, x| acc + x)
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

/// Product measure for IID base measures
///
/// This represents the base measure for IID collections: ∏ᵢh(xᵢ)
#[derive(Clone)]
pub struct IIDBaseMeasure<M> {
    /// The underlying base measure
    pub base_measure: M,
}

impl<M> IIDBaseMeasure<M> {
    /// Create a new IID base measure wrapper
    pub fn new(base_measure: M) -> Self {
        Self { base_measure }
    }
}

impl<M> MeasureMarker for IIDBaseMeasure<M> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<M, X> Measure<Vec<X>> for IIDBaseMeasure<M>
where
    M: Measure<X>,
    M::RootMeasure: IIDRootMeasure<X>,
    X: Clone,
{
    type RootMeasure = <M::RootMeasure as IIDRootMeasure<X>>::IIDRoot;

    fn in_support(&self, xs: Vec<X>) -> bool {
        xs.iter().all(|x| self.base_measure.in_support(x.clone()))
    }

    fn root_measure(&self) -> Self::RootMeasure {
        M::RootMeasure::iid_root_measure()
    }
}

impl<M, X, F> HasLogDensity<Vec<X>, F> for IIDBaseMeasure<M>
where
    M: HasLogDensity<X, F>,
    X: Clone,
    F: Float,
{
    fn log_density_wrt_root(&self, xs: &Vec<X>) -> F {
        // Product of individual base measures: ∑ᵢ log h(xᵢ)
        xs.iter()
            .map(|x| self.base_measure.log_density_wrt_root(x))
            .fold(F::zero(), |acc, x| acc + x)
    }
}

/// IID exponential family implementation
///
/// For an exponential family p(x|θ) = h(x) exp(η·T(x) - A(η)), the joint distribution
/// of n iid samples X₁, X₂, ..., Xₙ is also an exponential family:
///
/// p(x₁,...,xₙ|θ) = ∏ᵢ h(xᵢ) exp(η·∑ᵢT(xᵢ) - n·A(η))
///
/// This has:
/// - Same natural parameter η
/// - Sufficient statistic: sum of individual sufficient statistics ∑ᵢT(xᵢ)
/// - Log partition function: n times the original A(η)
/// - Base measure: product of individual base measures ∏ᵢh(xᵢ)
impl<D, X, F> ExponentialFamily<Vec<X>, F> for IID<D>
where
    D: ExponentialFamily<X, F>,
    X: Clone,
    F: Float,
    D::SufficientStat: SumSufficientStats,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::BaseMeasure: HasLogDensity<X, F> + Measure<X>,
    <D::BaseMeasure as Measure<X>>::RootMeasure: IIDRootMeasure<X>,
{
    /// Natural parameter: same as underlying distribution
    type NaturalParam = D::NaturalParam;

    /// Sufficient statistic: sum of individual sufficient statistics
    type SufficientStat = D::SufficientStat;

    /// Base measure: product of individual base measures
    type BaseMeasure = IIDBaseMeasure<D::BaseMeasure>;

    /// Cache: uses the underlying distribution's cache
    type Cache = D::Cache;

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self {
            distribution: D::from_natural(param),
        }
    }

    fn to_natural(&self) -> Self::NaturalParam {
        self.distribution.to_natural()
    }

    fn log_partition(&self) -> F {
        // For IID, this would be n·A(η), but we don't know n here.
        // This method should generally not be called directly for IID.
        // Instead, use compute_iid_exp_fam_log_density which handles the scaling.
        self.distribution.log_partition()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        // For IID, this would be (η, n·A(η)), but we don't know n here.
        // This method should generally not be called directly for IID.
        // Instead, use compute_iid_exp_fam_log_density which handles the scaling.
        self.distribution.natural_and_log_partition()
    }

    fn sufficient_statistic(&self, xs: &Vec<X>) -> Self::SufficientStat {
        // Compute ∑ᵢT(xᵢ) for the IID sample
        let individual_stats: Vec<D::SufficientStat> = xs
            .iter()
            .map(|x| self.distribution.sufficient_statistic(x))
            .collect();
        D::SufficientStat::sum_stats(&individual_stats)
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        IIDBaseMeasure::new(self.distribution.base_measure())
    }

    fn cached_log_density(&self, _cache: &Self::Cache, xs: &Vec<X>) -> F {
        // For IID, we need to use the specialized IID computation that handles 
        // the n·A(η) scaling properly. The cache here is for the underlying distribution.
        // We delegate to the centralized IID computation function.
        crate::exponential_family::traits::compute_iid_exp_fam_log_density(&self.distribution, xs)
    }
}
