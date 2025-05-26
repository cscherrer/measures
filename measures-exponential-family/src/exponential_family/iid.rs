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

use crate::exponential_family::traits::{ExponentialFamily, SumSufficientStats};
use measures_combinators::measures::derived::binomial_coefficient::BinomialCoefficientMeasure;
use measures_combinators::measures::derived::factorial::FactorialMeasure;
use measures_combinators::measures::derived::negative_binomial_coefficient::NegativeBinomialCoefficientMeasure;
use measures_core::DotProduct;
use measures_core::float_constant;
use measures_core::primitive::counting::CountingMeasure;
use measures_core::primitive::lebesgue::LebesgueMeasure;
use measures_core::{False, True};
use measures_core::{HasLogDensity, Measure, MeasureMarker};
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

/// Implementation for `FactorialMeasure`
impl<X: Clone, F: Float> IIDRootMeasure<X> for FactorialMeasure<F> {
    type IIDRoot = CountingMeasure<Vec<X>>;

    fn iid_root_measure() -> Self::IIDRoot {
        CountingMeasure::new()
    }
}

/// Implementation for `BinomialCoefficientMeasure`
impl<X: Clone, F: Float> IIDRootMeasure<X> for BinomialCoefficientMeasure<F> {
    type IIDRoot = CountingMeasure<Vec<X>>;

    fn iid_root_measure() -> Self::IIDRoot {
        CountingMeasure::new()
    }
}

/// Implementation for `NegativeBinomialCoefficientMeasure`
impl<X: Clone, F: Float> IIDRootMeasure<X> for NegativeBinomialCoefficientMeasure<F> {
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
        F: Float + std::iter::Sum,
        D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
        D::SufficientStat: SumSufficientStats<D::SufficientStat>,
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
        F: Float + std::iter::Sum,
    {
        xs.iter()
            .map(|x| self.distribution.log_density_wrt_root(x))
            .sum()
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
    F: Float + std::iter::Sum,
{
    fn log_density_wrt_root(&self, xs: &Vec<X>) -> F {
        // Product of individual base measures: ∑ᵢ log h(xᵢ)
        xs.iter()
            .map(|x| self.base_measure.log_density_wrt_root(x))
            .sum()
    }
}

/// IID exponential family implementation
///
/// For an exponential family p(x|θ) = h(x) exp(η·T(x) - A(η)), the joint distribution
/// of n iid samples X₁, X₂, ..., Xₙ is also an exponential family:
/// p(x₁,...,xₙ|θ) = ∏ᵢ h(xᵢ) exp(η·∑ᵢT(xᵢ) - n·A(η))
impl<D, X, F> ExponentialFamily<Vec<X>, F> for IID<D>
where
    D: ExponentialFamily<X, F> + Clone,
    X: Clone,
    F: Float + std::iter::Sum,
    D::SufficientStat: SumSufficientStats<D::SufficientStat>,
    D::BaseMeasure: HasLogDensity<X, F> + Measure<X>,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F> + Clone,
    <D::BaseMeasure as Measure<X>>::RootMeasure: IIDRootMeasure<X>,
{
    /// Same natural parameter as the underlying distribution
    type NaturalParam = D::NaturalParam;

    /// Same sufficient statistic type (will be summed)
    type SufficientStat = D::SufficientStat;

    /// Product base measure
    type BaseMeasure = IIDBaseMeasure<D::BaseMeasure>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let underlying = D::from_natural(param);
        IID::new(underlying)
    }

    fn sufficient_statistic(&self, xs: &Vec<X>) -> Self::SufficientStat {
        // Sum of individual sufficient statistics: ∑ᵢT(xᵢ)
        D::SufficientStat::sum_sufficient_stats(
            xs.iter().map(|x| self.distribution.sufficient_statistic(x)),
        )
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        IIDBaseMeasure::new(self.distribution.base_measure())
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        let (base_natural_params, base_log_partition) =
            self.distribution.natural_and_log_partition();

        // For IID, natural parameters are the same as base distribution
        let natural_params = base_natural_params;

        // Log partition is n * A(η) where n is the sample size
        // For the trait implementation, we use n=1 as a placeholder
        let n = float_constant::<F>(1.0); // This will be multiplied by actual sample size in usage
        let log_partition = n * base_log_partition;

        (natural_params, log_partition)
    }

    /// Override `exp_fam_log_density` to handle IID sample size scaling correctly
    ///
    /// This enables the standard API: `iid_normal.log_density().at(&samples)`
    /// The automatic `HasLogDensity` implementation will call this method.
    fn exp_fam_log_density(&self, xs: &Vec<X>) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<Vec<X>, F>,
    {
        // Use the efficient IID exponential family computation
        crate::exponential_family::traits::compute_iid_exp_fam_log_density(&self.distribution, xs)
    }
}

/// Implementation of `HasLogDensity` for IID distributions
impl<D, X, F> HasLogDensity<Vec<X>, F> for IID<D>
where
    D: ExponentialFamily<X, F>,
    X: Clone,
    F: Float + std::iter::Sum,
    D::NaturalParam: DotProduct<D::SufficientStat, Output = F>,
    D::SufficientStat: SumSufficientStats<D::SufficientStat>,
    D::BaseMeasure: HasLogDensity<X, F>,
{
    fn log_density_wrt_root(&self, xs: &Vec<X>) -> F {
        self.iid_log_density(xs)
    }
}
