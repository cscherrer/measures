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
