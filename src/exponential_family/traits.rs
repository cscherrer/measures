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

use crate::core::{LogDensity, Measure, True};
use num_traits::Float;

/// Trait for inner product operations between natural parameters and sufficient statistics.
/// This replaces the more restrictive `DotProduct` with a more general operation.
pub trait InnerProduct<Rhs, F: Float> {
    /// Compute the inner product between self and rhs, returning a value in field F.
    fn inner_product(&self, rhs: &Rhs) -> F;
}

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

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    fn to_natural(&self) -> Self::NaturalParam;

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> F;

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;

    /// Compute the carrier measure h(x)
    fn carrier_measure(&self, x: &X) -> F;

    /// Compute the log density in exponential family form
    fn log_density_ef<'a>(&'a self, x: &'a X) -> LogDensity<'a, X, Self>
    where
        Self: Sized + Clone + Measure<X>,
        X: Clone,
    {
        LogDensity::new(self, x)
    }
}

/// Implementation for array-based inner products of any dimension
impl<F, const N: usize> InnerProduct<[F; N], F> for [F; N]
where
    F: Float + std::iter::Sum,
{
    fn inner_product(&self, rhs: &[F; N]) -> F {
        self.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).sum()
    }
}

/// A helper trait for exponential family distributions to compute densities
/// Types can use this to implement `HasDensity` without repetitive code.
pub trait ExpFamDensity<X, F: Float>: ExponentialFamily<X, F> + Measure<X> {
    /// Compute log-density using exponential family form
    fn compute_log_density<'a>(&'a self, x: &'a X) -> LogDensity<'a, X, Self>
    where
        Self: Sized + Clone,
        Self::NaturalParam: InnerProduct<Self::SufficientStat, F>,
        X: Clone,
    {
        // Return the LogDensity directly
        self.log_density_ef(x)
    }
}

/// A marker trait for measures that are exponential families
///
/// This trait serves as a marker to identify exponential family distributions
/// and enables specialized implementations for density calculations.
pub trait ExponentialFamilyMeasure<X, F: Float>:
    Measure<X, IsExponentialFamily = True> + ExponentialFamily<X, F> + Clone
{
}
