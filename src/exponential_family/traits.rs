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

use crate::core::{Measure, True};
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
}


/// A marker trait for measures that are exponential families
///
/// This trait serves as a marker to identify exponential family distributions
/// and enables specialized implementations for density calculations.
pub trait ExponentialFamilyMeasure<X, F: Float>:
    Measure<X, IsExponentialFamily = True> + ExponentialFamily<X, F>
{
}

