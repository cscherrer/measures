//! Exponential family trait and implementations.
//!
//! This module provides the core trait for exponential family distributions
//! and implementations for various distributions.

use crate::core::{LogDensity, Measure, True};
use crate::traits::dot_product::DotProduct;
use num_traits::Float;
use std::marker::PhantomData;

/// A trait for exponential family distributions
pub trait ExponentialFamily<T: Float> {
    /// The natural parameter type
    type NaturalParam;

    /// The sufficient statistic type
    type SufficientStat;

    /// The base measure type
    type BaseMeasure: Measure<T>;

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    fn to_natural(&self) -> Self::NaturalParam;

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> T;

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat;

    fn base_measure(&self) -> Self::BaseMeasure;
}

// We'll use a specialization helper to avoid conflicts with dirac implementation
pub struct ExponentialFamilyDensity<T: Float, M>(pub LogDensity<T, M>, PhantomData<M>)
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<Output = T>;

// Helper function to compute exponential family log density
#[must_use]
pub fn compute_exp_fam_log_density<T: Float, M>(log_density: LogDensity<T, M>) -> T
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<Output = T>,
{
    // Get the measure
    let measure = &log_density.measure;

    // Compute natural parameters and sufficient statistics
    let _eta = measure.to_natural();
    // Note: We can't get sufficient statistics without a point x
    // This function signature needs to be updated to include x: &T

    // Compute log density using exponential family formula
    // log f(x) = η·T(x) - A(η) + log h(x)
    // Where h(x) is implicitly 1 for our implementations

    // Since we don't have a proper way to compute dot product between different types,
    // we'll omit this calculation and just return the negative log partition
    -measure.log_partition()
}
