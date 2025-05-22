//! Exponential family trait and implementations.
//!
//! This module provides the core trait for exponential family distributions
//! and implementations for various distributions.

use crate::traits::{LogDensity, Measure, True};
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
pub struct ExponentialFamilyDensity<'a, T: Float, M>(pub LogDensity<'a, T, M>, PhantomData<M>)
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>;


// Helper function to compute exponential family log density
pub fn compute_exp_fam_log_density<T: Float, M>(log_density: LogDensity<'_, T, M>) -> f64
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    ExponentialFamilyDensity(log_density, PhantomData).into()
}
