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

/// A helper trait for exponential family distributions to compute densities
/// Types can use this to implement `HasDensity` without repetitive code.
pub trait ExpFamDensity<T: Float>: ExponentialFamily<T> + Measure<T> {
    /// Compute log-density using exponential family form
    fn compute_log_density<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self>
    where
        Self: Sized + Clone,
        Self::NaturalParam: DotProduct<Self::SufficientStat, T>,
        T: Clone,
    {
        // Return the LogDensity directly
        self.log_density_ef(x)
    }
}

// Helper function to calculate log-density for any exponential family measure
pub fn exp_fam_log_density<'a, T: Float, M>(measure: &'a M, x: &'a T) -> LogDensity<'a, T, M>
where
    M: ExponentialFamily<T> + Measure<T> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    measure.log_density_ef(x)
}

// We'll use a specialization helper to avoid conflicts with dirac implementation
pub struct ExponentialFamilyDensity<'a, T: Float, M>(pub LogDensity<'a, T, M>, PhantomData<M>)
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>;

impl<'a, T: Float, M> From<ExponentialFamilyDensity<'a, T, M>> for f64
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    fn from(wrapper: ExponentialFamilyDensity<'a, T, M>) -> Self {
        let val = wrapper.0;
        let eta = val.measure.to_natural();
        let t = val.measure.sufficient_statistic(val.x);
        let a = val.measure.log_partition();
        let h = val.measure.carrier_measure(val.x);

        let result =
            <M::NaturalParam as DotProduct<M::SufficientStat, T>>::dot(&eta, &t) - a + h.ln();
        result.to_f64().unwrap()
    }
}

// Helper function to compute exponential family log density
pub fn compute_exp_fam_log_density<T: Float, M>(log_density: LogDensity<'_, T, M>) -> f64
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    ExponentialFamilyDensity(log_density, PhantomData).into()
}
