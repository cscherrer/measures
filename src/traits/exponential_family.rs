//! Exponential family trait and implementations.
//!
//! This module provides the core trait for exponential family distributions
//! and implementations for various distributions.

use crate::traits::{HasDensity, LogDensity, Measure};
use num_traits::Float;

/// A trait for exponential family distributions
pub trait ExponentialFamily<T: Float> {
    /// The natural parameter type
    type NaturalParam;

    /// The sufficient statistic type
    type SufficientStat;

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    fn to_natural(&self) -> Self::NaturalParam;

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> T;

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat;

    /// Compute the carrier measure h(x)
    fn carrier_measure(&self, x: &T) -> T;

    /// Compute the log density in exponential family form
    fn log_density_ef(&self, x: &T) -> T
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, T>,
    {
        let eta = self.to_natural();
        let t = self.sufficient_statistic(x);
        let a = self.log_partition();
        let h = self.carrier_measure(x);

        <Self::NaturalParam as DotProduct<Self::SufficientStat, T>>::dot(&eta, &t) - a + h.ln()
    }
}

/// Extension trait for dot product operations
pub trait DotProduct<Rhs, T> {
    fn dot(lhs: &Self, rhs: &Rhs) -> T;
}

pub trait ExpFamLogDensity<T: Float>: ExponentialFamily<T> {
    fn log_density(&self, x: &T) -> T
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, T>,
    {
        self.log_density_ef(x)
    }
}

// Implement HasDensity for all exponential family distributions
impl<T: Float, M> HasDensity<T> for M
where
    M: ExponentialFamily<T> + Measure<T> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    fn log_density<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self> {
        LogDensity::new(self, x)
    }
}

// Implement From for LogDensity to f64 for exponential family distributions
impl<T: Float, M> From<LogDensity<'_, T, M>> for f64
where
    M: ExponentialFamily<T> + Measure<T> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    fn from(val: LogDensity<'_, T, M>) -> Self {
        val.measure.log_density_ef(val.x).to_f64().unwrap()
    }
}
