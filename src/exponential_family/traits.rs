//! Exponential family trait definitions.
//!
//! This module provides the core trait for exponential family distributions.

use crate::core::{LogDensity, Measure};
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
    fn log_density_ef<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self>
    where
        Self: Sized + Clone + Measure<T>,
        T: Clone,
    {
        LogDensity::new(self, x)
    }
}

/// Extension trait for dot product operations
pub trait DotProduct<Rhs, T> {
    fn dot(lhs: &Self, rhs: &Rhs) -> T;
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
