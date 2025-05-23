use super::density::{HasLogDensity, LogDensity};
use super::types::{True, TypeLevelBool};
use num_traits::Zero;

/// A trait for measures that can indicate their properties.
pub trait MeasureMarker {
    /// Type-level boolean indicating if this is a primitive measure.
    type IsPrimitive: TypeLevelBool;

    /// Type-level boolean indicating if this is an exponential family.
    type IsExponentialFamily: TypeLevelBool;
}

/// The core trait for probability measures.
///
/// Every measure has a root measure, which is the most natural base measure
/// for computing densities. For example:
/// - Normal distribution's root measure is Lebesgue measure
/// - Dirac measure's root measure is counting measure
pub trait Measure<T>: MeasureMarker + Clone {
    /// The root measure for this measure
    type RootMeasure: Measure<T>;

    /// Check if a point is in the support of this measure
    fn in_support(&self, x: T) -> bool;

    /// Get the root measure for this measure
    fn root_measure(&self) -> Self::RootMeasure;
}

/// A primitive measure that serves as a building block for more complex measures.
///
/// Primitive measures are the basic building blocks of our measure system.
/// They are typically simple measures like Lebesgue measure or counting measure
/// that can be used to construct more complex measures.
pub trait PrimitiveMeasure<T>: Clone + MeasureMarker<IsPrimitive = True> {}

impl<P: PrimitiveMeasure<T>, T: Clone> Measure<T> for P {
    type RootMeasure = Self;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.clone()
    }
}

/// Automatic implementation: primitive measures have log-density 0 with respect to themselves
/// since log(dm/dm) = log(1) = 0
impl<P, T, F> HasLogDensity<T, F> for P
where
    P: PrimitiveMeasure<T>,
    T: Clone,
    F: Zero,
{
    fn log_density_wrt_root(&self, _x: &T) -> F {
        F::zero()
    }
}

/// Extension trait that provides log-density computation for all measures
pub trait LogDensityBuilder<T: Clone>: Measure<T> {
    /// Create a log-density computation for this measure
    fn log_density(&self) -> LogDensity<T, Self, Self::RootMeasure> {
        LogDensity::new(self.clone())
    }
}

// Automatically implement LogDensityBuilder for all measures
impl<T: Clone, M> LogDensityBuilder<T> for M where M: Measure<T> {}
