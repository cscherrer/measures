use super::density::LogDensity;
use super::types::{True, TypeLevelBool};

/// A trait for measures that can indicate if they are primitive.
pub trait MeasureMarker {
    /// Type-level boolean indicating if this is a primitive measure.
    type IsPrimitive: TypeLevelBool;

    /// Type-level boolean indicating if this is an exponential family.
    type IsExponentialFamily: TypeLevelBool;
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

/// A measure that can compute its density with respect to some base measure.
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

    /// Create a log-density computation for this measure
    ///
    /// This provides the entry point for the fluent interface:
    /// `measure.log_density().wrt(base_measure).at(&x)`
    fn log_density(&self) -> LogDensity<T, Self, Self::RootMeasure>
    where
        T: Clone,
    {
        LogDensity::new(self.clone())
    }
}
