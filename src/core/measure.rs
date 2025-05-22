use super::types::{LogDensityMethod, True, TypeLevelBool};

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
}

/// A trait for measures that can compute their density and log-density.
///
/// This trait extends `Measure` to add methods for density computation.
/// Separating this from `Measure` allows for specialized implementation strategies
/// like exponential family computations.
pub trait HasDensity<T>: Measure<T> {
    /// Compute the log-density of this measure at a point using the default method.
    ///
    /// Returns a builder that can be used to specify the base measure and
    /// then compute the actual log-density value.
    fn log_density<'a>(
        &'a self,
        x: &'a T,
    ) -> super::density::LogDensity<'a, T, Self, Self::RootMeasure>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        // Default implementation just creates a new log density object
        super::density::LogDensity::new(self, x)
    }

    /// Compute the log-density using the specialized implementation.
    ///
    /// This should be overridden by distributions that have specialized
    /// log-density computations.
    fn log_density_specialized<'a>(
        &'a self,
        x: &'a T,
    ) -> super::density::LogDensity<'a, T, Self, Self::RootMeasure>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        // Default implementation falls back to the default behavior
        super::density::LogDensity::new(self, x)
    }

    /// Compute the log-density using the exponential family form.
    ///
    /// This should be used by distributions that are exponential families.
    fn log_density_ef<'a>(
        &'a self,
        x: &'a T,
    ) -> super::density::LogDensity<'a, T, Self, Self::RootMeasure>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        // Default implementation falls back to the default behavior
        super::density::LogDensity::new(self, x)
    }
}
