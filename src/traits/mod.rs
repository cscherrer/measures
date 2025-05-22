//! Core traits and types for working with measures and their densities.
//!
//! This module provides the foundational types for working with probability measures
//! and their densities. The design emphasizes:
//!
//! 1. Clear separation between measures and their densities
//! 2. Efficient computation of log-densities
//! 3. Type safety for density computations
//! 4. Flexibility in choosing base measures
//!
//! # Key Concepts
//!
//! - A **measure** is a mathematical object that assigns a non-negative value to sets
//! - A **density** is a function that describes how a measure relates to a base measure
//! - A **log-density** is the natural logarithm of a density, often more efficient to compute
//!
//! # Example
//!
//! ```rust
//! use measures::{Normal, Measure, HasDensity};
//!
//! let normal = Normal::new(0.0, 1.0);
//!
//! // Compute density
//! let density: f64 = normal.density(&0.0).into();
//!
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(&0.0).into();
//! ```

pub mod exponential_family;
pub mod dot_product;

pub trait TypeLevelBool {
    const VALUE: bool;
}

pub struct True;

pub struct False;

impl TypeLevelBool for True {
    const VALUE: bool = true;
}

impl TypeLevelBool for False {
    const VALUE: bool = false;
}

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
pub trait Measure<T>: MeasureMarker {
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
    /// Compute the density of this measure at a point.
    ///
    /// Returns a builder that can be used to specify the base measure and
    /// then compute the actual density value.
    fn density<'a>(&'a self, x: &'a T) -> Density<'a, T, Self>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        Density::new(self, x)
    }

    /// Compute the log-density of this measure at a point.
    ///
    /// Returns a builder that can be used to specify the base measure and
    /// then compute the actual log-density value.
    ///
    /// The default implementation computes the density and takes its log,
    /// but measures should override this with a more efficient implementation
    /// when possible.
    fn log_density<'a>(&'a self, x: &'a T) -> LogDensity<'a, T, Self>
    where
        Self: Sized + Clone,
        T: Clone,
    {
        LogDensity::new(self, x)
    }
}

/// A builder for computing densities of a measure.
///
/// This type is used to build up density computations. It can be in two states:
/// 1. Initial state: just the measure and point
/// 2. Final state: includes the base measure and can be converted to a f64
#[derive(Clone)]
pub struct Density<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone = M1> {
    /// The measure whose density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the density
    pub base_measure: Option<&'a M2>,
    /// The point at which to compute the density
    pub x: &'a T,
}

impl<'a, T: Clone, M1: Measure<T> + Clone> Density<'a, T, M1> {
    /// Create a new density computation.
    pub fn new(measure: &'a M1, x: &'a T) -> Self {
        Self {
            measure,
            base_measure: None,
            x,
        }
    }

    /// Specify the base measure for this density computation.
    ///
    /// Returns a builder that can be converted into a f64 to get the actual
    /// density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> Density<'a, T, M1, M2> {
        Density {
            measure: self.measure,
            base_measure: Some(base_measure),
            x: self.x,
        }
    }

    /// Convert a density to its logarithm
    #[must_use]
    pub fn log(self) -> LogDensity<'a, T, M1> {
        LogDensity {
            measure: self.measure,
            base_measure: None,
            x: self.x,
        }
    }
}

impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> Density<'a, T, M1, M2> {
    /// Convert a density with respect to a base measure to its logarithm
    #[must_use]
    pub fn log_wrt(self) -> LogDensity<'a, T, M1, M2> {
        LogDensity {
            measure: self.measure,
            base_measure: self.base_measure,
            x: self.x,
        }
    }
}

/// A builder for computing log-densities of a measure.
///
/// This type is used to build up log-density computations. It can be in two states:
/// 1. Initial state: just the measure and point
/// 2. Final state: includes the base measure and can be converted to a f64
#[derive(Clone)]
pub struct LogDensity<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone = M1> {
    /// The measure whose log-density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the log-density
    pub base_measure: Option<&'a M2>,
    /// The point at which to compute the log-density
    pub x: &'a T,
}

impl<'a, T: Clone, M1: Measure<T> + Clone> LogDensity<'a, T, M1> {
    /// Create a new log-density computation.
    pub fn new(measure: &'a M1, x: &'a T) -> Self {
        Self {
            measure,
            base_measure: None,
            x,
        }
    }

    /// Specify the base measure for this log-density computation.
    ///
    /// Returns a builder that can be converted into a f64 to get the actual
    /// log-density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> LogDensity<'a, T, M1, M2> {
        LogDensity {
            measure: self.measure,
            base_measure: Some(base_measure),
            x: self.x,
        }
    }
}
