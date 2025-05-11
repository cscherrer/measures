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
//! use measures::{Normal, LebesgueMeasure};
//! 
//! let normal = Normal::new(0.0, 1.0);
//! let lebesgue = LebesgueMeasure::new();
//! 
//! // Compute density
//! let density: f64 = normal.density(0.0).wrt(&lebesgue).into();
//! 
//! // Compute log-density (more efficient)
//! let log_density: f64 = normal.log_density(0.0).wrt(&lebesgue).into();
//! ```

use num_traits::Float;

/// A primitive measure that serves as a building block for more complex measures.
/// 
/// Primitive measures are the basic building blocks of our measure system.
/// They are typically simple measures like Lebesgue measure or counting measure
/// that can be used to construct more complex measures.
pub trait PrimitiveMeasure<T>: Clone {}

/// A measure that can compute its density with respect to some base measure.
/// 
/// Every measure has a root measure, which is the most natural base measure
/// for computing densities. For example:
/// - Normal distribution's root measure is Lebesgue measure
/// - Dirac measure's root measure is counting measure
pub trait Measure<T> {
    /// The root measure for this measure
    type RootMeasure: Measure<T>;

    /// Check if a point is in the support of this measure
    fn in_support(&self, x: T) -> bool;

    /// Get the root measure for this measure
    fn root_measure(&self) -> Self::RootMeasure;
}

/// A builder for computing densities of a measure.
/// 
/// This type is used to build up density computations. It doesn't compute
/// the density directly, but rather provides a way to specify the base measure
/// and then compute the density.
#[derive(Clone)]
pub struct Density<'a, T: Clone, M: Measure<T> + Clone> {
    /// The measure whose density we're computing
    pub measure: &'a M,
    /// The point at which to compute the density
    pub x: T,
}

/// A builder for computing densities of a measure with respect to a specific base measure.
/// 
/// This type represents a density computation that has been fully specified,
/// including the base measure. It can be converted into a f64 to get the actual
/// density value.
#[derive(Clone)]
pub struct DensityWRT<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> {
    /// The measure whose density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the density
    pub base_measure: &'a M2,
    /// The point at which to compute the density
    pub x: T,
}

/// A builder for computing log-densities of a measure.
/// 
/// Similar to `Density`, but for log-densities. This type is used to build up
/// log-density computations. It doesn't compute the log-density directly, but
/// rather provides a way to specify the base measure and then compute the log-density.
#[derive(Clone)]
pub struct LogDensity<'a, T: Clone, M: Measure<T> + Clone> {
    /// The measure whose log-density we're computing
    pub measure: &'a M,
    /// The point at which to compute the log-density
    pub x: T,
}

/// A builder for computing log-densities of a measure with respect to a specific base measure.
/// 
/// This type represents a log-density computation that has been fully specified,
/// including the base measure. It can be converted into a f64 to get the actual
/// log-density value.
#[derive(Clone)]
pub struct LogDensityWRT<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> {
    /// The measure whose log-density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the log-density
    pub base_measure: &'a M2,
    /// The point at which to compute the log-density
    pub x: T,
}

impl<'a, T: Clone, M: Measure<T> + Clone> Density<'a, T, M> {
    /// Create a new density computation.
    pub fn new(measure: &'a M, x: T) -> Self {
        Self { measure, x }
    }

    /// Specify the base measure for this density computation.
    /// 
    /// Returns a builder that can be converted into a f64 to get the actual
    /// density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> DensityWRT<'a, T, M, M2> {
        DensityWRT::new(self.measure, base_measure, self.x)
    }
}

impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> DensityWRT<'a, T, M1, M2> {
    /// Create a new density computation with respect to a base measure.
    pub fn new(measure: &'a M1, base_measure: &'a M2, x: T) -> Self {
        Self { measure, base_measure, x }
    }

    /// Compute the log of this density.
    /// 
    /// This is less efficient than computing the log-density directly,
    /// but is provided for convenience.
    pub fn log(&self) -> f64 
    where 
        Self: Into<f64> + Clone
    {
        let density = Into::<f64>::into(self.clone());
        density.ln()
    }
}

impl<'a, T: Clone, M: Measure<T> + Clone> LogDensity<'a, T, M> {
    /// Create a new log-density computation.
    pub fn new(measure: &'a M, x: T) -> Self {
        Self { measure, x }
    }

    /// Specify the base measure for this log-density computation.
    /// 
    /// Returns a builder that can be converted into a f64 to get the actual
    /// log-density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> LogDensityWRT<'a, T, M, M2> {
        LogDensityWRT::new(self.measure, base_measure, self.x)
    }
}

impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> LogDensityWRT<'a, T, M1, M2> {
    /// Create a new log-density computation with respect to a base measure.
    pub fn new(measure: &'a M1, base_measure: &'a M2, x: T) -> Self {
        Self { measure, base_measure, x }
    }

    /// Compute the exponential of this log-density to get the regular density.
    /// 
    /// This is provided for convenience when you need the regular density
    /// but have already computed the log-density.
    pub fn exp(&self) -> f64 
    where 
        Self: Into<f64> + Clone
    {
        let log_density = Into::<f64>::into(self.clone());
        log_density.exp()
    }
}

/// A trait for measures that can compute their density and log-density.
/// 
/// This trait provides methods to compute both the density and log-density of a measure.
/// The density/log-density is returned as a builder type that can be used to specify
/// the base measure and then compute the actual value.
/// 
/// # Implementation Note
/// 
/// Measures should implement this trait when they can compute their density or log-density.
/// If a measure can compute its density, it can always compute its log-density by taking
/// the log of the density. Similarly, if a measure can compute its log-density, it can
/// always compute its density by taking the exp of the log-density.
/// 
/// However, for efficiency, measures should implement the most efficient computation
/// path. For example, the normal distribution can compute its log-density directly
/// without computing the full density first.
pub trait HasDensity<T>: Measure<T> {
    /// Compute the density of this measure at a point.
    /// 
    /// Returns a builder that can be used to specify the base measure and
    /// then compute the actual density value.
    fn density(&self, x: T) -> Density<'_, T, Self> 
    where 
        Self: Sized + Clone,
        T: Clone
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
    fn log_density(&self, x: T) -> LogDensity<'_, T, Self>
    where 
        Self: Sized + Clone,
        T: Clone
    {
        LogDensity::new(self, x)
    }
}

// Implement HasDensity for all measures that can compute densities
impl<T: Clone, M: Measure<T> + Clone> HasDensity<T> for M {} 