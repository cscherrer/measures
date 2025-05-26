//! Helper implementations for exponential family densities.

use num_traits::Float;
use std::marker::PhantomData;

use super::traits::ExponentialFamilyMeasure;
use measures_core::DotProduct;
use measures_core::{False, LogDensity, Measure, MeasureMarker, True};

// Helper for converting LogDensity to f64 for exponential family measures
pub struct ExponentialFamilyDensity<X, F, M>(pub LogDensity<X, M>, PhantomData<(X, F)>)
where
    X: Clone,
    F: Float,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: DotProduct<Output = F>;

/// A wrapper type for exponential family measures that provides specialized
/// implementations optimized for exponential families.
///
/// This is a combinator that makes it easier to work with exponential family
/// measures and provides optimized density calculations.
#[derive(Clone)]
pub struct ExpFam<M, F = f64> {
    /// The underlying measure
    pub measure: M,
    /// Phantom data to bind the field type
    _phantom: PhantomData<F>,
}

impl<M, F> ExpFam<M, F> {
    /// Create a new exponential family wrapper for the given measure.
    #[must_use]
    pub fn new(measure: M) -> Self {
        Self {
            measure,
            _phantom: PhantomData,
        }
    }
}

// Use separate type parameters for X and F
impl<X, F, M> Measure<X> for ExpFam<M, F>
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
{
    type RootMeasure = M::RootMeasure;

    fn in_support(&self, x: X) -> bool {
        self.measure.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.measure.root_measure()
    }
}

impl<M, F> MeasureMarker for ExpFam<M, F>
where
    F: Float,
    M: MeasureMarker,
{
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}
