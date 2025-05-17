//! Helper implementations for exponential family densities.

use num_traits::Float;
use std::marker::PhantomData;

use super::traits::{ExponentialFamily, ExponentialFamilyMeasure, InnerProduct};
use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};

// Helper function to calculate log-density for any exponential family measure
pub fn exp_fam_log_density<'a, X, F, M>(measure: &'a M, x: &'a X) -> LogDensity<'a, X, M>
where
    F: Float,
    X: Clone,
    M: ExponentialFamily<X, F> + Measure<X> + Clone,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    measure.log_density_ef(x)
}

// Helper for converting LogDensity to f64 for exponential family measures
pub struct ExponentialFamilyDensity<'a, X, F, M>(pub LogDensity<'a, X, M>, PhantomData<(X, F)>)
where
    X: Clone,
    F: Float,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>;

impl<'a, X, F, M> From<ExponentialFamilyDensity<'a, X, F, M>> for f64
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    fn from(wrapper: ExponentialFamilyDensity<'a, X, F, M>) -> Self {
        let val = wrapper.0;
        let eta = val.measure.to_natural();
        let t = val.measure.sufficient_statistic(val.x);
        let a = val.measure.log_partition();
        let h = val.measure.carrier_measure(val.x);

        let result = eta.inner_product(&t) - a + h.ln();
        result.to_f64().unwrap()
    }
}

// Helper function to compute exponential family log density
#[must_use]
pub fn compute_exp_fam_log_density<X, F, M>(log_density: LogDensity<'_, X, M>) -> f64
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    ExponentialFamilyDensity::<X, F, M>(log_density, PhantomData).into()
}

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

impl<X, F, M> HasDensity<X> for ExpFam<M, F>
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    // Empty implementation as the From impls below provide the actual computation
}

// Implement From for LogDensity to f64 (specialized computation)
impl<X, F, M> From<LogDensity<'_, X, ExpFam<M, F>>> for f64
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    fn from(val: LogDensity<'_, X, ExpFam<M, F>>) -> Self {
        // Compute using exponential family form for better performance
        let eta = val.measure.measure.to_natural();
        let t = val.measure.measure.sufficient_statistic(val.x);
        let a = val.measure.measure.log_partition();
        let h = val.measure.measure.carrier_measure(val.x);

        let result = eta.inner_product(&t) - a + h.ln();
        result.to_f64().unwrap()
    }
}
