use crate::core::{False, MeasureMarker, Measure};
use num_traits::Float;
use std::marker::PhantomData;

/// A weighted measure that applies a weight function to a base measure.
/// 
/// This is useful for measures that can be expressed as a weight function
/// applied to a simpler base measure. The weight is expressed as a log-weight.
#[derive(Clone)]
pub struct WeightedMeasure<M, F, X = u64>
where
    M: Measure<X> + Clone,
    F: Float + Clone,
    X: Clone,
{
    /// The base measure
    pub base_measure: M,
    /// The log-weight value
    pub log_weight: F,
    /// Phantom data for X
    _phantom: PhantomData<X>,
}

impl<M, F, X> WeightedMeasure<M, F, X>
where
    M: Measure<X> + Clone,
    F: Float + Clone,
    X: Clone,
{
    /// Create a new weighted measure with the given base measure and log-weight.
    #[must_use]
    pub fn new(base_measure: M, log_weight: F) -> Self {
        Self { 
            base_measure, 
            log_weight,
            _phantom: PhantomData,
        }
    }
}

impl<M, F, X> MeasureMarker for WeightedMeasure<M, F, X>
where
    M: Measure<X> + Clone,
    F: Float + Clone,
    X: Clone,
{
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<M, F, X> Measure<X> for WeightedMeasure<M, F, X>
where
    M: Measure<X> + Clone,
    F: Float + Clone,
    X: Clone,
{
    type RootMeasure = M::RootMeasure;

    fn in_support(&self, x: X) -> bool {
        self.base_measure.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.base_measure.root_measure()
    }
} 