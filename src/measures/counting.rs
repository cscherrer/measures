use crate::core::types::Default as DefaultMethod;
use crate::core::{False, MeasureMarker, PrimitiveMeasure, True};

#[derive(Clone, Default)]
pub struct CountingMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> CountingMeasure<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone> MeasureMarker for CountingMeasure<T> {
    type IsPrimitive = True;
    type IsExponentialFamily = False;
    type PreferredLogDensityMethod = DefaultMethod;
}

impl<T: Clone> PrimitiveMeasure<T> for CountingMeasure<T> {}
