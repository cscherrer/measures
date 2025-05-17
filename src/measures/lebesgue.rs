use crate::core::{False, MeasureMarker, PrimitiveMeasure, True};
use num_traits::Float;

#[derive(Clone, Default)]
pub struct LebesgueMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> LebesgueMeasure<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> MeasureMarker for LebesgueMeasure<T> {
    type IsPrimitive = True;
    type IsExponentialFamily = False;
}

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure<T> {}
