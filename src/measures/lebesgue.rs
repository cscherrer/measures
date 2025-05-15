use crate::traits::{MeasureMarker, PrimitiveMeasure, True};
use num_traits::Float;

#[derive(Clone, Default)]
pub struct LebesgueMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Float> LebesgueMeasure<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> MeasureMarker for LebesgueMeasure<T> {
    type IsPrimitive = True;
}

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure<T> {}
