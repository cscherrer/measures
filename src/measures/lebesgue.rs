use crate::core::types::Default as DefaultMethod;
use crate::core::{False, MeasureMarker, PrimitiveMeasure, True};

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

impl<T: Clone> MeasureMarker for LebesgueMeasure<T> {
    type IsPrimitive = True;
    type IsExponentialFamily = False;
}

impl<T: Clone> PrimitiveMeasure<T> for LebesgueMeasure<T> {}
