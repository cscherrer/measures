use crate::traits::{PrimitiveMeasure, Measure};
use num_traits::Float;

#[derive(Clone)]
pub struct LebesgueMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Float> LebesgueMeasure<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure<T> {}

impl<T: Float> Measure<T> for LebesgueMeasure<T> {
    type RootMeasure = Self;

    fn in_support(&self, x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.clone()
    }
} 