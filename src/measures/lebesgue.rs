use crate::traits::{Measure, PrimitiveMeasure};
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
