use crate::traits::{PrimitiveMeasure, Measure};

#[derive(Clone)]
pub struct CountingMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> CountingMeasure<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone> PrimitiveMeasure<T> for CountingMeasure<T> {}

impl<T: Clone> Measure<T> for CountingMeasure<T> {
    type RootMeasure = Self;

    fn in_support(&self, x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.clone()
    }
} 