mod traits;
use traits::{PrimitiveMeasure, Measure, Density};
use num_traits::Float;

#[derive(Clone)]
pub struct LebesgueMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure<T> {}

#[derive(Clone)]
pub struct CountingMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> PrimitiveMeasure<T> for CountingMeasure<T> {}

impl<T, P: PrimitiveMeasure<T>> Measure<T> for P {
    type RootMeasure = Self;

    fn in_support(&self, x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.clone()
    }
}

pub struct Dirac<T: PartialEq> {
    x: T,
}

impl<T: PartialEq + Clone> Measure<T> for Dirac<T> {
    type RootMeasure = CountingMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        self.x == x
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<T> {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: PartialEq + Clone> Density<T> for Dirac<T> {
    type BaseMeasure = CountingMeasure<T>;

    fn log_density(&self, x: T) -> f64 {
        0.0
    }

    fn density(&self, x: T) -> f64 {
        1.0
    }
}
