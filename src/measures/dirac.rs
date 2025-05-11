use crate::traits::{PrimitiveMeasure, Measure, Density, DensityWRT};
use crate::measures::counting::CountingMeasure;

#[derive(Clone)]
pub struct Dirac<T: PartialEq> {
    pub x: T,
}

impl<T: PartialEq + Clone> Dirac<T> {
    pub fn new(x: T) -> Self {
        Self { x }
    }
}

impl<T: PartialEq + Clone> PrimitiveMeasure<T> for Dirac<T> {}

impl<T: PartialEq + Clone> Measure<T> for Dirac<T> {
    type RootMeasure = CountingMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        self.x == x
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::new()
    }
}

// Implement specific density calculations for Dirac measure
impl<'a, T: PartialEq + Clone> Into<f64> for Density<'a, T, Dirac<T>> {
    fn into(self) -> f64 {
        if self.measure.x == self.x {
            1.0
        } else {
            0.0
        }
    }
}

impl<'a, T: PartialEq + Clone> Into<f64> for DensityWRT<'a, T, Dirac<T>, CountingMeasure<T>> {
    fn into(self) -> f64 {
        if self.measure.x == self.x {
            1.0
        } else {
            0.0
        }
    }
} 