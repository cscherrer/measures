use crate::measures::counting::CountingMeasure;
use crate::traits::{Density, False, HasDensity, LogDensity, Measure, MeasureMarker};

#[derive(Clone)]
pub struct Dirac<T: PartialEq> {
    pub x: T,
}

impl<T: PartialEq + Clone> Dirac<T> {
    pub fn new(x: T) -> Self {
        Self { x }
    }
}

impl<T: PartialEq + Clone> MeasureMarker for Dirac<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<T: PartialEq + Clone> Measure<T> for Dirac<T> {
    type RootMeasure = CountingMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        self.x == x
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::new()
    }
}

// Implement HasDensity for Dirac measure
impl<T: PartialEq + Clone> HasDensity<T> for Dirac<T> {
    // Empty implementation - functionality provided by the From impls below
}

// Implement specific density calculations for Dirac measure
impl<T: PartialEq + Clone> From<Density<'_, T, Dirac<T>>> for f64 {
    fn from(val: Density<'_, T, Dirac<T>>) -> Self {
        if val.measure.x == *val.x { 1.0 } else { 0.0 }
    }
}

impl<T: PartialEq + Clone> From<Density<'_, T, Dirac<T>, CountingMeasure<T>>> for f64 {
    fn from(val: Density<'_, T, Dirac<T>, CountingMeasure<T>>) -> Self {
        if val.measure.x == *val.x { 1.0 } else { 0.0 }
    }
}

// Implement conversion from LogDensity to f64 for Dirac measure
impl<T: PartialEq + Clone> From<LogDensity<'_, T, Dirac<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Dirac<T>>) -> Self {
        if val.measure.x == *val.x {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl<T: PartialEq + Clone> From<LogDensity<'_, T, Dirac<T>, CountingMeasure<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Dirac<T>, CountingMeasure<T>>) -> Self {
        if val.measure.x == *val.x {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}
