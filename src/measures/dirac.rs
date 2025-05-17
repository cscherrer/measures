use crate::core::{Density, False, HasDensity, LogDensity, Measure, MeasureMarker};
use crate::measures::counting::CountingMeasure;

#[derive(Clone)]
pub struct Dirac<T: Clone> {
    point: T,
}

impl<T: Clone> Dirac<T> {
    #[must_use]
    pub fn new(point: T) -> Self {
        Self { point }
    }
}

impl<T: Clone> MeasureMarker for Dirac<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<T: Clone + PartialEq> Measure<T> for Dirac<T> {
    type RootMeasure = CountingMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        x == self.point
    }

    fn root_measure(&self) -> CountingMeasure<T> {
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
        if val.measure.point == *val.x {
            1.0
        } else {
            0.0
        }
    }
}

impl<T: PartialEq + Clone> From<Density<'_, T, Dirac<T>, CountingMeasure<T>>> for f64 {
    fn from(val: Density<'_, T, Dirac<T>, CountingMeasure<T>>) -> Self {
        if val.measure.point == *val.x {
            1.0
        } else {
            0.0
        }
    }
}

// Implement conversion from LogDensity to f64 for Dirac measure
impl<T: PartialEq + Clone> From<LogDensity<'_, T, Dirac<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Dirac<T>>) -> Self {
        if val.measure.point == *val.x {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl<T: PartialEq + Clone> From<LogDensity<'_, T, Dirac<T>, CountingMeasure<T>>> for f64 {
    fn from(val: LogDensity<'_, T, Dirac<T>, CountingMeasure<T>>) -> Self {
        if val.measure.point == *val.x {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}
