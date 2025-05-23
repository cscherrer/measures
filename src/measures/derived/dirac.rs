use crate::core::{False, Measure, MeasureMarker};
use crate::measures::primitive::counting::CountingMeasure;

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
