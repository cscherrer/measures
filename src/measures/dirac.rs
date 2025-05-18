use crate::core::types::Specialized;
use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker};
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
    type PreferredLogDensityMethod = Specialized;
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

// Implement conversion from LogDensity to f64 for Dirac measure
// We only need one implementation that works for both cases
impl<T: PartialEq + Clone, M: Measure<T>> From<LogDensity<'_, T, Dirac<T>, M>> for f64 {
    fn from(val: LogDensity<'_, T, Dirac<T>, M>) -> Self {
        if val.measure.point == *val.x {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }
}
