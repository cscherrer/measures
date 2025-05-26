use measures_core::{False, HasLogDensity, MeasureMarker, PrimitiveMeasure, True};
use num_traits::Zero;

#[derive(Clone, Default)]
pub struct CountingMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> CountingMeasure<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone> MeasureMarker for CountingMeasure<T> {
    type IsPrimitive = True;
    type IsExponentialFamily = False;
}

impl<T: Clone> PrimitiveMeasure<T> for CountingMeasure<T> {}

/// Primitive measures have log-density 0 with respect to themselves
/// since log(dm/dm) = log(1) = 0
impl<T: Clone, F: Zero> HasLogDensity<T, F> for CountingMeasure<T> {
    fn log_density_wrt_root(&self, _x: &T) -> F {
        F::zero()
    }
}
