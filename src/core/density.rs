use super::measure::Measure;


pub trait LogDensityTrait<T>
{
    type Measure: Measure<T>;
    type BaseMeasure: Measure<T>;

    fn measure(&self) -> &Self::Measure;
    fn base_measure(&self) -> &Self::BaseMeasure;


}




/// A builder for computing log-densities of a measure.
///
/// This type is used to build up log-density computations. It can be in two states:
/// 1. Initial state: just the measure and point
/// 2. Final state: includes the base measure and can be converted to a f64
#[derive(Clone)]
pub struct LogDensity<
    'a,
    T: Clone,
    M1: Measure<T> + Clone,
    M2: Measure<T> + Clone = <M1 as Measure<T>>::RootMeasure,
> {
    /// The measure whose log-density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the log-density
    pub base_measure: Option<&'a M2>,
    /// The point at which to compute the log-density
    pub x: &'a T,
}

impl<'a, T, M, R> LogDensity<'a, T, M, R>
where
    T: Clone,
    M: Measure<T, RootMeasure = R>,
    R: Measure<T>,
{
    /// Create a new log-density computation.
    pub fn new(measure: &'a M, x: &'a T) -> Self {
        Self {
            measure,
            base_measure: None,
            x,
        }
    }

    /// Specify the base measure for this log-density computation.
    ///
    /// Returns a builder that can be converted into a f64 to get the actual
    /// log-density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> LogDensity<'a, T, M, M2> {
        LogDensity {
            measure: self.measure,
            base_measure: Some(base_measure),
            x: self.x,
        }
    }

}
