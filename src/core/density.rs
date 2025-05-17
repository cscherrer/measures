use super::measure::Measure;

/// A builder for computing densities of a measure.
///
/// This type is used to build up density computations. It can be in two states:
/// 1. Initial state: just the measure and point
/// 2. Final state: includes the base measure and can be converted to a f64
#[derive(Clone)]
pub struct Density<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone = M1> {
    /// The measure whose density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the density
    pub base_measure: Option<&'a M2>,
    /// The point at which to compute the density
    pub x: &'a T,
}

impl<'a, T: Clone, M1: Measure<T> + Clone> Density<'a, T, M1> {
    /// Create a new density computation.
    pub fn new(measure: &'a M1, x: &'a T) -> Self {
        Self {
            measure,
            base_measure: None,
            x,
        }
    }

    /// Specify the base measure for this density computation.
    ///
    /// Returns a builder that can be converted into a f64 to get the actual
    /// density value.
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> Density<'a, T, M1, M2> {
        Density {
            measure: self.measure,
            base_measure: Some(base_measure),
            x: self.x,
        }
    }

    /// Convert a density to its logarithm
    #[must_use]
    pub fn log(self) -> LogDensity<'a, T, M1> {
        LogDensity {
            measure: self.measure,
            base_measure: None,
            x: self.x,
        }
    }
}

impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> Density<'a, T, M1, M2> {
    /// Convert a density with respect to a base measure to its logarithm
    #[must_use]
    pub fn log_wrt(self) -> LogDensity<'a, T, M1, M2> {
        LogDensity {
            measure: self.measure,
            base_measure: self.base_measure,
            x: self.x,
        }
    }
}

/// A builder for computing log-densities of a measure.
///
/// This type is used to build up log-density computations. It can be in two states:
/// 1. Initial state: just the measure and point
/// 2. Final state: includes the base measure and can be converted to a f64
#[derive(Clone)]
pub struct LogDensity<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone = M1> {
    /// The measure whose log-density we're computing
    pub measure: &'a M1,
    /// The base measure with respect to which we're computing the log-density
    pub base_measure: Option<&'a M2>,
    /// The point at which to compute the log-density
    pub x: &'a T,
}

impl<'a, T: Clone, M1: Measure<T> + Clone> LogDensity<'a, T, M1> {
    /// Create a new log-density computation.
    pub fn new(measure: &'a M1, x: &'a T) -> Self {
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
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> LogDensity<'a, T, M1, M2> {
        LogDensity {
            measure: self.measure,
            base_measure: Some(base_measure),
            x: self.x,
        }
    }
}
