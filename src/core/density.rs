use super::measure::Measure;

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

impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> LogDensity<'a, T, M1, M2> {
    /// Compute the log-density using the exponential family form, if the
    /// measure is an exponential family measure.
    ///
    /// This is a performance optimization that allows specialized computation
    /// when we know we're working with an exponential family.
    #[must_use]
    pub fn compute_exp_fam_form(self) -> f64
    where
        T: num_traits::Float,
        M1: crate::exponential_family::ExponentialFamilyMeasure<T, T>,
        M1::NaturalParam: crate::exponential_family::InnerProduct<M1::SufficientStat, T>,
        M2: Default,
    {
        // We need to convert from LogDensity<'a, T, M1, M2> to LogDensity<'a, T, M1>
        // which is what the compute_exp_fam_log_density function expects
        let log_density = LogDensity::<'a, T, M1> {
            measure: self.measure,
            base_measure: None,
            x: self.x,
        };

        crate::exponential_family::compute_exp_fam_log_density(log_density)
    }
}

// General From implementation for exponential family measures removed
// to avoid conflict with specific implementations
