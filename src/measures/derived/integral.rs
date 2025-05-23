use crate::core::{False, HasLogDensity, Measure, MeasureMarker};
use num_traits::Float;

/// A measure constructed by integrating a log-density function against a base measure.
///
/// This represents the measure ν such that dν = exp(f) dμ, where:
/// - μ is the base measure
/// - f is the log-density function  
/// - ν is the resulting integral measure
///
/// This is the inverse operation of taking a Radon-Nikodym derivative.
#[derive(Clone)]
pub struct IntegralMeasure<M, LogDensityFn, X>
where
    M: Measure<X> + Clone,
    LogDensityFn: Clone,
    X: Clone,
{
    /// The base measure
    pub base_measure: M,
    /// The log-density function
    pub log_density_fn: LogDensityFn,
    /// Phantom data for X
    _phantom: std::marker::PhantomData<X>,
}

impl<M, LogDensityFn, X> IntegralMeasure<M, LogDensityFn, X>
where
    M: Measure<X> + Clone,
    LogDensityFn: Clone,
    X: Clone,
{
    /// Create a new integral measure from a base measure and log-density function.
    ///
    /// # Arguments
    /// * `base_measure` - The base measure μ
    /// * `log_density_fn` - Function f such that dν = exp(f) dμ
    #[must_use]
    pub fn new(base_measure: M, log_density_fn: LogDensityFn) -> Self {
        Self {
            base_measure,
            log_density_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M, LogDensityFn, X> MeasureMarker for IntegralMeasure<M, LogDensityFn, X>
where
    M: Measure<X> + Clone,
    LogDensityFn: Clone,
    X: Clone,
{
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<M, LogDensityFn, X> Measure<X> for IntegralMeasure<M, LogDensityFn, X>
where
    M: Measure<X> + Clone,
    LogDensityFn: Clone,
    X: Clone,
{
    type RootMeasure = M::RootMeasure;

    fn in_support(&self, x: X) -> bool {
        self.base_measure.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.base_measure.root_measure()
    }
}

/// Automatic HasLogDensity implementation: log-density wrt base measure is just the function
impl<M, LogDensityFn, X, F> HasLogDensity<X, F> for IntegralMeasure<M, LogDensityFn, X>
where
    M: Measure<X> + Clone + HasLogDensity<X, F>,
    LogDensityFn: Fn(&X) -> F + Clone,
    X: Clone,
    F: Float + std::ops::Add<Output = F>,
{
    fn log_density_wrt_root(&self, x: &X) -> F {
        // Chain rule: log(dIntegral/dRoot) = log(dIntegral/dBase) + log(dBase/dRoot)
        (self.log_density_fn)(x) + self.base_measure.log_density_wrt_root(x)
    }
} 