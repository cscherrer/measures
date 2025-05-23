use crate::core::{False, HasLogDensity, Measure, MeasureMarker};
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::Float;

/// A factorial measure for discrete distributions.
///
/// This represents the measure dν = (1/k!) dμ where μ is the counting measure.
/// It's the natural base measure for discrete exponential families like Poisson
/// that have factorial terms in their densities.
#[derive(Clone)]
pub struct FactorialMeasure<F: Float> {
    /// The underlying counting measure
    counting: CountingMeasure<u64>,
    /// Phantom data for the Float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> FactorialMeasure<F> {
    /// Create a new factorial measure
    #[must_use]
    pub fn new() -> Self {
        Self {
            counting: CountingMeasure::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> Default for FactorialMeasure<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> MeasureMarker for FactorialMeasure<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<F: Float> Measure<u64> for FactorialMeasure<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        self.counting.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.counting.clone()
    }
}

/// Implement `HasLogDensity` for `FactorialMeasure`
/// This provides the factorial term: log(dν/dμ) = -log(k!)
impl<F: Float> HasLogDensity<u64, F> for FactorialMeasure<F> {
    fn log_density_wrt_root(&self, x: &u64) -> F {
        let k = *x;

        // Compute -log(k!) = -sum(log(i) for i in 1..=k)
        if k == 0 {
            F::zero() // log(0!) = log(1) = 0
        } else {
            let mut neg_log_factorial = F::zero();
            for i in 1..=k {
                neg_log_factorial = neg_log_factorial - F::from(i).unwrap().ln();
            }
            neg_log_factorial
        }
    }
}
