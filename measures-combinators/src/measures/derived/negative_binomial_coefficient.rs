use crate::measures::primitive::counting::CountingMeasure;
use measures_core::{False, HasLogDensity, Measure, MeasureMarker};
use num_traits::{Float, ToPrimitive};
use special::Gamma as GammaTrait;

/// A negative binomial coefficient measure for discrete distributions.
///
/// This represents the measure dν = C(x+r-1,x) dμ where μ is the counting measure
/// and C(x+r-1,x) is the negative binomial coefficient.
/// It's the natural base measure for discrete exponential families like `NegativeBinomial`
/// that have negative binomial coefficient terms in their densities.
///
/// The log-density with respect to the counting measure is log(C(x+r-1,x)).
#[derive(Clone)]
pub struct NegativeBinomialCoefficientMeasure<F: Float> {
    /// Number of failures (fixed parameter)
    pub r: u64,
    /// The underlying counting measure
    counting: CountingMeasure<u64>,
    /// Phantom data for the Float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> NegativeBinomialCoefficientMeasure<F> {
    /// Create a new negative binomial coefficient measure with fixed r
    #[must_use]
    pub fn new(r: u64) -> Self {
        Self {
            r,
            counting: CountingMeasure::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> MeasureMarker for NegativeBinomialCoefficientMeasure<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<F: Float> Measure<u64> for NegativeBinomialCoefficientMeasure<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        // Negative binomial has support on {0, 1, 2, ...}
        self.counting.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.counting.clone()
    }
}

/// Compute log negative binomial coefficient log(C(x+r-1, x)) using log-gamma functions
///
/// This provides high accuracy for all values of x and r by using the identity:
/// log(C(x+r-1,x)) = log(Γ(x+r)) - log(Γ(x+1)) - log(Γ(r))
#[inline]
#[must_use]
pub fn log_negative_binomial_coefficient<F: Float>(r: u64, x: u64) -> F {
    // Use the log-gamma approach for numerical stability
    if let (Some(r_f64), Some(x_f64)) = (r.to_f64(), x.to_f64()) {
        let log_coeff = {
            let (ln_gamma_x_plus_r, _) = special::Gamma::ln_gamma(x_f64 + r_f64);
            let (ln_gamma_x_plus_1, _) = special::Gamma::ln_gamma(x_f64 + 1.0);
            let (ln_gamma_r, _) = special::Gamma::ln_gamma(r_f64);
            ln_gamma_x_plus_r - ln_gamma_x_plus_1 - ln_gamma_r
        };
        F::from(log_coeff).unwrap()
    } else {
        // Fallback for types that don't convert to f64
        // This should rarely happen in practice
        F::zero()
    }
}

/// Implement `HasLogDensity` for `NegativeBinomialCoefficientMeasure`
/// This provides the negative binomial coefficient term: log(dν/dμ) = log(C(x+r-1,x))
impl<F: Float> HasLogDensity<u64, F> for NegativeBinomialCoefficientMeasure<F> {
    #[cfg(feature = "profiling")]
    #[profiling::function]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        profiling::scope!("negative_binomial_coefficient_computation");
        let x_val = *x;

        // Use optimized log-negative-binomial coefficient computation
        log_negative_binomial_coefficient::<F>(self.r, x_val)
    }

    #[cfg(not(feature = "profiling"))]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        let x_val = *x;

        // Use optimized log-negative-binomial coefficient computation
        log_negative_binomial_coefficient::<F>(self.r, x_val)
    }
}
