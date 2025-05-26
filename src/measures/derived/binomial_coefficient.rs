use crate::measures::primitive::counting::CountingMeasure;
use measures_core::{False, HasLogDensity, Measure, MeasureMarker};
use num_traits::{Float, ToPrimitive};
use special::Gamma as GammaTrait;

/// A binomial coefficient measure for discrete distributions.
///
/// This represents the measure dν = C(n,k) dμ where μ is the counting measure
/// and C(n,k) is the binomial coefficient "n choose k".
/// It's the natural base measure for discrete exponential families like Binomial
/// that have binomial coefficient terms in their densities.
///
/// The log-density with respect to the counting measure is log(C(n,k)).
#[derive(Clone)]
pub struct BinomialCoefficientMeasure<F: Float> {
    /// Number of trials (fixed parameter)
    pub n: u64,
    /// The underlying counting measure
    counting: CountingMeasure<u64>,
    /// Phantom data for the Float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> BinomialCoefficientMeasure<F> {
    /// Create a new binomial coefficient measure with fixed n
    #[must_use]
    pub fn new(n: u64) -> Self {
        Self {
            n,
            counting: CountingMeasure::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> MeasureMarker for BinomialCoefficientMeasure<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<F: Float> Measure<u64> for BinomialCoefficientMeasure<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        x <= self.n && self.counting.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.counting.clone()
    }
}

/// Compute log binomial coefficient log(C(n, k)) using log-gamma functions
///
/// This provides high accuracy for all values of n and k by using the identity:
/// log(C(n,k)) = log(Γ(n+1)) - log(Γ(k+1)) - log(Γ(n-k+1))
///
/// For k > n, returns negative infinity (coefficient is 0).
#[inline]
#[must_use]
pub fn log_binomial_coefficient<F: Float>(n: u64, k: u64) -> F {
    if k > n {
        return F::neg_infinity();
    }

    // Use the more stable log-gamma approach
    if let (Some(n_f64), Some(k_f64)) = (n.to_f64(), k.to_f64()) {
        let log_coeff = {
            let (ln_gamma_n_plus_1, _) = (n_f64 + 1.0).ln_gamma();
            let (ln_gamma_k_plus_1, _) = (k_f64 + 1.0).ln_gamma();
            let (ln_gamma_n_minus_k_plus_1, _) = (n_f64 - k_f64 + 1.0).ln_gamma();
            ln_gamma_n_plus_1 - ln_gamma_k_plus_1 - ln_gamma_n_minus_k_plus_1
        };
        F::from(log_coeff).unwrap()
    } else {
        // Fallback for types that don't convert to f64
        // This should rarely happen in practice
        F::zero()
    }
}

/// Implement `HasLogDensity` for `BinomialCoefficientMeasure`
/// This provides the binomial coefficient term: log(dν/dμ) = log(C(n,k))
impl<F: Float> HasLogDensity<u64, F> for BinomialCoefficientMeasure<F> {
    #[cfg(feature = "profiling")]
    #[profiling::function]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        profiling::scope!("binomial_coefficient_computation");
        let k = *x;

        // Check if in support first
        if k > self.n {
            return F::neg_infinity();
        }

        // Use optimized log-binomial coefficient computation
        log_binomial_coefficient::<F>(self.n, k)
    }

    #[cfg(not(feature = "profiling"))]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        let k = *x;

        // Check if in support first
        if k > self.n {
            return F::neg_infinity();
        }

        // Use optimized log-binomial coefficient computation
        log_binomial_coefficient::<F>(self.n, k)
    }
}
