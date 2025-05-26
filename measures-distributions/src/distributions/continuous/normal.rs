//! Normal (Gaussian) distribution implementation.
//!
//! This module provides the Normal distribution, which is a continuous probability
//! distribution characterized by its mean and standard deviation. The density is
//! computed with respect to Lebesgue measure.
//!
//! # Example
//!
//! ```rust
//! use measures::{Normal, LogDensityBuilder};
//!
//! let normal = Normal::new(0.0, 1.0); // Standard normal distribution
//!
//! // Compute log-density at x = 0
//! let ld = normal.log_density();
//! let log_density_value: f64 = ld.at(&0.0);
//! ```

use measures_core::LebesgueMeasure;
use measures_core::core::utils::float_constant;
use measures_core::{
    DecompositionBuilder, False, HasLogDensityDecomposition, LogDensityDecomposition, Measure,
    MeasureMarker, True,
};
use measures_exponential_family::exponential_family::traits::ExponentialFamily;
use num_traits::{Float, FloatConst};

/// Normal distribution N(μ, σ²)
///
/// The normal distribution is a continuous probability distribution characterized by
/// its bell-shaped curve. It's parameterized by mean μ and standard deviation σ.
///
/// Log-density decomposition:
/// log f(x|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
///              = `f_const` + `f_param(σ)` + `f_mixed(x,μ,σ)`
///
/// Where:
/// - `f_const` = -0.5*log(2π) (constant)
/// - `f_param(σ)` = -log(σ) (parameter-only)
/// - `f_mixed(x,μ,σ)` = -0.5*(x-μ)²/σ² (mixed data-parameter)
#[derive(Debug, Clone, PartialEq)]
pub struct Normal<T> {
    /// Mean parameter μ
    pub mean: T,
    /// Standard deviation parameter σ
    pub std_dev: T,
}

/// Parameters for Normal distribution: (mean, `std_dev`)
pub type NormalParams<T> = (T, T);

impl<T: Float> Default for Normal<T> {
    fn default() -> Self {
        Self {
            mean: T::zero(),
            std_dev: T::one(),
        }
    }
}

impl<T: Float> MeasureMarker for Normal<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float + FloatConst> Normal<T> {
    /// Create a new normal distribution with given mean and standard deviation.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }

    /// Get the variance σ²
    pub fn variance(&self) -> T {
        self.std_dev * self.std_dev
    }

    /// Get parameters as a tuple (mean, `std_dev`).
    pub fn params(&self) -> NormalParams<T> {
        (self.mean, self.std_dev)
    }

    /// Compute log-density for IID samples efficiently using decomposition.
    ///
    /// For n IID samples, this is much more efficient than computing individual densities.
    pub fn log_density_iid(&self, samples: &[T]) -> T
    where
        T: std::iter::Sum,
    {
        let decomp = self.log_density_decomposition();
        decomp.evaluate_iid(samples, &self.params())
    }
}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

/// Implementation of structured log-density decomposition for Normal distribution.
///
/// This shows how exponential families can also benefit from the structured approach,
/// even though they have their own specialized implementations.
impl<T: Float + FloatConst> HasLogDensityDecomposition<T, NormalParams<T>, T> for Normal<T> {
    fn log_density_decomposition(&self) -> LogDensityDecomposition<T, NormalParams<T>, T> {
        let two_pi = float_constant::<T>(2.0) * T::PI();

        DecompositionBuilder::new()
            // Constant term: -0.5*log(2π)
            .constant(-float_constant::<T>(0.5) * two_pi.ln())
            // Parameter-only term: -log(σ)
            .param_term(
                |(_mean, std_dev): &NormalParams<T>| -std_dev.ln(),
                "negative log standard deviation",
            )
            // Mixed term: -0.5*(x-μ)²/σ²
            .mixed_term(
                |x: &T, (mean, std_dev): &NormalParams<T>| {
                    let standardized = (*x - *mean) / *std_dev;
                    -float_constant::<T>(0.5) * standardized * standardized
                },
                "negative half squared standardized residual",
            )
            .build()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Normal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -(float_constant::<T>(2.0) * eta2).recip();
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<T, T>>::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<T, T>>::NaturalParam, T) {
        let sigma2 = self.variance();
        let mu2 = self.mean * self.mean;
        let inv_sigma2 = sigma2.recip();

        let natural_params = [
            self.mean * inv_sigma2,
            float_constant::<T>(-0.5) * inv_sigma2,
        ];

        let log_partition = (float_constant::<T>(2.0) * T::PI() * sigma2).ln()
            * float_constant::<T>(0.5)
            + float_constant::<T>(0.5) * mu2 * inv_sigma2;

        (natural_params, log_partition)
    }
}

// Implementation of HasLogDensity for Normal distribution
impl<T: Float + FloatConst> measures_core::HasLogDensity<T, T> for Normal<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        let two_pi = float_constant::<T>(2.0) * T::PI();
        let standardized = (*x - self.mean) / self.std_dev;

        -float_constant::<T>(0.5) * two_pi.ln()
            - self.std_dev.ln()
            - float_constant::<T>(0.5) * standardized * standardized
    }
}

// Auto-JIT implementation for Normal distribution
#[cfg(feature = "jit")]
measures_exponential_family::auto_jit_impl!(Normal<f64>);
