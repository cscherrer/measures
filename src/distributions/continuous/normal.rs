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

use crate::core::types::{False, True};
use crate::core::{Measure, MeasureMarker};
use crate::exponential_family::{
    ExponentialFamily, ExponentialFamilyMeasure, GenericExpFamCache, GenericExpFamImpl,
};
use crate::exponential_family::traits::PrecomputeCache;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

/// A normal (Gaussian) distribution.
///
/// The normal distribution is characterized by its mean and standard deviation.
/// The density is computed with respect to Lebesgue measure.
///
/// Uses the generic exponential family cache - no need for distribution-specific cache!
#[derive(Clone)]
pub struct Normal<T: Float> {
    /// The mean of the distribution
    pub mean: T,
    /// The standard deviation of the distribution
    pub std_dev: T,
}

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

impl<T: Float + FloatConst> ExponentialFamilyMeasure<T, T> for Normal<T> {}

impl<T: Float + FloatConst> Normal<T> {
    /// Create a new normal distribution with the given mean and standard deviation.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the distribution
    /// * `std_dev` - The standard deviation of the distribution (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `std_dev` is not positive.
    #[must_use]
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }
}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> <Self as Measure<T>>::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Note: HasLogDensity implementation is now automatic via the blanket impl
// for exponential families in density.rs! No manual implementation needed.

impl<T: Float + FloatConst> ExponentialFamily<T, T> for Normal<T> {
    // Types specified once - no redundancy!
    type NaturalParam = [T; 2]; // (η₁, η₂) = (μ/σ², -1/(2σ²))
    type SufficientStat = [T; 2]; // (x, x²)
    type BaseMeasure = LebesgueMeasure<T>;
    type Cache = GenericExpFamCache<Self, T, T>; // Generic cache!

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -T::one() / (T::from(2.0).unwrap() * eta2);
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        // Compute sigma2 once and reuse
        let sigma2 = self.std_dev * self.std_dev;
        let inv_sigma2 = T::one() / sigma2;
        [
            self.mean * inv_sigma2,              // μ/σ²
            -inv_sigma2 / T::from(2.0).unwrap(), // -1/(2σ²)
        ]
    }

    fn log_partition(&self) -> T {
        // Compute log partition: A(η) = μ²/(2σ²) + ½log(2πσ²)
        let mu2 = self.mean * self.mean;
        let sigma2 = self.std_dev * self.std_dev;
        (T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap()
            + mu2 / (T::from(2.0).unwrap() * sigma2)
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn cached_log_density(&self, cache: &Self::Cache, x: &T) -> T {
        self.cached_log_density_generic(cache, x)
    }
}

impl<T: Float + FloatConst> PrecomputeCache<T, T> for Normal<T> {
    fn precompute_cache(&self) -> Self::Cache {
        self.precompute_generic_cache()
    }
}
