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
use crate::core::{HasLogDensity, Measure, MeasureMarker};
use crate::exponential_family::{ExponentialFamily, ExponentialFamilyMeasure};
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

/// Precomputed cache for optimal Normal distribution density computation.
///
/// This struct caches the expensive exponential family components:
/// - Natural parameters η = [μ/σ², -1/(2σ²)]
/// - Log partition function A(η) = μ²/(2σ²) + ½log(2πσ²)
/// - Base measure (for chain rule computation)
#[derive(Clone)]
pub struct NormalCache<T: Float> {
    /// Cached natural parameters η = [μ/σ², -1/(2σ²)]
    pub natural_params: [T; 2],
    /// Cached log partition function A(η)
    pub log_partition: T,
    /// Cached base measure
    pub base_measure: LebesgueMeasure<T>,
}

impl<T: Float + FloatConst> NormalCache<T> {
    #[must_use]
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");

        // Compute natural parameters: η = [μ/σ², -1/(2σ²)]
        let sigma2 = std_dev * std_dev;
        let inv_sigma2 = T::one() / sigma2;
        let natural_params = [
            mean * inv_sigma2,                   // μ/σ²
            -inv_sigma2 / T::from(2.0).unwrap(), // -1/(2σ²)
        ];

        // Compute log partition: A(η) = μ²/(2σ²) + ½log(2πσ²)
        let mu2 = mean * mean;
        let log_partition = (T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap()
            + mu2 / (T::from(2.0).unwrap() * sigma2);

        Self {
            natural_params,
            log_partition,
            base_measure: LebesgueMeasure::new(),
        }
    }
}

/// A normal (Gaussian) distribution.
///
/// The normal distribution is characterized by its mean and standard deviation.
/// The density is computed with respect to Lebesgue measure.
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

/// Implement `HasLogDensity` for automatic shared-root computation
impl<T: Float + FloatConst> HasLogDensity<T, T> for Normal<T> {
    #[inline]
    fn log_density_wrt_root(&self, x: &T) -> T {
        // Use optimized cached computation from exponential family framework
        let cache = self.precompute_cache();
        self.cached_log_density(&cache, x)
    }
}

impl<T: Float + FloatConst> ExponentialFamily<T, T> for Normal<T> {
    type NaturalParam = [T; 2]; // (η₁, η₂) = (μ/σ², -1/(2σ²))
    type SufficientStat = [T; 2]; // (x, x²)
    type BaseMeasure = LebesgueMeasure<T>;
    type Cache = NormalCache<T>; // Use our existing NormalCache as the cached type

    fn from_natural(param: <Self as ExponentialFamily<T, T>>::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -T::one() / (T::from(2.0).unwrap() * eta2);
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn to_natural(&self) -> <Self as ExponentialFamily<T, T>>::NaturalParam {
        // Compute sigma2 once and reuse
        let sigma2 = self.std_dev * self.std_dev;
        let inv_sigma2 = T::one() / sigma2;
        [
            self.mean * inv_sigma2,              // μ/σ²
            -inv_sigma2 / T::from(2.0).unwrap(), // -1/(2σ²)
        ]
    }

    fn log_partition(&self) -> T {
        // Compute sigma2 once and reuse
        let sigma2 = self.std_dev * self.std_dev;
        let mu2 = self.mean * self.mean;
        (T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap()
            + mu2 / (T::from(2.0).unwrap() * sigma2)
    }

    fn sufficient_statistic(&self, x: &T) -> <Self as ExponentialFamily<T, T>>::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn precompute_cache(&self) -> Self::Cache {
        NormalCache::new(self.mean, self.std_dev)
    }

    fn cached_log_density(&self, cache: &Self::Cache, x: &T) -> T {
        // Use generic exponential family computation: η·T(x) - A(η) + log h(x)
        use crate::traits::DotProduct;

        // Sufficient statistics: T(x) = [x, x²]
        let sufficient_stats = [*x, *x * *x];

        // Exponential family part: η·T(x) - A(η)
        let exp_fam_part = cache.natural_params.dot(&sufficient_stats) - cache.log_partition;

        // Chain rule part: log h(x) = 0 for Lebesgue measure
        let chain_rule_part = cache.base_measure.log_density_wrt_root(x);

        // Complete log-density
        exp_fam_part + chain_rule_part
    }
}
