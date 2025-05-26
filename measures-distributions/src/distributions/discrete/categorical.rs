//! Categorical distribution implementation.
//!
//! This module provides the Categorical distribution, which is a discrete probability
//! distribution that describes the possible results of a random variable that can take
//! on one of K possible categories. The density is computed with respect to counting measure.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Categorical;
//! use measures::LogDensityBuilder;
//!
//! let categorical = Categorical::new(vec![0.2, 0.3, 0.5]); // 3 categories with given probabilities
//!
//! // Compute log-density at x = 1 (second category, 0-indexed)
//! let ld = categorical.log_density();
//! let log_density_value: f64 = ld.at(&1);
//! ```

use measures_core::DotProduct;
use measures_core::primitive::counting::CountingMeasure;
use measures_core::{False, True};
use measures_core::{HasLogDensity, Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
use num_traits::Float;

/// Categorical distribution Cat(p₁, p₂, ..., pₖ).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [log(p₁/pₖ), log(p₂/pₖ), ..., log(pₖ₋₁/pₖ)]
/// - Sufficient statistics: T(x) = [I(x=0), I(x=1), ..., I(x=K-2)]
/// - Log partition: A(η) = log(1 + Σᵢ exp(ηᵢ))
/// - Base measure: Counting measure on {0, 1, ..., K-1}
#[derive(Clone, Debug)]
pub struct Categorical<T> {
    pub probs: Vec<T>, // Probabilities for each category
}

impl<T: Float> MeasureMarker for Categorical<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float + std::iter::Sum> Categorical<T> {
    /// Create a new categorical distribution with given probabilities.
    #[must_use]
    pub fn new(probs: Vec<T>) -> Self {
        assert!(!probs.is_empty(), "Must have at least one category");
        assert!(
            probs.iter().all(|&p| p >= T::zero()),
            "All probabilities must be non-negative"
        );

        let sum: T = probs.iter().copied().sum();
        assert!(
            (sum - T::one()).abs() < T::from(1e-10).unwrap(),
            "Probabilities must sum to 1"
        );

        Self { probs }
    }

    /// Create a uniform categorical distribution with k categories.
    #[must_use]
    pub fn uniform(k: usize) -> Self {
        assert!(k > 0, "Must have at least one category");
        let prob = T::one() / T::from(k).unwrap();
        Self::new(vec![prob; k])
    }

    /// Get the number of categories.
    #[must_use]
    pub fn num_categories(&self) -> usize {
        self.probs.len()
    }

    /// Get the mean (expected category index).
    #[must_use]
    pub fn mean(&self) -> T {
        self.probs
            .iter()
            .enumerate()
            .map(|(i, &p)| T::from(i).unwrap() * p)
            .sum()
    }

    /// Get the variance.
    #[must_use]
    pub fn variance(&self) -> T {
        let mean = self.mean();
        self.probs
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let diff = T::from(i).unwrap() - mean;
                p * diff * diff
            })
            .sum()
    }

    /// Get the log-odds relative to the last category.
    fn log_odds(&self) -> Vec<T> {
        let last_prob = self.probs[self.probs.len() - 1];
        self.probs[..self.probs.len() - 1]
            .iter()
            .map(|&p| (p / last_prob).ln())
            .collect()
    }
}

impl<T: Float> Measure<usize> for Categorical<T> {
    type RootMeasure = CountingMeasure<usize>;

    fn in_support(&self, x: usize) -> bool {
        x < self.probs.len()
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<usize>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<usize, T> for Categorical<T>
where
    T: Float + std::fmt::Debug + 'static + std::iter::Sum,
{
    type NaturalParam = Vec<T>;
    type SufficientStat = Vec<T>;
    type BaseMeasure = CountingMeasure<usize>;

    fn from_natural(param: <Self as ExponentialFamily<usize, T>>::NaturalParam) -> Self {
        // Convert from log-odds to probabilities using softmax
        let k = param.len() + 1; // Number of categories
        let mut probs = Vec::with_capacity(k);

        // Compute exp(η_i) for i = 0, ..., K-2
        let exp_etas: Vec<T> = param.iter().map(|&eta| eta.exp()).collect();

        // Compute normalization constant: 1 + Σᵢ exp(ηᵢ)
        let normalizer = T::one() + exp_etas.iter().copied().sum::<T>();

        // Compute probabilities: pᵢ = exp(ηᵢ) / normalizer for i < K-1
        for &exp_eta in &exp_etas {
            probs.push(exp_eta / normalizer);
        }

        // Last probability: pₖ₋₁ = 1 / normalizer
        probs.push(T::one() / normalizer);

        Self::new(probs)
    }

    fn sufficient_statistic(
        &self,
        x: &usize,
    ) -> <Self as ExponentialFamily<usize, T>>::SufficientStat {
        let k = self.probs.len();
        let mut stats = vec![T::zero(); k - 1];

        // One-hot encoding for categories 0 to K-2
        if *x < k - 1 {
            stats[*x] = T::one();
        }

        stats
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<usize, T>>::BaseMeasure {
        CountingMeasure::<usize>::new()
    }

    fn natural_and_log_partition(
        &self,
    ) -> (<Self as ExponentialFamily<usize, T>>::NaturalParam, T) {
        let natural_params = self.log_odds();

        // Log partition: A(η) = log(1 + Σᵢ exp(ηᵢ))
        let sum_exp_eta: T = natural_params.iter().map(|&eta| eta.exp()).sum();
        let log_partition = (T::one() + sum_exp_eta).ln();

        (natural_params, log_partition)
    }

    /// Override `exp_fam_log_density` to check support first
    fn exp_fam_log_density(&self, x: &usize) -> T
    where
        Self::NaturalParam: measures_core::DotProduct<Self::SufficientStat, Output = T>,
        Self::BaseMeasure: measures_core::HasLogDensity<usize, T>,
    {
        // Check if in support first
        if !self.in_support(*x) {
            return T::neg_infinity();
        }

        // Use the default exponential family computation
        let (natural_params, log_partition) = self.natural_and_log_partition();
        let sufficient_stats = self.sufficient_statistic(x);

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

        // Chain rule part: log-density of base measure with respect to root measure
        let chain_rule_part = self.base_measure().log_density_wrt_root(x);

        // Complete log-density: exponential family + chain rule
        exp_fam_part + chain_rule_part
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> measures_exponential_family::JITOptimizer<usize, T> for Categorical<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<measures_exponential_family::JITFunction, measures_exponential_family::JITError>
    {
        let _log_probs: Vec<f64> = self
            .probs
            .iter()
            .map(|&p| p.to_f64().unwrap().ln())
            .collect();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            measures_exponential_family::JITError::UnsupportedExpression(
                "Categorical distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Implementation of HasLogDensity for Categorical distribution
impl<T: Float> measures_core::HasLogDensity<usize, T> for Categorical<T> {
    fn log_density_wrt_root(&self, x: &usize) -> T {
        if *x < self.probs.len() {
            self.probs[*x].ln()
        } else {
            T::neg_infinity() // Outside support
        }
    }
}
