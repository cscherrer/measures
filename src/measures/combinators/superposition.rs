//! Superposition (mixture) measures for combining multiple measures with weights.
//!
//! This module implements mixture measures that combine multiple probability measures
//! with specified weights. This is fundamental for mixture models and hierarchical
//! Bayesian modeling.

use crate::core::types::False;
use crate::core::{HasLogDensity, Measure, MeasureMarker};
use num_traits::Float;

/// A superposition (mixture) measure that combines multiple measures with weights.
///
/// For measures μ₁, μ₂, ..., μₙ with weights w₁, w₂, ..., wₙ (where Σwᵢ = 1),
/// the mixture measure is: ν = Σᵢ wᵢ μᵢ
///
/// The density is: dν/dλ = Σᵢ wᵢ (dμᵢ/dλ)
#[derive(Clone, Debug)]
pub struct MixtureMeasure<M, F> {
    // TODO: We really have two cases:
    //     1. Mixture of measures (weights need not sum to one)
    //     2. Mixture of distributions, intending a new distribution
    /// The component measures
    pub components: Vec<M>,
    /// The mixture weights (should sum to 1)
    // TODO: Maybe log-weights are better?
    pub weights: Vec<F>,
}

impl<M, F: Float> MixtureMeasure<M, F> {
    /// Create a new mixture measure from components and weights.
    ///
    /// # Arguments
    /// * `components` - Vector of component measures
    /// * `weights` - Vector of mixture weights (should sum to 1)
    ///
    /// # Panics
    /// Panics if components and weights have different lengths or if weights don't sum to 1.
    #[must_use]
    pub fn new(components: Vec<M>, weights: Vec<F>) -> Self {
        assert_eq!(
            components.len(),
            weights.len(),
            "Components and weights must have same length"
        );

        let weight_sum: F = weights.iter().copied().fold(F::zero(), |acc, w| acc + w);
        assert!(
            (weight_sum - F::one()).abs() < F::from(1e-10).unwrap(),
            "Weights must sum to 1"
        );

        Self {
            components,
            weights,
        }
    }

    /// Create a uniform mixture (equal weights).
    #[must_use]
    pub fn uniform(components: Vec<M>) -> Self {
        let n = components.len();
        let weight = F::one() / F::from(n).unwrap();
        let weights = vec![weight; n];
        Self::new(components, weights)
    }

    /// Get the number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

impl<M, F> MeasureMarker for MixtureMeasure<M, F> {
    type IsPrimitive = False;
    type IsExponentialFamily = False; // Mixtures generally don't preserve exponential family structure
}

impl<M, F, X> Measure<X> for MixtureMeasure<M, F>
where
    M: Measure<X>,
    F: Float,
    X: Clone,
{
    type RootMeasure = M::RootMeasure; // All components should have same root measure

    fn in_support(&self, x: X) -> bool {
        // x is in support if it's in support of any component with positive weight
        self.components
            .iter()
            .zip(&self.weights)
            .any(|(component, &weight)| weight > F::zero() && component.in_support(x.clone()))
    }

    fn root_measure(&self) -> Self::RootMeasure {
        // Assume all components have the same root measure
        self.components[0].root_measure()
    }
}

impl<M, F, X> HasLogDensity<X, F> for MixtureMeasure<M, F>
where
    M: HasLogDensity<X, F>,
    F: Float + std::iter::Sum,
    X: Clone,
{
    fn log_density_wrt_root(&self, x: &X) -> F {
        // Mixture density: log(Σᵢ wᵢ exp(log_densityᵢ))
        // Use log-sum-exp trick for numerical stability

        if self.components.is_empty() {
            return F::neg_infinity();
        }

        // Compute log(wᵢ) + log_densityᵢ for each component
        let log_weighted_densities: Vec<F> = self
            .components
            .iter()
            .zip(&self.weights)
            .map(|(component, &weight)| {
                let log_density = component.log_density_wrt_root(x);
                weight.ln() + log_density
            })
            .collect();

        // Apply log-sum-exp trick
        log_sum_exp(&log_weighted_densities)
    }
}

/// Compute log(Σᵢ exp(xᵢ)) using the log-sum-exp trick for numerical stability.
fn log_sum_exp<F: Float>(log_values: &[F]) -> F {
    // TODO: use the streaming implementation as in rv
    if log_values.is_empty() {
        return F::neg_infinity();
    }

    // Find the maximum value
    let max_val = log_values
        .iter()
        .copied()
        .fold(F::neg_infinity(), |acc, x| if x > acc { x } else { acc });

    if max_val.is_infinite() && max_val < F::zero() {
        return F::neg_infinity();
    }

    // Compute log(Σᵢ exp(xᵢ - max)) + max
    let sum_exp: F = log_values
        .iter()
        .map(|&x| (x - max_val).exp())
        .fold(F::zero(), |acc, val| acc + val);

    max_val + sum_exp.ln()
}

/// Extension trait for creating mixture measures with a fluent interface.
pub trait MixtureExt<X>: Measure<X> + Sized {
    /// Create a mixture with another measure using specified weights.
    fn mixture<F: Float>(self, other: Self, weight1: F, weight2: F) -> MixtureMeasure<Self, F> {
        let total_weight = weight1 + weight2;
        let normalized_weight1 = weight1 / total_weight;
        let normalized_weight2 = weight2 / total_weight;

        MixtureMeasure::new(
            vec![self, other],
            vec![normalized_weight1, normalized_weight2],
        )
    }

    /// Create a uniform mixture with another measure (equal weights).
    fn uniform_mixture(self, other: Self) -> MixtureMeasure<Self, f64> {
        MixtureMeasure::uniform(vec![self, other])
    }
}

impl<M, X> MixtureExt<X> for M where M: Measure<X> {}

/// Macro for creating mixture measures from multiple components.
#[macro_export]
macro_rules! mixture {
    // Uniform mixture
    (uniform: $($component:expr),+ $(,)?) => {
        $crate::measures::combinators::superposition::MixtureMeasure::uniform(
            vec![$($component),+]
        )
    };

    // Weighted mixture
    ($(($weight:expr, $component:expr)),+ $(,)?) => {{
        let components = vec![$($component),+];
        let weights = vec![$($weight),+];
        $crate::measures::combinators::superposition::MixtureMeasure::new(components, weights)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LogDensityBuilder;
    use crate::distributions::continuous::normal::Normal;

    #[test]
    fn test_mixture_creation() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(2.0, 1.0);

        let mixture = MixtureMeasure::new(vec![normal1, normal2], vec![0.3, 0.7]);

        assert_eq!(mixture.num_components(), 2);
    }

    #[test]
    fn test_uniform_mixture() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(2.0, 1.0);
        let normal3 = Normal::new(-1.0, 1.0);

        let mixture: MixtureMeasure<Normal<f64>, f64> =
            MixtureMeasure::uniform(vec![normal1, normal2, normal3]);

        assert_eq!(mixture.num_components(), 3);
        // Each weight should be 1/3
        assert!((mixture.weights[0] - 1.0f64 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixture_log_density() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(2.0, 1.0);

        let mixture = MixtureMeasure::new(vec![normal1.clone(), normal2.clone()], vec![0.5, 0.5]);

        let x = 1.0;
        let mixture_density: f64 = mixture.log_density().at(&x);

        // Should be log(0.5 * exp(density1) + 0.5 * exp(density2))
        let density1: f64 = normal1.log_density().at(&x);
        let density2: f64 = normal2.log_density().at(&x);
        let expected = log_sum_exp(&[0.5f64.ln() + density1, 0.5f64.ln() + density2]);

        assert!((mixture_density - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fluent_interface() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(2.0, 1.0);

        let mixture = normal1.mixture(normal2, 0.3, 0.7);

        assert_eq!(mixture.num_components(), 2);
        assert!((mixture.weights[0] - 0.3).abs() < 1e-10);
        assert!((mixture.weights[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_mixture_macro() {
        let n1 = Normal::new(0.0, 1.0);
        let n2 = Normal::new(1.0, 1.0);
        let n3 = Normal::new(2.0, 1.0);

        // Uniform mixture
        let uniform_mix: MixtureMeasure<Normal<f64>, f64> =
            mixture!(uniform: n1.clone(), n2.clone(), n3.clone());
        assert_eq!(uniform_mix.num_components(), 3);

        // Weighted mixture
        let weighted_mix: MixtureMeasure<Normal<f64>, f64> =
            mixture![(0.5, n1), (0.3, n2), (0.2, n3)];
        assert_eq!(weighted_mix.num_components(), 3);
        assert!((weighted_mix.weights[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_invalid_weights() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(2.0, 1.0);

        // Weights don't sum to 1
        let _mixture = MixtureMeasure::new(
            vec![normal1, normal2],
            vec![0.3, 0.5], // Sum = 0.8 ≠ 1
        );
    }
}
