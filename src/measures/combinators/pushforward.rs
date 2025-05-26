//! Pushforward measures for change of variables transformations.
//!
//! This module implements pushforward measures, which are fundamental in measure theory
//! for handling transformations of random variables. If X ~ μ and Y = f(X), then
//! Y ~ f₊μ (the pushforward of μ under f).

use crate::core::types::False;
use crate::core::{HasLogDensity, Measure, MeasureMarker};
use crate::measures::primitive::lebesgue::LebesgueMeasure;

/// A pushforward measure representing the distribution of a transformed random variable.
///
/// If X ~ μ and Y = f(X), then Y ~ f₊μ (pushforward of μ under f).
/// The density transformation follows the change of variables formula:
/// dν/dλ(y) = (dμ/dλ)(f⁻¹(y)) * |det(Df⁻¹(y))|
///
/// where Df⁻¹ is the Jacobian of the inverse transformation.
#[derive(Clone, Debug)]
pub struct PushforwardMeasure<M, F, InvF, J> {
    /// The base measure μ
    pub base_measure: M,
    /// The forward transformation f
    pub forward: F,
    /// The inverse transformation f⁻¹
    pub inverse: InvF,
    /// The log absolute determinant of the Jacobian of f⁻¹
    pub log_abs_det_jacobian: J,
}

impl<M, F, InvF, J> PushforwardMeasure<M, F, InvF, J> {
    /// Create a new pushforward measure.
    ///
    /// # Arguments
    /// * `base_measure` - The original measure μ
    /// * `forward` - The transformation function f
    /// * `inverse` - The inverse transformation f⁻¹
    /// * `log_abs_det_jacobian` - Function computing log|det(Df⁻¹(y))|
    pub fn new(base_measure: M, forward: F, inverse: InvF, log_abs_det_jacobian: J) -> Self {
        Self {
            base_measure,
            forward,
            inverse,
            log_abs_det_jacobian,
        }
    }
}

impl<M, F, InvF, J> MeasureMarker for PushforwardMeasure<M, F, InvF, J> {
    type IsPrimitive = False;
    type IsExponentialFamily = False; // Pushforward generally doesn't preserve exponential family
}

// For now, we'll implement Measure for the specific case where both input and output are real numbers
// This avoids the complex root measure type issues
impl<M, F, InvF, J> Measure<f64> for PushforwardMeasure<M, F, InvF, J>
where
    M: Measure<f64, RootMeasure = LebesgueMeasure<f64>>,
    F: Fn(&f64) -> f64 + Clone,
    InvF: Fn(&f64) -> f64 + Clone,
    J: Clone,
{
    type RootMeasure = LebesgueMeasure<f64>; // Output space is also real numbers with Lebesgue measure

    fn in_support(&self, y: f64) -> bool {
        // y is in support if f⁻¹(y) is in support of the base measure
        let x = (self.inverse)(&y);
        self.base_measure.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new() // Lebesgue measure on the output space
    }
}

impl<M, F, InvF, J> HasLogDensity<f64, f64> for PushforwardMeasure<M, F, InvF, J>
where
    M: HasLogDensity<f64, f64>,
    F: Fn(&f64) -> f64 + Clone,
    InvF: Fn(&f64) -> f64 + Clone,
    J: Fn(&f64) -> f64 + Clone,
{
    fn log_density_wrt_root(&self, y: &f64) -> f64 {
        // Change of variables formula:
        // log(dν/dλ)(y) = log(dμ/dλ)(f⁻¹(y)) + log|det(Df⁻¹(y))|
        let x = (self.inverse)(y);
        let base_log_density = self.base_measure.log_density_wrt_root(&x);
        let log_jacobian = (self.log_abs_det_jacobian)(y);

        base_log_density + log_jacobian
    }
}

/// Extension trait for creating pushforward measures with a fluent interface.
pub trait PushforwardExt<X>: Measure<X> + Sized {
    /// Create a pushforward measure using a transformation.
    fn pushforward<Y, F, InvF, J>(
        self,
        forward: F,
        inverse: InvF,
        log_abs_det_jacobian: J,
    ) -> PushforwardMeasure<Self, F, InvF, J>
    where
        F: Fn(&X) -> Y + Clone,
        InvF: Fn(&Y) -> X + Clone,
        J: Clone,
        Y: Clone,
    {
        PushforwardMeasure::new(self, forward, inverse, log_abs_det_jacobian)
    }
}

impl<M, X> PushforwardExt<X> for M where M: Measure<X> {}

/// Common transformations for univariate distributions.
pub mod transforms {
    use num_traits::Float;

    /// Log transformation: Y = log(X)
    pub fn log_transform<F: Float>() -> (
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
    ) {
        let forward = |x: &F| x.ln();
        let inverse = |y: &F| y.exp();
        let log_abs_det_jacobian = |y: &F| *y; // log|d/dy exp(y)| = log(exp(y)) = y

        (forward, inverse, log_abs_det_jacobian)
    }

    /// Exponential transformation: Y = exp(X)
    pub fn exp_transform<F: Float>() -> (
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
    ) {
        let forward = |x: &F| x.exp();
        let inverse = |y: &F| y.ln();
        let log_abs_det_jacobian = |y: &F| -y.ln(); // log|d/dy ln(y)| = log(1/y) = -ln(y)

        (forward, inverse, log_abs_det_jacobian)
    }

    /// Linear transformation: Y = a*X + b
    pub fn linear_transform<F: Float>(
        a: F,
        b: F,
    ) -> (
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
    ) {
        let forward = move |x: &F| a * *x + b;
        let inverse = move |y: &F| (*y - b) / a;
        let log_abs_det_jacobian = move |_y: &F| -a.abs().ln(); // log|d/dy (y-b)/a| = log(1/|a|) = -ln|a|

        (forward, inverse, log_abs_det_jacobian)
    }

    /// Logit transformation: Y = log(X/(1-X)) for X ∈ (0,1)
    pub fn logit_transform<F: Float>() -> (
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
        impl Fn(&F) -> F + Clone,
    ) {
        let forward = |x: &F| (*x / (F::one() - *x)).ln();
        let inverse = |y: &F| {
            let exp_y = y.exp();
            exp_y / (F::one() + exp_y)
        };
        let log_abs_det_jacobian = |y: &F| {
            // d/dy sigmoid(y) = sigmoid(y) * (1 - sigmoid(y))
            // log|det| = log(sigmoid(y)) + log(1 - sigmoid(y)) = -y - 2*log(1 + exp(-y))
            let exp_neg_y = (-*y).exp();
            -(*y) - (F::one() + exp_neg_y).ln() - (F::one() + exp_neg_y).ln()
        };

        (forward, inverse, log_abs_det_jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LogDensityBuilder;
    use crate::distributions::continuous::normal::Normal;
    use crate::measures::combinators::pushforward::transforms::*;

    #[test]
    fn test_log_normal_via_pushforward() {
        // Create log-normal by pushing forward normal through exp
        let normal = Normal::new(0.0, 1.0);
        let (forward, inverse, log_jacobian) = exp_transform();

        let log_normal = normal.pushforward(forward, inverse, log_jacobian);

        // Test that it's in support for positive values
        assert!(log_normal.in_support(1.0));
        assert!(log_normal.in_support(0.1));

        // Test log-density computation
        let density: f64 = log_normal.log_density().at(&1.0);
        assert!(density.is_finite());
    }

    #[test]
    fn test_linear_transformation() {
        let normal = Normal::new(0.0, 1.0);
        let (forward, inverse, log_jacobian) = linear_transform(2.0, 1.0); // Y = 2X + 1

        let transformed = normal.pushforward(forward, inverse, log_jacobian);

        // Test support
        assert!(transformed.in_support(1.0)); // Should be in support everywhere

        // Test density
        let density: f64 = transformed.log_density().at(&1.0);
        assert!(density.is_finite());
    }

    #[test]
    fn test_exp_transformation_density() {
        let normal = Normal::new(0.0, 1.0);
        let (forward, inverse, log_jacobian) = exp_transform();

        let log_normal = normal.pushforward(forward, inverse, log_jacobian);

        // Test density at a few points
        let points = [0.5, 1.0, 2.0];
        for &y in &points {
            let density: f64 = log_normal.log_density().at(&y);
            assert!(density.is_finite(), "Density should be finite at y={}", y);
        }
    }
}
