//! Product measures for creating joint distributions.
//!
//! This module implements product measures that combine multiple independent
//! measures into a joint measure. This is fundamental for modeling independence
//! in probabilistic models.

use crate::core::types::False;
use crate::core::{HasLogDensity, Measure, MeasureMarker};
use num_traits::Float;

/// A product measure representing the joint distribution of independent random variables.
///
/// For measures μ₁, μ₂, ..., μₙ, the product measure μ₁ × μ₂ × ... × μₙ
/// satisfies: d(μ₁ × μ₂ × ... × μₙ) = dμ₁ ⊗ dμ₂ ⊗ ... ⊗ dμₙ
///
/// The log-density of the product is the sum of individual log-densities.
#[derive(Clone, Debug)]
pub struct ProductMeasure<M1, M2> {
    pub measure1: M1,
    pub measure2: M2,
}

impl<M1, M2> ProductMeasure<M1, M2> {
    /// Create a new product measure from two independent measures.
    pub fn new(measure1: M1, measure2: M2) -> Self {
        Self { measure1, measure2 }
    }
}

impl<M1, M2> MeasureMarker for ProductMeasure<M1, M2> {
    type IsPrimitive = False;
    type IsExponentialFamily = False; // Product may not preserve exponential family structure
}

impl<M1, M2, X1, X2> Measure<(X1, X2)> for ProductMeasure<M1, M2>
where
    M1: Measure<X1>,
    M2: Measure<X2>,
    X1: Clone,
    X2: Clone,
{
    type RootMeasure = ProductMeasure<M1::RootMeasure, M2::RootMeasure>;

    fn in_support(&self, x: (X1, X2)) -> bool {
        self.measure1.in_support(x.0) && self.measure2.in_support(x.1)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        ProductMeasure::new(self.measure1.root_measure(), self.measure2.root_measure())
    }
}

impl<M1, M2, X1, X2, F> HasLogDensity<(X1, X2), F> for ProductMeasure<M1, M2>
where
    M1: HasLogDensity<X1, F>,
    M2: HasLogDensity<X2, F>,
    X1: Clone,
    X2: Clone,
    F: Float + std::ops::Add<Output = F>,
{
    fn log_density_wrt_root(&self, x: &(X1, X2)) -> F {
        // Product measure: log(dμ₁ × dμ₂) = log(dμ₁) + log(dμ₂)
        self.measure1.log_density_wrt_root(&x.0) + self.measure2.log_density_wrt_root(&x.1)
    }
}

/// Extension trait for creating product measures with a fluent interface.
pub trait ProductMeasureExt<X>: Measure<X> + Sized {
    /// Create a product measure with another measure.
    fn product<M2, X2>(self, other: M2) -> ProductMeasure<Self, M2>
    where
        M2: Measure<X2>,
        X2: Clone,
    {
        ProductMeasure::new(self, other)
    }
}

impl<M, X> ProductMeasureExt<X> for M where M: Measure<X> {}

/// Macro for creating product measures from multiple measures.
#[macro_export]
macro_rules! product_measure {
    ($m1:expr, $m2:expr) => {
        $crate::measures::combinators::product::ProductMeasure::new($m1, $m2)
    };
    ($m1:expr, $m2:expr, $($rest:expr),+) => {
        $crate::measures::combinators::product::ProductMeasure::new(
            $m1,
            product_measure!($m2, $($rest),+)
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LogDensityBuilder;
    use crate::distributions::continuous::normal::Normal;
    use crate::distributions::discrete::poisson::Poisson;

    #[test]
    fn test_product_measure_creation() {
        let normal = Normal::new(0.0, 1.0);
        let poisson = Poisson::new(2.0);

        let product = ProductMeasure::new(normal, poisson);

        // Test support
        assert!(product.in_support((0.0, 3u64)));
        assert!(product.in_support((-1.0, 0u64)));
    }

    #[test]
    fn test_product_measure_log_density() {
        let normal1 = Normal::new(0.0, 1.0);
        let normal2 = Normal::new(1.0, 2.0);

        let product = ProductMeasure::new(normal1.clone(), normal2.clone());

        let x = (0.5, 1.5);
        let product_density: f64 = product.log_density().at(&x);

        // Should equal sum of individual densities
        let individual_sum: f64 = normal1.log_density().at(&x.0) + normal2.log_density().at(&x.1);

        assert!((product_density - individual_sum).abs() < 1e-10);
    }

    #[test]
    fn test_fluent_interface() {
        let normal = Normal::new(0.0, 1.0);
        let poisson = Poisson::new(2.0);

        let product = normal.product(poisson);

        assert!(product.in_support((0.0, 1u64)));
    }

    #[test]
    fn test_product_measure_macro() {
        let n1 = Normal::new(0.0, 1.0);
        let n2 = Normal::new(1.0, 1.0);
        let n3 = Normal::new(2.0, 1.0);

        let product = product_measure!(n1, n2, n3);

        // This creates a nested structure: ProductMeasure<Normal, ProductMeasure<Normal, Normal>>
        assert!(product.in_support((0.0, (1.0, 2.0))));
    }
}
