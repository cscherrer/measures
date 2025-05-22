//! Multivariate Normal distribution implementation.
//!
//! This module provides the Multivariate Normal (Gaussian) distribution, which is
//! defined over n-dimensional real vectors.

use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{
    ExponentialFamily, InnerProduct,
};
use crate::measures::lebesgue::LebesgueMeasure;
use nalgebra::{ComplexField, DMatrix, DVector, RealField, Scalar};
use num_traits::{Float, FloatConst};
use std::fmt::Debug;

/// A multivariate normal (Gaussian) distribution.
///
/// The multivariate normal distribution is characterized by its mean vector
/// and covariance matrix.
#[derive(Clone)]
pub struct MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    /// The mean vector
    pub mean: DVector<F>,
    /// The covariance matrix
    pub covariance: DMatrix<F>,
    /// The precision matrix (inverse of covariance)
    precision: DMatrix<F>,
    /// Determinant of covariance (cached)
    det_covariance: F,
    /// Dimension
    dim: usize,
}

impl<F> MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    /// Create a new multivariate normal distribution with the given mean and covariance.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean vector
    /// * `covariance` - The covariance matrix (must be positive definite)
    ///
    /// # Panics
    ///
    /// Panics if covariance is not square or if dimensions don't match.
    #[must_use]
    pub fn new(mean: DVector<F>, covariance: DMatrix<F>) -> Self {
        assert_eq!(
            covariance.nrows(),
            covariance.ncols(),
            "Covariance must be square"
        );
        assert_eq!(
            mean.len(),
            covariance.nrows(),
            "Mean and covariance dimensions must match"
        );

        // Compute matrix inverse - using nalgebra's built-in functions
        let Some(precision) = covariance.clone().try_inverse() else {
            panic!("Covariance matrix is not invertible")
        };

        // Compute determinant efficiently
        let det_covariance = covariance.determinant();
        let dim = mean.len();

        Self {
            mean,
            covariance,
            precision,
            det_covariance,
            dim,
        }
    }

    /// Create a standard multivariate normal distribution.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the distribution
    #[must_use]
    pub fn standard(dim: usize) -> Self {
        let mean = DVector::zeros(dim);
        let covariance = DMatrix::identity(dim, dim);
        Self::new(mean, covariance)
    }
}

impl<F> MeasureMarker for MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<F> Measure<DVector<F>> for MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    type RootMeasure = LebesgueMeasure<DVector<F>>;

    fn in_support(&self, _x: DVector<F>) -> bool {
        true // All vectors are in the support
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

// Natural parameters for MVN are (Σ^-1 μ, -0.5 Σ^-1)
// Sufficient statistics are (x, xx^T)
impl<F> ExponentialFamily<DVector<F>, F> for MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    // Using nalgebra types in the natural parameters and sufficient statistics
    type NaturalParam = (DVector<F>, DMatrix<F>); // (Σ^-1 μ, -0.5 Σ^-1)
    type SufficientStat = (DVector<F>, DMatrix<F>); // (x, xx^T)
    type BaseMeasure = LebesgueMeasure<DVector<F>>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let (eta1, eta2) = param;

        // Convert natural parameters back to standard parameters
        // η₂ = -0.5 Σ^-1, so Σ^-1 = -2η₂
        let neg_two = F::from(-2.0).unwrap();
        let precision_matrix = eta2.map(|x| neg_two * x);

        // Compute covariance as inverse of precision
        let Some(covariance_matrix) = precision_matrix.clone().try_inverse() else {
            panic!("Precision matrix is not invertible")
        };

        // Solve for mean: Σ^-1 μ = η₁, so μ = Σ η₁
        let mean_vector = &covariance_matrix * &eta1;

        Self::new(mean_vector, covariance_matrix)
    }

    fn to_natural(&self) -> Self::NaturalParam {
        // η₁ = Σ^-1 μ
        let eta1 = &self.precision * &self.mean;

        // η₂ = -0.5 Σ^-1
        let neg_half = F::from(-0.5).unwrap();
        let eta2 = self.precision.map(|x| neg_half * x);

        (eta1, eta2)
    }

    fn log_partition(&self) -> F {
        // A(η) = 0.5 μᵀΣ^-1μ + 0.5 log(|2πΣ|)
        let half = F::from(0.5).unwrap();

        // First term: 0.5 μᵀΣ^-1μ
        let quad_form = self.mean.dot(&(&self.precision * &self.mean));
        let term1 = half * quad_form;

        // Second term: 0.5 log(|2πΣ|)
        // log(|2πΣ|) = n*log(2π) + log(|Σ|)
        let two_pi = F::from(2.0).unwrap() * F::PI();
        let n = F::from(self.dim).unwrap();
        let log_det_cov = Float::ln(self.det_covariance);
        let term2 = half * (n * Float::ln(two_pi) + log_det_cov);

        term1 + term2
    }

    fn sufficient_statistic(&self, x: &DVector<F>) -> Self::SufficientStat {
        // First component: x
        let t1 = x.clone();

        // Second component: xx^T (outer product)
        let t2 = x * x.transpose();

        (t1, t2)
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::new()
    }
}

// Implement inner product for the MVN natural parameters and sufficient statistics
impl<F> InnerProduct<(DVector<F>, DMatrix<F>), F> for (DVector<F>, DMatrix<F>)
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    fn inner_product(&self, rhs: &(DVector<F>, DMatrix<F>)) -> F {
        let (eta1, eta2) = self;
        let (t1, t2) = rhs;

        // η₁ᵀT₁: dot product of first components
        let term1 = eta1.dot(t1);

        // trace(η₂ᵀT₂): component-wise multiplication and sum
        // Using the Frobenius inner product of matrices
        let term2 = eta2.component_mul(t2).sum();

        term1 + term2
    }
}

// Implement HasDensity
impl<F> HasDensity<DVector<F>> for MultivariateNormal<F>
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    fn log_density<'a>(&'a self, x: &'a DVector<F>) -> LogDensity<'a, DVector<F>, Self>
    where
        Self: Sized + Clone,
    {
        crate::core::measure::HasDensity::log_density_ef(self, x)
    }
}
