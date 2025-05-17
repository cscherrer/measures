//! Multivariate Normal distribution implementation.
//!
//! This module provides the Multivariate Normal (Gaussian) distribution, which is
//! defined over n-dimensional real vectors.

use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{ExponentialFamily, ExponentialFamilyMeasure, InnerProduct};
use crate::measures::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

// Simple vector type for demonstration
// In a real implementation, you'd use a proper vector/matrix library
#[derive(Clone, Debug)]
pub struct Vector<F: Float> {
    pub data: Vec<F>,
}

impl<F: Float> Vector<F> {
    #[must_use]
    pub fn new(data: Vec<F>) -> Self {
        Self { data }
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> F {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Vectors must have the same length"
        );
        let mut sum = F::zero();
        for i in 0..self.data.len() {
            sum = sum + self.data[i] * other.data[i];
        }
        sum
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// Simple matrix type for demonstration
#[derive(Clone, Debug)]
pub struct Matrix<F: Float> {
    pub data: Vec<Vec<F>>, // Row-major order
    pub rows: usize,
    pub cols: usize,
}

impl<F: Float> Matrix<F> {
    #[must_use]
    pub fn new(data: Vec<Vec<F>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        // Ensure all rows have the same length
        for row in &data {
            assert_eq!(row.len(), cols, "All rows must have the same length");
        }
        Self { data, rows, cols }
    }

    // Create identity matrix
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            data[i][i] = F::one();
        }
        Self {
            data,
            rows: n,
            cols: n,
        }
    }

    // Matrix-vector multiplication
    #[must_use]
    pub fn multiply_vec(&self, v: &Vector<F>) -> Vector<F> {
        assert_eq!(
            self.cols,
            v.len(),
            "Matrix columns must match vector length"
        );
        let mut result = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] = result[i] + self.data[i][j] * v.data[j];
            }
        }
        Vector { data: result }
    }

    // Compute determinant (for simple matrices)
    #[must_use]
    pub fn determinant(&self) -> F {
        assert_eq!(self.rows, self.cols, "Matrix must be square");

        if self.rows == 1 {
            return self.data[0][0];
        } else if self.rows == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }

        // For simplicity, we'll only implement det for 1x1 and 2x2 matrices
        // In practice, you'd use a proper linear algebra library
        unimplemented!("Determinant for matrices larger than 2x2 not implemented")
    }

    // Compute inverse (for simple matrices)
    #[must_use]
    pub fn inverse(&self) -> Self {
        assert_eq!(self.rows, self.cols, "Matrix must be square");

        if self.rows == 1 {
            return Self::new(vec![vec![F::one() / self.data[0][0]]]);
        } else if self.rows == 2 {
            let det = self.determinant();
            assert!(det != F::zero(), "Matrix is singular");

            let a = self.data[0][0];
            let b = self.data[0][1];
            let c = self.data[1][0];
            let d = self.data[1][1];

            let inv_det = F::one() / det;
            return Self::new(vec![
                vec![d * inv_det, -b * inv_det],
                vec![-c * inv_det, a * inv_det],
            ]);
        }

        // For simplicity, we'll only implement inverse for 1x1 and 2x2 matrices
        unimplemented!("Inverse for matrices larger than 2x2 not implemented")
    }
}

/// A multivariate normal (Gaussian) distribution.
///
/// The multivariate normal distribution is characterized by its mean vector
/// and covariance matrix.
#[derive(Clone)]
pub struct MultivariateNormal<F: Float> {
    /// The mean vector
    pub mean: Vector<F>,
    /// The covariance matrix
    pub covariance: Matrix<F>,
    /// The precision matrix (inverse of covariance)
    precision: Matrix<F>,
    /// Dimension
    dim: usize,
}

impl<F: Float + FloatConst> MultivariateNormal<F> {
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
    pub fn new(mean: Vector<F>, covariance: Matrix<F>) -> Self {
        assert_eq!(
            covariance.rows, covariance.cols,
            "Covariance must be square"
        );
        assert_eq!(
            mean.len(),
            covariance.rows,
            "Mean and covariance dimensions must match"
        );

        let precision = covariance.inverse(); // In practice, compute this more carefully
        let dim = mean.len();

        Self {
            mean,
            covariance,
            precision,
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
        let mean = Vector::new(vec![F::zero(); dim]);
        let covariance = Matrix::identity(dim);
        Self::new(mean, covariance)
    }
}

impl<F: Float> MeasureMarker for MultivariateNormal<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<F: Float> Measure<Vector<F>> for MultivariateNormal<F> {
    type RootMeasure = LebesgueMeasure<Vector<F>>;

    fn in_support(&self, _x: Vector<F>) -> bool {
        true // All vectors are in the support
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

// Natural parameters for MVN are (Σ^-1 μ, -0.5 Σ^-1)
// Sufficient statistics are (x, xx^T)
impl<F: Float + FloatConst> ExponentialFamily<Vector<F>, F> for MultivariateNormal<F> {
    // For simplicity, we'll use tuples of our vector/matrix types
    type NaturalParam = (Vector<F>, Matrix<F>); // (Σ^-1 μ, -0.5 Σ^-1)
    type SufficientStat = (Vector<F>, Matrix<F>); // (x, xx^T)

    fn from_natural(param: Self::NaturalParam) -> Self {
        let (eta1, eta2) = param;

        // Convert natural parameters back to standard parameters
        // η₂ = -0.5 Σ^-1, so Σ^-1 = -2η₂
        let neg_two = F::from(-2.0).unwrap();

        // This is simplified; in practice handle this conversion more carefully
        let precision_matrix = Matrix::new(
            eta2.data
                .iter()
                .map(|row| row.iter().map(|&val| neg_two * val).collect())
                .collect(),
        );

        let covariance_matrix = precision_matrix.inverse();
        let mean_vector = covariance_matrix.multiply_vec(&eta1);

        Self::new(mean_vector, covariance_matrix)
    }

    fn to_natural(&self) -> Self::NaturalParam {
        // η₁ = Σ^-1 μ
        let eta1 = self.precision.multiply_vec(&self.mean);

        // η₂ = -0.5 Σ^-1
        let neg_half = F::from(-0.5).unwrap();
        let eta2 = Matrix::new(
            self.precision
                .data
                .iter()
                .map(|row| row.iter().map(|&val| neg_half * val).collect())
                .collect(),
        );

        (eta1, eta2)
    }

    fn log_partition(&self) -> F {
        // A(η) = 0.5 μᵀΣ^-1μ + 0.5 log(|2πΣ|)
        let half = F::from(0.5).unwrap();

        // First term: 0.5 μᵀΣ^-1μ
        let mu_precision_mu = self.mean.dot(&self.precision.multiply_vec(&self.mean));
        let term1 = half * mu_precision_mu;

        // Second term: 0.5 log(|2πΣ|)
        // log(|2πΣ|) = n*log(2π) + log(|Σ|)
        let two_pi = F::from(2.0).unwrap() * F::PI();
        let n = F::from(self.dim).unwrap();
        let log_det_cov = self.covariance.determinant().ln(); // In practice, compute this more carefully
        let term2 = half * (n * two_pi.ln() + log_det_cov);

        term1 + term2
    }

    fn sufficient_statistic(&self, x: &Vector<F>) -> Self::SufficientStat {
        // First component: x
        let t1 = x.clone();

        // Second component: xx^T (outer product)
        let n = x.len();
        let mut t2_data = vec![vec![F::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                t2_data[i][j] = x.data[i] * x.data[j];
            }
        }
        let t2 = Matrix::new(t2_data);

        (t1, t2)
    }

    fn carrier_measure(&self, _x: &Vector<F>) -> F {
        F::one() // h(x) = 1 for multivariate normal
    }
}

// Implement inner product for the MVN natural parameters and sufficient statistics
impl<F: Float> InnerProduct<(Vector<F>, Matrix<F>), F> for (Vector<F>, Matrix<F>) {
    fn inner_product(&self, rhs: &(Vector<F>, Matrix<F>)) -> F {
        let (eta1, eta2) = self;
        let (t1, t2) = rhs;

        // η₁ᵀT₁: dot product of first components
        let term1 = eta1.dot(t1);

        // trace(η₂ᵀT₂): sum of element-wise products
        let mut term2 = F::zero();
        for i in 0..eta2.rows {
            for j in 0..eta2.cols {
                term2 = term2 + eta2.data[i][j] * t2.data[i][j];
            }
        }

        term1 + term2
    }
}

// Mark MultivariateNormal as an exponential family measure
impl<F: Float + FloatConst> ExponentialFamilyMeasure<Vector<F>, F> for MultivariateNormal<F> {}

// Implement HasDensity
impl<F: Float + FloatConst> HasDensity<Vector<F>> for MultivariateNormal<F> {
    fn log_density<'a>(&'a self, x: &'a Vector<F>) -> LogDensity<'a, Vector<F>, Self>
    where
        Self: Sized + Clone,
    {
        self.log_density_ef(x)
    }
}

// Implement From for LogDensity to f64
impl<F: Float + FloatConst> From<LogDensity<'_, Vector<F>, MultivariateNormal<F>>> for f64 {
    fn from(val: LogDensity<'_, Vector<F>, MultivariateNormal<F>>) -> Self {
        let x = val.x;
        let mvn = val.measure;

        // Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
        let centered = Vector::new(
            x.data
                .iter()
                .zip(mvn.mean.data.iter())
                .map(|(&xi, &mui)| xi - mui)
                .collect(),
        );

        let quad_form = centered.dot(&mvn.precision.multiply_vec(&centered));

        // log(p(x)) = -0.5 * [n*log(2π) + log(|Σ|) + (x-μ)ᵀΣ⁻¹(x-μ)]
        let n = F::from(mvn.dim).unwrap();
        let two_pi = F::from(2.0).unwrap() * F::PI();
        let log_det_cov = mvn.covariance.determinant().ln();

        let result = F::from(-0.5).unwrap() * (n * two_pi.ln() + log_det_cov + quad_form);

        result.to_f64().unwrap()
    }
}
