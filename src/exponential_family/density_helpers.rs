//! Helper functions for computing log-densities
//! 
//! This module provides utility functions for computing log-densities in a 
//! consistent, non-redundant way across different distribution types.

use crate::exponential_family::{ExponentialFamily, ExponentialFamilyMeasure, InnerProduct};
use num_traits::{Float, FloatConst};
use nalgebra::{DVector, DMatrix};
use std::fmt::Debug;
use nalgebra::{RealField, ComplexField, Scalar};

/// Compute log-density for any exponential family measure
/// 
/// This provides a common implementation that distributions can use instead of
/// duplicating the calculation logic in each distribution.
pub fn compute_ef_log_density<X, F, M>(measure: &M, x: &X) -> f64 
where
    F: Float,
    X: Clone,
    M: ExponentialFamilyMeasure<X, F>,
    M::NaturalParam: InnerProduct<M::SufficientStat, F>,
{
    let eta = measure.to_natural();
    let t = measure.sufficient_statistic(x);
    let a = measure.log_partition();
    let h = measure.carrier_measure(x);

    (eta.inner_product(&t) - a + h.ln()).to_f64().unwrap()
}

/// Compute log-density for normal distribution
pub fn compute_normal_log_density<T: Float + FloatConst>(mean: T, std_dev: T, x: T) -> f64 {
    let sigma2 = std_dev * std_dev;
    let diff = x - mean;
    let norm_constant = -(T::from(2.0).unwrap() * T::PI() * sigma2).ln() / T::from(2.0).unwrap();
    let exponent = -(diff * diff) / (T::from(2.0).unwrap() * sigma2);

    (norm_constant + exponent).to_f64().unwrap()
}

/// Compute log-density for standard normal distribution
pub fn compute_stdnormal_log_density<T: Float + FloatConst>(x: T) -> f64 {
    let norm_constant = -(T::from(2.0).unwrap() * T::PI()).ln() / T::from(2.0).unwrap();
    let exponent = -(x * x) / T::from(2.0).unwrap();
    
    (norm_constant + exponent).to_f64().unwrap()
}

/// Compute log-density for multivariate normal distribution
pub fn compute_mvn_log_density<F>(
    mean: &DVector<F>, 
    precision: &DMatrix<F>, 
    det_covariance: F,
    dim: usize,
    x: &DVector<F>
) -> f64 
where
    F: Float + FloatConst + RealField + ComplexField + Scalar + Debug + Clone,
{
    // Centered vector: (x-μ)
    let centered = x - mean;

    // Mahalanobis distance: (x-μ)ᵀΣ⁻¹(x-μ)
    let quad_form = centered.dot(&(precision * &centered));

    // log(p(x)) = -0.5 * [n*log(2π) + log(|Σ|) + (x-μ)ᵀΣ⁻¹(x-μ)]
    let n = F::from(dim).unwrap();
    let two_pi = F::from(2.0).unwrap() * F::PI();
    let log_det_cov = Float::ln(det_covariance);

    let result = F::from(-0.5).unwrap() * (n * Float::ln(two_pi) + log_det_cov + quad_form);

    result.to_f64().unwrap()
} 