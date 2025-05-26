//! Student's t-distribution implementation.
//!
//! The Student's t-distribution is a continuous probability distribution that is not
//! an exponential family. It has heavier tails than the normal distribution and
//! reduces to the Cauchy distribution when degrees of freedom = 1.
//!
//! # Examples
//!
//! ```rust
//! use measures::{StudentT, LogDensityBuilder};
//!
//! let t_dist = StudentT::new(3.0);  // 3 degrees of freedom
//! let log_density: f64 = t_dist.log_density().at(&0.0);
//! ```

use measures_core::False;
use measures_core::float_constant;
use measures_core::primitive::lebesgue::LebesgueMeasure;
use measures_core::{DecompositionBuilder, HasLogDensityDecomposition, LogDensityDecomposition};
use measures_core::{HasLogDensity, Measure, MeasureMarker};
use num_traits::{Float, FloatConst};
use special::Gamma;

/// Student's t-distribution with degrees of freedom parameter `nu`.
///
/// The Student's t-distribution has probability density function:
/// f(x|ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
///
/// Log-density decomposition:
/// log f(x|ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - (ν+1)/2 * log(1 + x²/ν)
///            = `f_param(ν)` + `f_mixed(x,ν)`
///
/// Where:
/// - `f_param(ν)` = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) (parameter-only)
/// - `f_mixed(x,ν)` = -(ν+1)/2 * log(1 + x²/ν) (mixed data-parameter)
///
/// This is NOT an exponential family distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT<T> {
    nu: T, // degrees of freedom
}

/// Parameters for Student's t distribution: just the degrees of freedom
pub type StudentTParams<T> = T;

impl<T: Float> StudentT<T> {
    /// Create a new Student's t-distribution with given degrees of freedom.
    ///
    /// # Arguments
    /// * `nu` - Degrees of freedom parameter (must be positive)
    ///
    /// # Panics
    /// Panics if nu <= 0
    pub fn new(nu: T) -> Self {
        assert!(nu > T::zero(), "Degrees of freedom must be positive");
        Self { nu }
    }

    /// Get the degrees of freedom parameter.
    pub fn nu(&self) -> T {
        self.nu
    }

    /// Get parameters (just nu for Student's t).
    pub fn params(&self) -> StudentTParams<T> {
        self.nu
    }

    /// Compute the log probability density function at point x.
    pub fn log_pdf(&self, x: T) -> T
    where
        T: Float + std::fmt::Debug,
    {
        // Convert to f64 for gamma function computation
        let nu_f64 = self.nu.to_f64().unwrap();
        let x_f64 = x.to_f64().unwrap();

        // Compute log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ)
        let (log_gamma_half_nu_plus_1, _) = f64::midpoint(nu_f64, 1.0).ln_gamma();
        let (log_gamma_half_nu, _) = (nu_f64 / 2.0).ln_gamma();
        let log_normalization = log_gamma_half_nu_plus_1
            - log_gamma_half_nu
            - 0.5 * (nu_f64 * std::f64::consts::PI).ln();

        // Compute -(ν+1)/2 * log(1 + x²/ν)
        let log_kernel = -f64::midpoint(nu_f64, 1.0) * (1.0 + x_f64 * x_f64 / nu_f64).ln();

        T::from(log_normalization + log_kernel).unwrap()
    }

    /// Compute log-density for IID samples efficiently using decomposition.
    ///
    /// For n IID samples, this is much more efficient than computing individual densities.
    pub fn log_density_iid(&self, samples: &[T]) -> T
    where
        T: Float + FloatConst + std::fmt::Debug + std::iter::Sum,
    {
        let decomp = self.log_density_decomposition();
        decomp.evaluate_iid(samples, &self.params())
    }
}

impl<T: Float> MeasureMarker for StudentT<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = False; // Student's t is NOT an exponential family
}

impl<T: Float + Clone> Measure<T> for StudentT<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true // Student's t has support on all real numbers
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

/// Manual implementation of `HasLogDensity` for Student's t since it's not an exponential family.
impl<T: Float + FloatConst + std::fmt::Debug> HasLogDensity<T, T> for StudentT<T> {
    fn log_density_wrt_root(&self, x: &T) -> T {
        self.log_pdf(*x)
    }
}

/// Implementation of structured log-density decomposition for Student's t.
///
/// This shows how non-exponential families can still benefit from the structured approach.
impl<T: Float + FloatConst + std::fmt::Debug> HasLogDensityDecomposition<T, StudentTParams<T>, T>
    for StudentT<T>
{
    fn log_density_decomposition(&self) -> LogDensityDecomposition<T, StudentTParams<T>, T> {
        DecompositionBuilder::new()
            // Parameter-only term: log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ)
            .param_term(
                |nu: &StudentTParams<T>| {
                    let nu_f64 = nu.to_f64().unwrap();
                    let (log_gamma_half_nu_plus_1, _) = f64::midpoint(nu_f64, 1.0).ln_gamma();
                    let (log_gamma_half_nu, _) = (nu_f64 / 2.0).ln_gamma();
                    let log_normalization = log_gamma_half_nu_plus_1
                        - log_gamma_half_nu
                        - 0.5 * (nu_f64 * std::f64::consts::PI).ln();
                    T::from(log_normalization).unwrap()
                },
                "log normalization constant",
            )
            // Mixed term: -(ν+1)/2 * log(1 + x²/ν)
            .mixed_term(
                |x: &T, nu: &StudentTParams<T>| {
                    let nu_f64 = nu.to_f64().unwrap();
                    let x_f64 = x.to_f64().unwrap();
                    let log_kernel =
                        -f64::midpoint(nu_f64, 1.0) * (1.0 + x_f64 * x_f64 / nu_f64).ln();
                    T::from(log_kernel).unwrap()
                },
                "negative scaled log of one plus standardized squared",
            )
            .build()
    }
}

impl<T: Float> Default for StudentT<T> {
    fn default() -> Self {
        Self::new(float_constant::<T>(1.0)) // Default to Cauchy distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use measures_core::LogDensityBuilder;

    #[test]
    fn test_student_t_creation() {
        let t_dist = StudentT::new(3.0);
        assert_eq!(t_dist.nu(), 3.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_nu() {
        let _t_dist = StudentT::new(-1.0); // Should panic
    }

    #[test]
    fn test_log_density_builder() {
        let t_dist = StudentT::new(3.0);

        // Test that the builder pattern works
        let log_density: f64 = t_dist.log_density().at(&0.0);

        // At x = 0, should be finite
        assert!(log_density.is_finite());
    }

    #[test]
    fn test_log_density_decomposition() {
        let t_dist = StudentT::new(5.0);
        let decomp = t_dist.log_density_decomposition();

        let x = 1.5;
        let params = t_dist.params();

        // Test that decomposition matches direct computation
        let decomp_result = decomp.evaluate(&x, &params);
        let direct_result = t_dist.log_pdf(x);

        assert!((decomp_result - direct_result).abs() < 1e-10);
    }

    #[test]
    fn test_iid_log_density() {
        let t_dist = StudentT::new(4.0);
        let samples = vec![0.0, 1.0, -1.0, 0.5];

        // Test IID computation
        let iid_result = t_dist.log_density_iid(&samples);

        // Manual computation
        let manual_result: f64 = samples.iter().map(|&x| t_dist.log_pdf(x)).sum();

        assert!((iid_result - manual_result).abs() < 1e-10);
    }

    #[test]
    fn test_decomposition_efficiency() {
        let t_dist = StudentT::new(3.0);
        let decomp = t_dist.log_density_decomposition();
        let params = t_dist.params();

        // Test that we can evaluate parameter terms separately
        let param_terms = decomp.evaluate_param_terms(&params);

        // This should be independent of data
        assert!(param_terms.is_finite());

        // The parameter term should be the log normalization constant
        // For ν=3: log Γ(2) - log Γ(1.5) - 0.5*log(3π)
        let expected_param = {
            let (log_gamma_2, _) = 2.0f64.ln_gamma();
            let (log_gamma_1_5, _) = 1.5f64.ln_gamma();
            log_gamma_2 - log_gamma_1_5 - 0.5 * (3.0 * std::f64::consts::PI).ln()
        };

        assert!((param_terms - expected_param).abs() < 1e-10);
    }

    #[test]
    fn test_support() {
        let t_dist = StudentT::new(3.0);

        // Student's t has support on all real numbers
        assert!(t_dist.in_support(0.0));
        assert!(t_dist.in_support(100.0));
        assert!(t_dist.in_support(-100.0));
    }

    #[test]
    fn test_cauchy_limit() {
        let t_dist = StudentT::new(1.0); // Should be Cauchy
        let cauchy = crate::distributions::continuous::cauchy::Cauchy::new(0.0, 1.0);

        // At x = 0, both should give similar results
        let t_density: f64 = t_dist.log_density().at(&0.0);
        let cauchy_density: f64 = cauchy.log_density().at(&0.0);

        // Should be approximately equal (within numerical precision)
        assert!((t_density - cauchy_density).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_optimization_scenario() {
        // Simulate parameter optimization: data is fixed, parameters change
        let samples = vec![0.5, -0.3, 1.2, -0.8, 0.1];

        // Test different degrees of freedom
        let nus = vec![1.0, 2.0, 5.0, 10.0];

        for &nu in &nus {
            let t_dist = StudentT::new(nu);
            let decomp = t_dist.log_density_decomposition();

            // In parameter optimization, we'd cache data terms and only recompute parameter terms
            let param_terms = decomp.evaluate_param_terms(&nu);

            // Compute mixed terms for each sample
            let mixed_sum: f64 = samples
                .iter()
                .map(|&x| {
                    decomp
                        .mixed_terms
                        .iter()
                        .map(|term| (term.compute)(&x, &nu))
                        .sum::<f64>()
                })
                .sum();

            let total_via_decomp = param_terms * (samples.len() as f64) + mixed_sum;
            let total_direct = t_dist.log_density_iid(&samples);

            assert!((total_via_decomp - total_direct).abs() < 1e-10);
        }
    }
}
