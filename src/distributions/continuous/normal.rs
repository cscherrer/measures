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
use crate::exponential_family::traits::ExponentialFamily;
use crate::measures::primitive::lebesgue::LebesgueMeasure;
use num_traits::{Float, FloatConst};

/// Normal distribution N(μ, σ²).
///
/// This is a member of the exponential family with:
/// - Natural parameters: η = [μ/σ², -1/(2σ²)]
/// - Sufficient statistics: T(x) = [x, x²]
/// - Log partition: A(η) = -η₁²/(4η₂) - ½log(-2η₂) - ½log(2π)
/// - Base measure: Lebesgue measure (dx)
#[derive(Clone, Debug)]
pub struct Normal<T> {
    pub mean: T,
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

impl<T: Float + FloatConst> Normal<T> {
    /// Create a new normal distribution with given mean and standard deviation.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev > T::zero(), "Standard deviation must be positive");
        Self { mean, std_dev }
    }

    /// Get the variance σ²
    pub fn variance(&self) -> T {
        self.std_dev * self.std_dev
    }
}

impl<T: Float> Measure<T> for Normal<T> {
    type RootMeasure = LebesgueMeasure<T>;

    fn in_support(&self, _x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::<T>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<T, T> for Normal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 2];
    type SufficientStat = [T; 2];
    type BaseMeasure = LebesgueMeasure<T>;

    fn from_natural(param: Self::NaturalParam) -> Self {
        let [eta1, eta2] = param;
        let sigma2 = -(T::from(2.0).unwrap() * eta2).recip();
        let mu = eta1 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }

    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStat {
        [*x, *x * *x]
    }

    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::<T>::new()
    }

    fn natural_and_log_partition(&self) -> (Self::NaturalParam, T) {
        let sigma2 = self.variance();
        let mu2 = self.mean * self.mean;
        let inv_sigma2 = sigma2.recip();

        let natural_params = [self.mean * inv_sigma2, T::from(-0.5).unwrap() * inv_sigma2];

        let log_partition = (T::from(2.0).unwrap() * T::PI() * sigma2).ln() * T::from(0.5).unwrap()
            + T::from(0.5).unwrap() * mu2 * inv_sigma2;

        (natural_params, log_partition)
    }
}

// Symbolic optimization implementation
#[cfg(feature = "symbolic")]
impl<T> crate::exponential_family::symbolic::SymbolicOptimizer<T, T> for Normal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    fn symbolic_log_density(&self) -> crate::exponential_family::symbolic::SymbolicLogDensity {
        use crate::exponential_family::symbolic::utils::{symbolic_const, symbolic_var};
        use std::collections::HashMap;

        // Create symbolic variable
        let x = symbolic_var("x");

        // Build symbolic log-density: -½log(2πσ²) - (x-μ)²/(2σ²)
        let mu_f64 = self.mean.to_f64().unwrap();
        let sigma_f64 = self.std_dev.to_f64().unwrap();

        // Constant term: -½log(2πσ²)
        let two_pi_sigma_sq = 2.0 * std::f64::consts::PI * sigma_f64 * sigma_f64;
        let log_norm_const = -0.5 * two_pi_sigma_sq.ln();

        // Quadratic term: -(x-μ)²/(2σ²)
        let mu_expr = symbolic_const(mu_f64);
        let coeff = -1.0 / (2.0 * sigma_f64 * sigma_f64);
        let diff = x.clone() - mu_expr;
        let quadratic_term = symbolic_const(coeff) * diff.clone() * diff;

        // Complete expression
        let expr = symbolic_const(log_norm_const) + quadratic_term;

        // Store parameters
        let mut parameters = HashMap::new();
        parameters.insert("mu".to_string(), mu_f64);
        parameters.insert("sigma".to_string(), sigma_f64);
        parameters.insert("log_norm_const".to_string(), log_norm_const);
        parameters.insert("coeff".to_string(), coeff);

        crate::exponential_family::symbolic::SymbolicLogDensity::new(
            expr,
            parameters,
            vec!["x".to_string()],
        )
    }

    fn generate_optimized_function(
        &self,
    ) -> crate::exponential_family::symbolic::OptimizedFunction<T, T> {
        use std::collections::HashMap;

        // Pre-compute constants
        let mu_f64 = self.mean.to_f64().unwrap();
        let sigma_f64 = self.std_dev.to_f64().unwrap();
        let sigma_sq = sigma_f64 * sigma_f64;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_2sigma_sq = 1.0 / (2.0 * sigma_sq);

        // Convert back to T type
        let mu_t = T::from(mu_f64).unwrap();
        let log_norm_constant_t = T::from(log_norm_constant).unwrap();
        let inv_2sigma_sq_t = T::from(inv_2sigma_sq).unwrap();

        // Create optimized function
        let function = Box::new(move |x: &T| -> T {
            let diff = *x - mu_t;
            log_norm_constant_t - diff * diff * inv_2sigma_sq_t
        });

        // Store constants for documentation
        let mut constants = HashMap::new();
        constants.insert("mu".to_string(), mu_f64);
        constants.insert("sigma".to_string(), sigma_f64);
        constants.insert("log_norm_constant".to_string(), log_norm_constant);
        constants.insert("inv_2sigma_sq".to_string(), inv_2sigma_sq);

        let source_expression = format!(
            "Normal(μ={mu_f64}, σ={sigma_f64}): {log_norm_constant} - (x - {mu_f64})² * {inv_2sigma_sq}"
        );

        crate::exponential_family::symbolic::OptimizedFunction::new(
            function,
            constants,
            source_expression,
        )
    }
}

// JIT compilation implementation
#[cfg(feature = "jit")]
impl<T> crate::exponential_family::jit::JITOptimizer<T, T> for Normal<T>
where
    T: Float + FloatConst + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<crate::exponential_family::jit::JITFunction, crate::exponential_family::jit::JITError>
    {
        use crate::exponential_family::jit::JITCompiler;
        use crate::exponential_family::symbolic::{ConstantPool, SymbolicOptimizer};

        // Create the symbolic representation
        let symbolic = self.symbolic_log_density();

        // Build enhanced constant pool for JIT
        let mut constants = ConstantPool::new();

        let mu_f64 = self.mean.to_f64().unwrap();
        let sigma_f64 = self.std_dev.to_f64().unwrap();
        let sigma_sq = sigma_f64 * sigma_f64;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

        // Add constants with proper names expected by JIT compiler
        constants.constants.insert("mu".to_string(), mu_f64);
        constants.constants.insert("sigma".to_string(), sigma_f64);
        constants
            .constants
            .insert("log_norm_constant".to_string(), log_norm_constant);
        constants
            .constants
            .insert("inv_two_sigma_squared".to_string(), inv_two_sigma_sq);

        // Create JIT compiler and compile the expression
        let compiler = JITCompiler::new()?;
        compiler.compile_expression(&symbolic, &constants)
    }
}
