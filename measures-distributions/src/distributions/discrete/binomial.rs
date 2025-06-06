//! Binomial distribution implementation.
//!
//! This module provides the Binomial distribution, which is a discrete probability
//! distribution that models the number of successes in n independent Bernoulli trials.
//! The density is computed with respect to counting measure.
//!
//! Note: This is only an exponential family when n is fixed and known.
//!
//! # Example
//!
//! ```rust
//! use measures::distributions::Binomial;
//! use measures::LogDensityBuilder;
//!
//! let binomial = Binomial::new(10, 0.3); // n = 10 trials, success probability p = 0.3
//!
//! // Compute log-density at x = 3 (3 successes out of 10 trials)
//! let ld = binomial.log_density();
//! let log_density_value: f64 = ld.at(&3);
//! ```

use measures_combinators::measures::derived::binomial_coefficient::BinomialCoefficientMeasure;
use measures_core::primitive::counting::CountingMeasure;
use measures_core::{False, True};
use measures_core::{Measure, MeasureMarker};
use measures_exponential_family::ExponentialFamily;
use num_traits::Float;

/// Binomial distribution Binomial(n, p) with fixed n.
///
/// This is a member of the exponential family when n is fixed with:
/// - Natural parameter: η = log(p/(1-p)) (log-odds)
/// - Sufficient statistic: T(x) = x
/// - Log partition: A(η) = n * log(1 + exp(η))
/// - Base measure: Binomial coefficient measure (binomial coefficient term)
#[derive(Clone, Debug)]
pub struct Binomial<T> {
    pub n: u64,  // Number of trials (fixed)
    pub prob: T, // Success probability p
}

impl<T: Float> MeasureMarker for Binomial<T> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
}

impl<T: Float> Binomial<T> {
    /// Create a new binomial distribution with given number of trials and success probability.
    pub fn new(n: u64, prob: T) -> Self {
        assert!(
            prob >= T::zero() && prob <= T::one(),
            "Probability must be in [0, 1]"
        );
        Self { n, prob }
    }

    /// Get the mean n*p
    pub fn mean(&self) -> T {
        T::from(self.n).unwrap() * self.prob
    }

    /// Get the variance n*p*(1-p)
    pub fn variance(&self) -> T {
        let n_t = T::from(self.n).unwrap();
        n_t * self.prob * (T::one() - self.prob)
    }

    /// Get the log-odds log(p/(1-p))
    pub fn log_odds(&self) -> T {
        (self.prob / (T::one() - self.prob)).ln()
    }
}

impl<T: Float> Measure<u64> for Binomial<T> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        x <= self.n
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<u64>::new()
    }
}

// Exponential family implementation
impl<T> ExponentialFamily<u64, T> for Binomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    type NaturalParam = [T; 1];
    type SufficientStat = [T; 1];
    type BaseMeasure = BinomialCoefficientMeasure<T>;

    fn from_natural(_param: Self::NaturalParam) -> Self {
        // This requires knowing n, which we can't determine from just η
        // In practice, you'd need to specify n separately
        panic!("Cannot construct Binomial from natural parameter without knowing n");
    }

    fn sufficient_statistic(&self, x: &u64) -> <Self as ExponentialFamily<u64, T>>::SufficientStat {
        [T::from(*x).unwrap()]
    }

    fn base_measure(&self) -> <Self as ExponentialFamily<u64, T>>::BaseMeasure {
        BinomialCoefficientMeasure::<T>::new(self.n)
    }

    fn natural_and_log_partition(&self) -> (<Self as ExponentialFamily<u64, T>>::NaturalParam, T) {
        let natural_param = [self.log_odds()];

        // Log partition: A(η) = n * log(1 + exp(η))
        let n_t = T::from(self.n).unwrap();
        let log_partition = n_t * (T::one() + natural_param[0].exp()).ln();

        (natural_param, log_partition)
    }
}

// JIT optimization implementation
#[cfg(feature = "jit")]
impl<T> measures_exponential_family::JITOptimizer<u64, T> for Binomial<T>
where
    T: Float + std::fmt::Debug + 'static,
{
    fn compile_jit(
        &self,
    ) -> Result<measures_exponential_family::JITFunction, measures_exponential_family::JITError>
    {
        let _prob_f64 = self.prob.to_f64().unwrap();
        let _n_f64 = self.n as f64;
        let _log_odds = (_prob_f64 / (1.0 - _prob_f64)).ln();
        let _log_partition = _n_f64 * (1.0 + _log_odds.exp()).ln();

        // For now, return an error since compile_function is not available
        // This can be implemented later when the JIT infrastructure is complete
        Err(
            measures_exponential_family::JITError::UnsupportedExpression(
                "Binomial distribution JIT compilation not yet implemented".to_string(),
            ),
        )
    }
}

// Implementation of HasLogDensity for Binomial distribution
impl<T: Float> measures_core::HasLogDensity<u64, T> for Binomial<T> {
    fn log_density_wrt_root(&self, x: &u64) -> T {
        if *x <= self.n {
            // Binomial PMF: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
            // log P(X = k) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
            let k = T::from(*x).unwrap();
            let n = T::from(self.n).unwrap();

            // Use log binomial coefficient (this is handled by the base measure)
            k * self.prob.ln() + (n - k) * (T::one() - self.prob).ln()
        } else {
            T::neg_infinity() // Outside support
        }
    }
}
