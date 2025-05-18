//! Poisson distribution implementation.
//!
//! This module provides the Poisson distribution, which is a discrete probability
//! distribution that expresses the probability of a given number of events occurring
//! in a fixed interval of time or space.

use crate::core::types::{ExponentialFamily as EFMethod, Specialized};
use crate::core::{False, HasDensity, LogDensity, Measure, MeasureMarker, True};
use crate::exponential_family::{ExponentialFamily, ExponentialFamilyMeasure, InnerProduct};
use crate::measures::counting::CountingMeasure;
use num_traits::{Float, FloatConst};

/// A Poisson distribution.
///
/// The Poisson distribution has a single parameter lambda (rate) and
/// is defined over non-negative integers.
#[derive(Clone)]
pub struct Poisson<F: Float> {
    /// The rate parameter (expected number of occurrences)
    pub lambda: F,
}

impl<F: Float> Poisson<F> {
    /// Create a new Poisson distribution with the given rate.
    ///
    /// # Arguments
    ///
    /// * `lambda` - The rate parameter (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not positive.
    #[must_use]
    pub fn new(lambda: F) -> Self {
        assert!(lambda > F::zero(), "Rate parameter must be positive");
        Self { lambda }
    }
}

impl<F: Float> MeasureMarker for Poisson<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = True;
    type PreferredLogDensityMethod = Specialized;
}

impl<F: Float> Measure<u64> for Poisson<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, _x: u64) -> bool {
        true // All non-negative integers are in the support
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::new()
    }
}

// Natural parameter for Poisson is log(lambda), sufficient statistic is k
impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    type NaturalParam = F; // η = log(λ)
    type SufficientStat = u64; // T(x) = x

    fn from_natural(param: Self::NaturalParam) -> Self {
        Self::new(param.exp())
    }

    fn to_natural(&self) -> Self::NaturalParam {
        self.lambda.ln()
    }

    fn log_partition(&self) -> F {
        self.lambda
    }

    fn sufficient_statistic(&self, x: &u64) -> Self::SufficientStat {
        *x
    }

    fn carrier_measure(&self, x: &u64) -> F {
        // h(x) = 1/x!
        let mut factorial = F::one();
        for i in 1..=*x {
            factorial = factorial * F::from(i).unwrap();
        }
        F::one() / factorial
    }
}

// Implement inner product for scalar natural parameter and u64 sufficient statistic
impl<F: Float> InnerProduct<u64, F> for F {
    fn inner_product(&self, rhs: &u64) -> F {
        *self * F::from(*rhs).unwrap()
    }
}

// Mark Poisson as an exponential family measure
impl<F: Float + FloatConst> ExponentialFamilyMeasure<u64, F> for Poisson<F> {}

// Implement HasDensity
impl<F: Float + FloatConst> HasDensity<u64> for Poisson<F> {
    fn log_density<'a>(&'a self, x: &'a u64) -> LogDensity<'a, u64, Self>
    where
        Self: Sized + Clone,
    {
        self.log_density_specialized(x)
    }

    fn log_density_specialized<'a>(&'a self, x: &'a u64) -> LogDensity<'a, u64, Self>
    where
        Self: Sized + Clone,
    {
        LogDensity::new(self, x)
    }

    fn log_density_ef<'a>(&'a self, x: &'a u64) -> LogDensity<'a, u64, Self>
    where
        Self: Sized + Clone,
    {
        // Invoke the base Log density creation
        LogDensity::new(self, x)
    }
}

// Implement From for LogDensity to f64
impl<F: Float + FloatConst> From<LogDensity<'_, u64, Poisson<F>>> for f64 {
    fn from(val: LogDensity<'_, u64, Poisson<F>>) -> Self {
        let k = *val.x;
        let lambda = val.measure.lambda;

        let k_f = F::from(k).unwrap();

        // PMF: P(X = k) = (e^-λ * λ^k) / k!
        // Log-PMF: -λ + k*log(λ) - log(k!)
        let mut log_factorial = F::zero();
        for i in 1..=k {
            log_factorial = log_factorial + F::from(i).unwrap().ln();
        }

        let result = -lambda + k_f * lambda.ln() - log_factorial;
        result.to_f64().unwrap()
    }
}

// Implement specialized From for LogDensityWithMethod<_, _, _, _, Specialized>
impl<F: Float + FloatConst>
    From<
        crate::core::density::LogDensityWithMethod<
            '_,
            u64,
            Poisson<F>,
            CountingMeasure<u64>,
            Specialized,
        >,
    > for f64
{
    fn from(
        val: crate::core::density::LogDensityWithMethod<
            '_,
            u64,
            Poisson<F>,
            CountingMeasure<u64>,
            Specialized,
        >,
    ) -> Self {
        // Use the specialized implementation
        let inner = val.log_density;
        let k = *inner.x;
        let lambda = inner.measure.lambda;

        let k_f = F::from(k).unwrap();

        // PMF: P(X = k) = (e^-λ * λ^k) / k!
        // Log-PMF: -λ + k*log(λ) - log(k!)
        let mut log_factorial = F::zero();
        for i in 1..=k {
            log_factorial = log_factorial + F::from(i).unwrap().ln();
        }

        let result = -lambda + k_f * lambda.ln() - log_factorial;
        result.to_f64().unwrap()
    }
}

// Implement exponential family From for LogDensityWithMethod<_, _, _, _, ExponentialFamily>
impl<F: Float + FloatConst>
    From<
        crate::core::density::LogDensityWithMethod<
            '_,
            u64,
            Poisson<F>,
            CountingMeasure<u64>,
            EFMethod,
        >,
    > for f64
{
    fn from(
        val: crate::core::density::LogDensityWithMethod<
            '_,
            u64,
            Poisson<F>,
            CountingMeasure<u64>,
            EFMethod,
        >,
    ) -> Self {
        // Use the exponential family implementation
        let inner = val.log_density;
        let poisson = inner.measure;
        let x = inner.x;

        let eta = poisson.to_natural();
        let t = poisson.sufficient_statistic(x);
        let a = poisson.log_partition();
        let h = poisson.carrier_measure(x);

        let result = eta.inner_product(&t) - a + h.ln();
        result.to_f64().unwrap()
    }
}
