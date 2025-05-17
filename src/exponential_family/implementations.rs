//! Helper implementations for exponential family densities.

use num_traits::Float;
use std::marker::PhantomData;

use super::traits::{DotProduct, ExponentialFamily};
use crate::core::{LogDensity, Measure};

// Helper function to calculate log-density for any exponential family measure
pub fn exp_fam_log_density<'a, T: Float, M>(measure: &'a M, x: &'a T) -> LogDensity<'a, T, M>
where
    M: ExponentialFamily<T> + Measure<T> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    measure.log_density_ef(x)
}

// We'll use a specialization helper to avoid conflicts with dirac implementation
pub struct ExponentialFamilyDensity<'a, T: Float, M>(pub LogDensity<'a, T, M>, PhantomData<M>)
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = crate::core::True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>;

impl<'a, T: Float, M> From<ExponentialFamilyDensity<'a, T, M>> for f64
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = crate::core::True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    fn from(wrapper: ExponentialFamilyDensity<'a, T, M>) -> Self {
        let val = wrapper.0;
        let eta = val.measure.to_natural();
        let t = val.measure.sufficient_statistic(val.x);
        let a = val.measure.log_partition();
        let h = val.measure.carrier_measure(val.x);

        let result =
            <M::NaturalParam as DotProduct<M::SufficientStat, T>>::dot(&eta, &t) - a + h.ln();
        result.to_f64().unwrap()
    }
}

// Helper function to compute exponential family log density
#[must_use]
pub fn compute_exp_fam_log_density<T: Float, M>(log_density: LogDensity<'_, T, M>) -> f64
where
    M: ExponentialFamily<T> + Measure<T, IsExponentialFamily = crate::core::True> + Clone,
    M::NaturalParam: DotProduct<M::SufficientStat, T>,
{
    ExponentialFamilyDensity(log_density, PhantomData).into()
}
