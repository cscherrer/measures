use num_traits::Float;

pub trait PrimitiveMeasure<T>: Clone {}

pub trait Measure<T> {
    type RootMeasure: Measure<T>;

    fn in_support(&self, x: T) -> bool;

    fn root_measure(&self) -> Self::RootMeasure;
}

// New builder types for density calculations
#[derive(Clone)]
pub struct DensityBuilder<'a, T: Clone, M: Measure<T> + Clone> {
    pub measure: &'a M,
    pub x: T,
}

#[derive(Clone)]
pub struct DensityWithRespectTo<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> {
    pub measure: &'a M1,
    pub base_measure: &'a M2,
    pub x: T,
}

// Trait for types that can compute densities
pub trait HasDensity<T>: Measure<T> {
    fn density(&self, x: T) -> DensityBuilder<'_, T, Self> 
    where 
        Self: Sized + Clone,
        T: Clone
    {
        DensityBuilder {
            measure: self,
            x,
        }
    }
}

// Implement HasDensity for all measures that can compute densities
impl<T: Clone, M: Measure<T> + Clone> HasDensity<T> for M {}

// Implementation for DensityBuilder
impl<'a, T: Clone, M: Measure<T> + Clone> DensityBuilder<'a, T, M> {
    pub fn wrt<M2: Measure<T> + Clone>(self, base_measure: &'a M2) -> DensityWithRespectTo<'a, T, M, M2> {
        DensityWithRespectTo {
            measure: self.measure,
            base_measure,
            x: self.x,
        }
    }
}

// Add some convenience methods for common operations
impl<'a, T: Clone, M1: Measure<T> + Clone, M2: Measure<T> + Clone> DensityWithRespectTo<'a, T, M1, M2> {
    pub fn log(&self) -> f64 
    where 
        Self: Into<f64> + Clone
    {
        let density = Into::<f64>::into(self.clone());
        density.ln()
    }
} 