use num_traits::Float;

pub trait PrimitiveMeasure<T>: Clone {}

pub trait Measure<T> {
    type RootMeasure: Measure<T>;

    fn in_support(&self, x: T) -> bool;

    fn root_measure(&self) -> Self::RootMeasure;
}

pub trait Density<T> {
    type BaseMeasure: Measure<T>;

    fn log_density(&self, x: T) -> f64 {
        self.density(x).ln()
    }

    fn density(&self, x: T) -> f64 {
        self.log_density(x).exp()
    }
} 