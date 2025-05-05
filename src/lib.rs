use num_traits::Float;

trait PrimitiveMeasure<T>: Clone + Copy {}

#[derive(Clone, Copy)]
struct LebesgueMeasure;

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure {}

#[derive(Clone, Copy)]
struct CountingMeasure;

impl<T> PrimitiveMeasure<T> for CountingMeasure {}

trait Measure<T> {
    type RootMeasure: Measure<T>;

    fn in_support(&self, x: T) -> bool;

    fn root_measure(&self) -> Self::RootMeasure;
}

impl<T, U: PrimitiveMeasure<T>> Measure<T> for U {
    type RootMeasure = Self;

    fn in_support(&self, x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        *self
    }
}

trait Density<T> {
    type BaseMeasure: Measure<T>;

    fn log_density(&self, x: T) -> f64 {
        self.density(x).ln()
    }

    fn density(&self, x: T) -> f64 {
        self.log_density(x).exp()
    }
}

impl<T, U: PrimitiveMeasure<T>> Density<T> for U {
    type BaseMeasure = U;

    fn log_density(&self, x: T) -> f64 {
        0.0
    }

    fn density(&self, x: T) -> f64 {
        1.0
    }
}
