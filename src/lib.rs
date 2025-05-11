use num_traits::Float;

trait PrimitiveMeasure<T>: Clone {}

#[derive(Clone)]
struct LebesgueMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Float> PrimitiveMeasure<T> for LebesgueMeasure<T> {}

#[derive(Clone)]
struct CountingMeasure<T: Clone> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Clone> PrimitiveMeasure<T> for CountingMeasure<T> {}

trait Measure<T> {
    type RootMeasure: Measure<T>;

    fn in_support(&self, x: T) -> bool;

    fn root_measure(&self) -> Self::RootMeasure;
}

impl<T, P: PrimitiveMeasure<T>> Measure<T> for P {
    type RootMeasure = Self;

    fn in_support(&self, x: T) -> bool {
        true
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.clone()
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

struct Dirac<T: PartialEq> {
    x: T,
}

impl<T: PartialEq + Clone> Measure<T> for Dirac<T> {
    type RootMeasure = CountingMeasure<T>;

    fn in_support(&self, x: T) -> bool {
        self.x == x
    }

    fn root_measure(&self) -> Self::RootMeasure {
        CountingMeasure::<T> {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: PartialEq + Clone> Density<T> for Dirac<T> {
    type BaseMeasure = CountingMeasure<T>;

    fn log_density(&self, x: T) -> f64 {
        0.0
    }

    fn density(&self, x: T) -> f64 {
        1.0
    }
}
