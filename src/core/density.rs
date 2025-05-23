use super::measure::Measure;
use std::ops::Neg;

/// A trait representing the log-density between two measures. Goals for the
/// design:
/// - for a log-density `l` we should be able to say `l.wrt(new_base_measure)`
///   and get a new log-density having the same measure but a different base
///   measure
/// - `l.at(x)` should represent the log-density of the measure at the point `x`
///   with respect to the base measure
/// - `l.at(x)` should be able to be called multiple times with the same `x` and
///   get the same result
/// - We should have a way to cache computations when useful, so repeated
///   computation of `l.at(...)` is as fast as possible
/// - Computations must be static and stack-allocated when possible
/// - log-density computations should not be specifically f64 or any other
///   concrete type. Instead we should be able to say e.g. `let x: f64 =
///   m.log_density().wrt(b).at(x)` and get an `f64` or any other Float type
/// - If base measure is not specified, it should default to the root measure
/// - There's an algebra for log-densities:
///   - If `l` is the log-density of `m` with respect to `b` then `-l` is the
///     log-density of `b` with respect to `m`
///   - If `l1` is the log-density of `m1` with respect to `m2` and `l2` is the
///     log-density of `m2` with respect to `m3` then `l1 + l2` is the
///     log-density of `m1` with respect to `m3`
///
/// # Design Philosophy
///
/// The design splits functionality between:
/// 1. **`LogDensityTrait<T>`** - Core interface for what a log-density is (mathematical relationship)
/// 2. **`EvaluateAt<T, F>`** - Evaluation capability that can work with different numeric types
/// 3. **`LogDensity<T, M, B>`** - Builder type for constructing log-densities
/// 4. **Algebraic combinators** - Types for manipulating log-densities
///
/// This split provides several advantages:
/// - **Zero-cost abstractions**: Type system tracks measure relationships at compile time
/// - **Static dispatch**: Each combination gets its own optimized implementation
/// - **Algebraic operations**: Can be implemented as type-level combinators
/// - **Ergonomic API**: Fluent interface feels natural to use
/// - **Generic evaluation**: Same log-density can be evaluated with f64, f32, dual numbers, etc.
///
/// # Usage Examples
///
/// ```rust,ignore
/// use measures::{Normal, LogDensity, Measure};
///
/// let normal = Normal::new(0.0, 1.0);
/// let x = 0.5;
///
/// // Same log-density, different numeric types
/// let ld = normal.log_density();
/// let f64_result: f64 = ld.at(&x);           // Regular evaluation
/// let f32_result: f32 = ld.at(&(x as f32));  // Lower precision
/// let dual_result: Dual64 = ld.at(&dual_x);  // Autodiff with dual numbers
///
/// // With different base measure  
/// let lebesgue = LebesgueMeasure::new();
/// let ld2 = normal.log_density().wrt(lebesgue);
/// let value2: f64 = ld2.at(&x);
///
/// // Algebraic operations
/// let ld_neg = -ld;       // Negated log-density
/// let ld_sum = ld + ld2;  // Sum of log-densities (chain rule)
///
/// // Caching for repeated evaluations
/// let ld_cached = normal.log_density().cached();
/// for &xi in &[0.1, 0.2, 0.1, 0.3, 0.1] {  // 0.1 computed only once
///     let _val: f64 = ld_cached.at(&xi);
/// }
/// ```
///
/// This separation is essential for modern scientific computing where you want to evaluate the same mathematical object with different number systems.
///
/// # Automatic Computation for Shared Root Measures
///
/// When two measures share the same root measure, log-densities between them are automatically
/// computed using the mathematical relationship:
///
/// ```rust,ignore
/// // If normal1 and normal2 both have LebesgueMeasure as their root:
/// let normal1 = Normal::new(0.0, 1.0);
/// let normal2 = Normal::new(1.0, 2.0);
///
/// // This is automatically computed as normal1.log_density() - normal2.log_density()
/// let ld = normal1.log_density().wrt(normal2);
/// let value: f64 = ld.at(&x);  // = log(dnormal1/dlebesgue) - log(dnormal2/dlebesgue)
/// ```
///
/// This leverages the mathematical fact that:
/// `log(dm1/dm2) = log(dm1/root) - log(dm2/root)` when both measures share the same root.
pub trait LogDensityTrait<T> {
    type Measure: Measure<T>;
    type BaseMeasure: Measure<T>;

    /// Get the measure whose log-density this represents
    fn measure(&self) -> &Self::Measure;

    /// Get the base measure with respect to which the density is computed
    fn base_measure(&self) -> &Self::BaseMeasure;
}

/// Trait for evaluating log-densities at specific points with different numeric types.
///
/// This separation allows the same log-density to be evaluated with different
/// numeric types:
/// - `f64` for regular computation
/// - `f32` for lower precision  
/// - `Dual64` for forward-mode autodiff
/// - Custom number types for other specialized computation
pub trait EvaluateAt<T, F> {
    /// Evaluate the log-density at point x, returning type F
    fn at(&self, x: &T) -> F;
}

/// A builder for computing log-densities of a measure.
///
/// This type is the entry point for log-density computations. It implements
/// the fluent interface: `measure.log_density().wrt(base).at(x)`
#[derive(Clone)]
pub struct LogDensity<T, M, B = <M as Measure<T>>::RootMeasure>
where
    T: Clone,
    M: Measure<T> + Clone,
    B: Measure<T> + Clone,
{
    /// The measure whose log-density we're computing
    pub measure: M,
    /// The base measure with respect to which we're computing the log-density
    pub base_measure: B,
    /// Phantom data for T
    _phantom: std::marker::PhantomData<T>,
}

impl<T, M> LogDensity<T, M, <M as Measure<T>>::RootMeasure>
where
    T: Clone,
    M: Measure<T> + Clone,
{
    /// Create a new log-density computation with the root measure as base.
    pub fn new(measure: M) -> Self {
        let base_measure = measure.root_measure();
        Self {
            measure,
            base_measure,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, M, B> LogDensity<T, M, B>
where
    T: Clone,
    M: Measure<T> + Clone,
    B: Measure<T> + Clone,
{
    /// Specify a different base measure for this log-density computation.
    pub fn wrt<NewB: Measure<T> + Clone>(self, base_measure: NewB) -> LogDensity<T, M, NewB> {
        LogDensity {
            measure: self.measure,
            base_measure,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Evaluate the log-density at a specific point.
    ///
    /// The return type is inferred from context or can be specified explicitly:
    /// ```rust,ignore
    /// let result: f64 = log_density.at(&x);        // f64 inferred
    /// let result: f32 = log_density.at(&x);        // f32 inferred  
    /// let result = log_density.at::<f64>(&x);      // f64 explicit
    /// ```
    pub fn at<F>(&self, x: &T) -> F
    where
        Self: EvaluateAt<T, F>,
    {
        EvaluateAt::at(self, x)
    }
}

impl<T, M, B> LogDensityTrait<T> for LogDensity<T, M, B>
where
    T: Clone,
    M: Measure<T> + Clone,
    B: Measure<T> + Clone,
{
    type Measure = M;
    type BaseMeasure = B;

    fn measure(&self) -> &Self::Measure {
        &self.measure
    }

    fn base_measure(&self) -> &Self::BaseMeasure {
        &self.base_measure
    }
}

/// Default implementation when measure and base measure share the same root measure.
///
/// This implements the mathematical relationship:
/// `log(dm1/dm2) = log(dm1/root) - log(dm2/root)`
/// when `m1` and `m2` have the same root measure.
impl<T, M1, M2, F> EvaluateAt<T, F> for LogDensity<T, M1, M2>
where
    T: Clone,
    M1: Measure<T> + HasLogDensity<T, F> + Clone,
    M2: Measure<T, RootMeasure = M1::RootMeasure> + HasLogDensity<T, F> + Clone,
    F: std::ops::Sub<Output = F>,
{
    fn at(&self, x: &T) -> F {
        // log(dm1/dm2) = log(dm1/root) - log(dm2/root)
        self.measure.log_density_wrt_root(x) - self.base_measure.log_density_wrt_root(x)
    }
}

/// A helper trait for measures that can provide their own log-density evaluation logic.
///
/// This allows different measure types to implement custom evaluation for different
/// numeric types while keeping the main traits clean.
pub trait LogDensityEval<T, F> {
    fn evaluate_log_density(&self, base_measure: &impl Measure<T>, x: &T) -> F;
}

/// A trait for measures that can compute their log-density with respect to their root measure.
///
/// This enables default implementations for log-densities between measures that share
/// the same root measure: `log(dm1/dm2) = log(dm1/root) - log(dm2/root)`
///
/// # Example Implementation
///
/// ```rust,ignore
/// // For a Normal distribution with LebesgueMeasure as root:
/// impl<T> HasLogDensity<T, f64> for Normal<T>
/// where T: Into<f64> + Clone
/// {
///     fn log_density_wrt_root(&self, x: &T) -> f64 {
///         let x_val: f64 = x.clone().into();
///         let normalized = (x_val - self.mean) / self.std_dev;
///         -0.5 * normalized * normalized - (2.0 * std::f64::consts::PI * self.std_dev.powi(2)).ln() / 2.0
///     }
/// }
/// ```
pub trait HasLogDensity<T, F>: Measure<T> {
    /// Compute the log-density of this measure with respect to its root measure
    fn log_density_wrt_root(&self, x: &T) -> F;
}

/// Algebraic operation: Negation (swaps measure and base measure)
pub struct NegLogDensity<L> {
    inner: L,
}

impl<L, T> LogDensityTrait<T> for NegLogDensity<L>
where
    L: LogDensityTrait<T>,
    T: Clone,
{
    type Measure = L::BaseMeasure;
    type BaseMeasure = L::Measure;

    fn measure(&self) -> &Self::Measure {
        self.inner.base_measure()
    }

    fn base_measure(&self) -> &Self::BaseMeasure {
        self.inner.measure()
    }
}

impl<L, T, F> EvaluateAt<T, F> for NegLogDensity<L>
where
    L: EvaluateAt<T, F>,
    F: Neg<Output = F>,
    T: Clone,
{
    fn at(&self, x: &T) -> F {
        -self.inner.at(x)
    }
}

impl<T, M, B> Neg for LogDensity<T, M, B>
where
    T: Clone,
    M: Measure<T> + Clone,
    B: Measure<T> + Clone,
{
    type Output = NegLogDensity<Self>;

    fn neg(self) -> Self::Output {
        NegLogDensity { inner: self }
    }
}

/// Cached log-density for repeated evaluations with a specific numeric type
pub struct CachedLogDensity<L, T, F>
where
    T: Clone + std::hash::Hash + Eq,
    F: Clone,
    L: LogDensityTrait<T>,
{
    inner: L,
    cache: std::cell::RefCell<std::collections::HashMap<T, F>>,
    _phantom: std::marker::PhantomData<F>,
}

impl<L, T, F> CachedLogDensity<L, T, F>
where
    L: LogDensityTrait<T>,
    T: Clone + std::hash::Hash + Eq,
    F: Clone,
{
    pub fn new(inner: L) -> Self {
        Self {
            inner,
            cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<L, T, F> LogDensityTrait<T> for CachedLogDensity<L, T, F>
where
    L: LogDensityTrait<T>,
    T: Clone + std::hash::Hash + Eq,
    F: Clone,
{
    type Measure = L::Measure;
    type BaseMeasure = L::BaseMeasure;

    fn measure(&self) -> &Self::Measure {
        self.inner.measure()
    }

    fn base_measure(&self) -> &Self::BaseMeasure {
        self.inner.base_measure()
    }
}

impl<L, T, F> EvaluateAt<T, F> for CachedLogDensity<L, T, F>
where
    L: LogDensityTrait<T> + EvaluateAt<T, F>,
    T: Clone + std::hash::Hash + Eq,
    F: Clone,
{
    fn at(&self, x: &T) -> F {
        let mut cache = self.cache.borrow_mut();
        if let Some(cached_value) = cache.get(x) {
            cached_value.clone()
        } else {
            let value = self.inner.at(x);
            cache.insert(x.clone(), value.clone());
            value
        }
    }
}

/// Extension trait to add caching capability for a specific numeric type
pub trait LogDensityCaching<T>: LogDensityTrait<T> + Sized
where
    T: Clone,
{
    fn cached_for<F: Clone>(self) -> CachedLogDensity<Self, T, F>
    where
        T: std::hash::Hash + Eq,
    {
        CachedLogDensity::new(self)
    }

    /// Convenience method for caching with f64
    fn cached(self) -> CachedLogDensity<Self, T, f64>
    where
        T: std::hash::Hash + Eq,
    {
        self.cached_for::<f64>()
    }
}

impl<L, T> LogDensityCaching<T> for L
where
    L: LogDensityTrait<T>,
    T: Clone,
{
}
