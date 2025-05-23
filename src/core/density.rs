use super::measure::Measure;
use crate::core::types::True;
use crate::exponential_family::ExponentialFamily;
use num_traits::Float;

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
    #[inline]
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
    #[inline]
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
    #[inline]
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

/// Base trait for measures that can compute their log-density with respect to their root measure.
///
/// This trait provides automatic specialization based on `IsExponentialFamily`:
/// - Exponential families get optimized cached computation automatically  
/// - Other measures implement the trait manually
///
/// This design mirrors how `PrimitiveMeasure` splits on `IsPrimitive`.
pub trait HasLogDensity<T, F> {
    /// Compute the log-density of this measure with respect to its root measure
    fn log_density_wrt_root(&self, x: &T) -> F;
}

/// Specialized automatic implementation for exponential family measures.
///
/// This provides optimized cached computation using the exponential family formula:
/// log p(x|θ) = η·T(x) - A(η) + log h(x)
impl<T: Clone, F, M> HasLogDensity<T, F> for M
where
    M: Measure<T, IsExponentialFamily = True> + ExponentialFamily<T, F> + Clone,
    F: Float,
    M::NaturalParam: crate::traits::DotProduct<M::SufficientStat, Output = F> + Clone,
    M::BaseMeasure: HasLogDensity<T, F>,
{
    #[inline]
    fn log_density_wrt_root(&self, x: &T) -> F {
        // Use cached exponential family computation automatically
        let cache = self.precompute_cache();
        self.cached_log_density(&cache, x)
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
    #[inline]
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

/// A trait for computing log-densities.
///
/// This trait provides a builder pattern for computing log-densities
/// with explicit control over the reference measure.
pub trait LogDensityBuilder<T>
where
    T: Clone,
{
    /// Get the log-density with respect to the root measure.
    ///
    /// # Example
    ///
    /// ```rust
    /// use measures::{Normal, LogDensityBuilder};
    ///
    /// let normal = Normal::new(0.0, 1.0);
    /// let ld = normal.log_density();
    /// let log_density_value: f64 = ld.at(&0.0);
    /// ```
    fn log_density(&self) -> LogDensity<T, Self>
    where
        Self: Measure<T> + Clone;
}

impl<T, M> LogDensityBuilder<T> for M
where
    T: Clone,
    M: Measure<T>,
{
    fn log_density(&self) -> LogDensity<T, Self>
    where
        Self: Clone,
    {
        LogDensity::new(self.clone())
    }
}

/// Helper trait for shared root computation optimization.
///
/// When multiple measures share the same root measure, this trait provides
/// utilities for efficient batch computation of log-densities.
pub trait SharedRootMeasure<T, F>
where
    T: Clone,
{
    /// The shared root measure type
    type SharedRoot: Measure<T>;

    /// Get the shared root measure
    fn shared_root(&self) -> Self::SharedRoot;

    /// Compute log-density with respect to the shared root measure
    fn log_density_wrt_shared_root(&self, x: &T) -> F;
}

/// Automatic implementation of `SharedRootMeasure` for measures with the same root.
impl<T, F, M1, M2> SharedRootMeasure<T, F> for (M1, M2)
where
    T: Clone,
    M1: Measure<T> + HasLogDensity<T, F>,
    M2: Measure<T, RootMeasure = M1::RootMeasure> + HasLogDensity<T, F>,
    F: Float,
{
    type SharedRoot = M1::RootMeasure;

    fn shared_root(&self) -> Self::SharedRoot {
        self.0.root_measure()
    }

    fn log_density_wrt_shared_root(&self, x: &T) -> F {
        let density1 = self.0.log_density_wrt_root(x);
        let density2 = self.1.log_density_wrt_root(x);
        density1 + density2
    }
}

/// Marker trait for measures that can be used in log-density computations.
pub trait DensityMeasure<T, F>: Measure<T> + HasLogDensity<T, F>
where
    T: Clone,
{
}

impl<T, F, M> DensityMeasure<T, F> for M
where
    T: Clone,
    M: Measure<T> + HasLogDensity<T, F>,
{
}

/// Helper function for computing log-densities in a functional style.
pub fn log_density_at<T, F, M>(measure: &M, x: &T) -> F
where
    T: Clone,
    M: HasLogDensity<T, F>,
{
    measure.log_density_wrt_root(x)
}

/// Batch computation helper for multiple points.
pub fn log_density_batch<T, F, M>(measure: &M, points: &[T]) -> Vec<F>
where
    T: Clone,
    M: HasLogDensity<T, F>,
{
    points
        .iter()
        .map(|x| measure.log_density_wrt_root(x))
        .collect()
}
