# Proposal: Drop Caching from Exponential Family Framework

## Summary

Remove all caching infrastructure from the exponential family framework in favor of symbolic optimization as the primary performance optimization strategy.

## Rationale

### Performance Evidence
- **Symbolic optimization**: 12.91x speedup demonstrated
- **Caching**: Typically 2-3x speedup at best
- **Symbolic is clearly superior** and represents the future direction

### Architectural Benefits
- **Massive simplification**: Remove entire caching infrastructure
- **Cleaner APIs**: No cache parameters throughout the codebase
- **Better focus**: Concentrate on symbolic optimization and JIT compilation
- **Reduced maintenance**: Far fewer code paths and complexity

### Strategic Alignment
- **Future-oriented**: Symbolic → JIT compilation roadmap
- **Performance-first**: Best optimization strategy becomes the primary one
- **Developer experience**: Much simpler API for users

## What Changes

### Simplified ExponentialFamily Trait

```rust
pub trait ExponentialFamily<X: Clone, F: Float>: Clone {
    /// The natural parameter type
    type NaturalParam: Clone;
    /// The sufficient statistic type
    type SufficientStat;
    /// The base measure type
    type BaseMeasure: Measure<X> + Clone;

    /// Convert from natural parameters to standard parameters
    fn from_natural(param: Self::NaturalParam) -> Self;

    /// Convert from standard parameters to natural parameters
    fn to_natural(&self) -> Self::NaturalParam {
        self.natural_and_log_partition().0
    }

    /// Compute the log partition function A(η)
    fn log_partition(&self) -> F {
        self.natural_and_log_partition().1
    }

    /// Compute both natural parameters and log partition function efficiently
    fn natural_and_log_partition(&self) -> (Self::NaturalParam, F) {
        (self.to_natural(), self.log_partition())
    }

    /// Compute the sufficient statistic T(x)
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;

    /// Get the base measure for this exponential family
    fn base_measure(&self) -> Self::BaseMeasure;

    /// Exponential family log-density computation with automatic chain rule
    fn exp_fam_log_density(&self, x: &X) -> F
    where
        Self::NaturalParam: DotProduct<Self::SufficientStat, Output = F>,
        Self::BaseMeasure: HasLogDensity<X, F>,
    {
        let (natural_params, log_partition) = self.natural_and_log_partition();
        let sufficient_stats = self.sufficient_statistic(x);

        // Standard exponential family part: η·T(x) - A(η)
        let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;

        // Chain rule part: log-density of base measure with respect to root measure
        let chain_rule_part = self.base_measure().log_density_wrt_root(x);

        // Complete log-density: exponential family + chain rule
        exp_fam_part + chain_rule_part
    }
}
```

### Simplified Distribution Implementations

```rust
impl<T: Float + FloatConst> ExponentialFamily<T, T> for Normal<T> {
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

        let natural_params = [
            self.mean * inv_sigma2,
            T::from(-0.5).unwrap() * inv_sigma2,
        ];

        let log_partition = (T::from(2.0).unwrap() * T::PI() * sigma2).ln() 
            * T::from(0.5).unwrap() + T::from(0.5).unwrap() * mu2 * inv_sigma2;

        (natural_params, log_partition)
    }
}
```

### Performance Optimization Path

```rust
// Standard evaluation
let normal = Normal::new(2.0, 1.5);
let result = normal.log_density().at(&x);  // Works, but slower

// Symbolic optimization (current)
let optimized = normal.symbolic_optimize();
let result = optimized.call(&x);  // 12.91x faster!

// JIT compilation (future)
let jit_fn = normal.compile_jit()?;
let result = jit_fn.call(&x);  // Even faster with native code!
```

## Removed Components

### Traits
- `PrecomputeCache<X, F>`
- `ExponentialFamilyCache<X, F>`
- `GenericExpFamImpl<X, F>`

### Structs
- `GenericExpFamCache<D, X, F>`
- `StdNormalCache<T>`
- All distribution-specific cache types

### Methods
- `precompute_cache()`
- `cached_log_density()`
- `cached_log_density_batch()`
- `cached_log_density_fn()`

### Files
- `src/exponential_family/cache_trait.rs`
- `src/exponential_family/generic_cache.rs`
- `examples/cached_exponential_family.rs`
- `examples/generic_implementation_demo.rs`

## Benefits

### Code Simplification
- **~1000 lines removed** from caching infrastructure
- **Cleaner trait hierarchy** with clear responsibilities
- **Simpler distribution implementations** (no cache boilerplate)
- **Better API ergonomics** (no cache parameters to manage)

### Performance Clarity
- **Clear optimization path**: Standard → Symbolic → JIT
- **No confusing middle ground** with partial optimizations
- **Focus on best-in-class solutions** rather than incremental improvements

### Development Velocity
- **Faster compilation** (less code to compile)
- **Easier testing** (fewer code paths)
- **Simpler debugging** (no cache-related bugs)
- **Better documentation** (focus on core concepts)

### Future Readiness
- **Symbolic optimization** becomes the primary optimization strategy
- **JIT compilation** path is cleaner without cache interference
- **Better foundation** for advanced optimizations

## Migration Path

### Phase 1: Deprecation
1. Mark all cache-related traits and methods as `#[deprecated]`
2. Update documentation to recommend symbolic optimization
3. Add warnings in examples about upcoming removal

### Phase 2: Removal
1. Remove all cache-related code
2. Simplify exponential family traits
3. Update all distribution implementations
4. Update examples and documentation

### Phase 3: Enhancement
1. Expand symbolic optimization to more distributions
2. Implement JIT compilation with Cranelift
3. Add advanced symbolic optimizations

## Risk Assessment

### Low Risk
- **Breaking change**: Caching was never a public API commitment
- **Performance regression**: Symbolic optimization is much faster
- **Maintenance**: Less code to maintain is always better

### Mitigation
- **Clear migration guide** for any users depending on caching
- **Better performance** through symbolic optimization
- **Future-proof API** focused on best optimization strategies

## Conclusion

Dropping caching simplifies the codebase significantly while focusing on the superior symbolic optimization approach. This creates a cleaner foundation for future JIT compilation and provides better performance through the 12.91x faster symbolic optimization.

The change aligns with the project's direction toward cutting-edge optimization techniques while eliminating complexity that doesn't provide best-in-class performance. 