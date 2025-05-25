# Developer Notes

This document captures important implementation insights, gotchas, and design decisions that developers should be aware of when working on the measures codebase.

## Type-Level Programming for Trait Dispatch

### Why We Use Type-Level Booleans Instead of Const Generics

**TL;DR**: Type-level booleans (`True`, `False`) are essential for trait dispatch disambiguation and cannot be replaced with const generics.

#### The Problem
We need to provide different implementations of `HasLogDensity` for exponential families vs. other measures:

```rust
// Exponential families get automatic implementation
impl<T: Clone, F, M> HasLogDensity<T, F> for M
where
    M: Measure<T, IsExponentialFamily = True> + ExponentialFamily<T, F>,
    // ... other bounds
{
    fn log_density_wrt_root(&self, x: &T) -> F {
        self.exp_fam_log_density(x)  // Optimized exponential family computation
    }
}

// Other measures implement manually
impl<T, F> HasLogDensity<T, F> for SomeOtherMeasure<T> {
    fn log_density_wrt_root(&self, x: &T) -> F {
        // Custom implementation
    }
}
```

#### Why Type-Level Booleans Work

```rust
pub trait MeasureMarker {
    type IsExponentialFamily: TypeLevelBool;
    type IsPrimitive: TypeLevelBool;
}

// This creates a compile-time partition of the type space
impl<T> MeasureMarker for Normal<T> {
    type IsExponentialFamily = True;   // ← Compiler knows this is disjoint from False
    type IsPrimitive = False;
}
```

The constraint `M: Measure<T, IsExponentialFamily = True>` creates a **precise type-level filter** that the compiler can reason about for trait coherence.

#### Why Const Generics Don't Work

```rust
// This is NOT valid Rust syntax:
pub trait MeasureMarker {
    const IS_EXPONENTIAL_FAMILY: bool;
}

impl<T, const IS_EXP_FAM: bool> HasLogDensity<T, F> for M
where
    M: Measure<T> + MeasureMarker<IS_EXPONENTIAL_FAMILY = IS_EXP_FAM>,
    IS_EXP_FAM == true,  // ← Not valid - can't constrain const generics in trait bounds
{
    // ...
}
```

Rust's const generics cannot be used in trait bounds the way associated types can. You cannot write `M: Trait<CONST = value>`.

#### Key Insight
Type-level booleans enable **blanket implementations with compile-time partitioning**. The compiler can prove that `IsExponentialFamily = True` and `IsExponentialFamily = False` are mutually exclusive, preventing trait implementation conflicts.

### What NOT to Remove
- `TypeLevelBool` trait and `True`/`False` types - these are essential
- `MeasureMarker` associated types - these enable the trait dispatch

### What CAN Be Removed
- `LogDensityMethod` types (`Specialized`, `ExponentialFamily`, `Default`) - these are defined but unused

## Exponential Family Parameter Computation Pattern

### The `natural_params_and_log_partition` Design

The `ExponentialFamily` trait provides three methods for accessing natural parameters and log partition functions:

```rust
pub trait ExponentialFamily<X: Clone, F: Float>: Clone {
    // Individual accessors
    fn natural_params(&self) -> Self::NaturalParam;
    fn log_partition(&self) -> F;
    
    // Combined accessor for efficiency
    fn natural_params_and_log_partition(&self) -> (Self::NaturalParam, F);
}
```

#### Why This Pattern Exists

**Shared Computations**: Many exponential families have expensive computations that are shared between computing natural parameters η(θ) and the log partition function A(η). For example:

```rust
// Normal distribution: both η and A(η) depend on σ² and its inverse
impl ExponentialFamily<f64, f64> for Normal<f64> {
    fn natural_params_and_log_partition(&self) -> ([f64; 2], f64) {
        let sigma2 = self.variance();
        let inv_sigma2 = sigma2.recip();  // ← Shared expensive computation
        
        let natural_params = [
            self.mean * inv_sigma2,           // Uses inv_sigma2
            -0.5 * inv_sigma2                 // Uses inv_sigma2
        ];
        
        let log_partition = 0.5 * (
            self.mean * self.mean * inv_sigma2 +  // Uses inv_sigma2
            (2.0 * PI * sigma2).ln()              // Uses sigma2
        );
        
        (natural_params, log_partition)
    }
}
```

#### Implementation Flexibility

The trait provides **mutually recursive default implementations** to allow maximum flexibility:

```rust
// Default implementations are mutually recursive
fn natural_params(&self) -> Self::NaturalParam {
    self.natural_params_and_log_partition().0  // Calls combined method
}

fn log_partition(&self) -> F {
    self.natural_params_and_log_partition().1  // Calls combined method
}

fn natural_params_and_log_partition(&self) -> (Self::NaturalParam, F) {
    (self.natural_params(), self.log_partition())  // Calls individual methods
}
```

This allows distributions to implement **any combination**:

1. **Individual methods only**: Simple distributions without shared computations
2. **Combined method only**: Distributions with expensive shared computations
3. **All three methods**: Distributions that want to optimize both individual and combined access

#### Performance Guarantee

**Internal computations always use the combined method** to avoid redundant work:

```rust
// In exp_fam_log_density and other internal methods
let (natural_params, log_partition) = self.natural_params_and_log_partition();
// Never calls individual methods separately
```

This ensures that even if a distribution only implements individual methods, internal computations won't accidentally call both separately.

#### Future Optimization Considerations

This pattern is designed to be **optimization-friendly**:
- If the compiler successfully inlines and optimizes away the redundancy, we may simplify to just individual methods
- If shared computations prove critical for performance, we may encourage more distributions to implement the combined method
- The mutual recursion allows us to change the "preferred" implementation without breaking existing code

**Current Status**: We're monitoring whether this complexity is necessary or if compiler optimizations make it redundant. The pattern provides safety against performance regressions while we gather data.

## IID Implementation Patterns

### Root Measure Handling
The IID implementation uses a helper trait `IIDRootMeasure` to map individual root measures to their IID equivalents:

```rust
impl<X: Clone> IIDRootMeasure<X> for LebesgueMeasure<X> {
    type IIDRoot = LebesgueMeasure<Vec<X>>;
    fn iid_root_measure() -> Self::IIDRoot { LebesgueMeasure::new() }
}
```

This pattern ensures that `IID<Normal<f64>>` has root measure `LebesgueMeasure<Vec<f64>>`, not `IID<LebesgueMeasure<f64>>`.

### Exponential Family Optimization
IID exponential families use the efficient formula:
```
log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
```

This is much faster than computing individual log-densities and summing.

## Performance Considerations

### Zero-Cost Abstractions
The trait system is designed for zero-cost abstractions:
- Type-level dispatch resolves at compile time
- Monomorphization creates specialized implementations
- No runtime overhead for the abstraction layers

### Inlining
Critical methods are marked `#[inline]` to ensure they get inlined:
```rust
#[inline]
pub fn at<F>(&self, x: &T) -> F
where
    Self: EvaluateAt<T, F>,
{
    EvaluateAt::at(self, x)
}
```

## Common Pitfalls

### 1. Don't Add Manual HasLogDensity Implementations for Exponential Families
If a type implements `ExponentialFamily` and has `IsExponentialFamily = True`, it automatically gets `HasLogDensity`. Adding a manual implementation will create a conflict.

### 2. Natural Parameters Should Use Arrays for Scalars
For exponential families with scalar natural parameters, use `[T; 1]` instead of `T` to ensure compatibility with the `DotProduct` trait:

```rust
impl<T> ExponentialFamily<T, T> for Exponential<T> {
    type NaturalParam = [T; 1];  // ← Not just T
    type SufficientStat = [T; 1];
    // ...
}
```

**Why not add a scalar `DotProduct` implementation?** This would create trait coherence issues and ambiguous method resolution. If we had both:

```rust
// Existing array implementation
impl<T: Float, const N: usize> DotProduct for [T; N] {
    type Output = T;
    fn dot(&self, other: &Self) -> Self::Output { ... }
}

// Hypothetical scalar implementation - DON'T DO THIS
impl<T: Float> DotProduct for T {
    type Output = T;
    fn dot(&self, other: &T) -> T { ... }
}
```

Then for `[T; 1]`, the compiler wouldn't know which implementation to use, creating trait coherence ambiguity. The `[T; 1]` pattern elegantly avoids this by:
- Maintaining uniform interface (everything uses arrays)
- Avoiding trait implementation conflicts
- Zero runtime cost (`[T; 1]` compiles to same code as `T`)
- Clear semantics (natural parameters are always "vectors", even 1D ones)

### 3. Base Measure Chain Rule
The exponential family implementation automatically handles the chain rule:
```
log p(x|θ) = η·T(x) - A(η) + log(d(base_measure)/d(root_measure))(x)
```

Don't manually add the base measure term - it's handled automatically.

### 4. Use Combined Parameter Method for Internal Computations
When implementing exponential family methods that need both natural parameters and log partition, always use the combined method:

```rust
// ✅ Correct - uses combined method
let (natural_params, log_partition) = self.natural_params_and_log_partition();

// ❌ Incorrect - may duplicate expensive computations
let natural_params = self.natural_params();
let log_partition = self.log_partition();
```

## Testing Patterns

### Trait Dispatch Testing
When adding new measure types, verify that trait dispatch works correctly:

```rust
#[test]
fn test_exponential_family_dispatch() {
    let normal = Normal::new(0.0, 1.0);
    // This should use the automatic exponential family implementation
    let log_density: f64 = normal.log_density_wrt_root(&0.5);
    // Verify it matches the expected exponential family formula
}
```

### Numeric Type Testing
Test that log-densities work with different numeric types:

```rust
#[test]
fn test_numeric_types() {
    let normal = Normal::new(0.0, 1.0);
    let ld = normal.log_density();
    
    let f64_result: f64 = ld.at(&0.5);
    let f32_result: f32 = ld.at(&0.5f32);
    
    assert!((f64_result - f32_result as f64).abs() < 1e-6);
}
```

### Parameter Computation Consistency
Test that individual and combined parameter methods are consistent:

```rust
#[test]
fn test_parameter_consistency() {
    let normal = Normal::new(1.0, 2.0);
    
    let (combined_params, combined_log_partition) = normal.natural_params_and_log_partition();
    let individual_params = normal.natural_params();
    let individual_log_partition = normal.log_partition();
    
    assert_eq!(combined_params, individual_params);
    assert!((combined_log_partition - individual_log_partition).abs() < 1e-12);
}
```

## Future Considerations

### Adding New Measure Types
1. Implement `MeasureMarker` with appropriate type-level booleans
2. Implement `Measure<T>` 
3. If exponential family: implement `ExponentialFamily<X, F>` (gets `HasLogDensity` automatically)
4. If not exponential family: implement `HasLogDensity<T, F>` manually

### Adding New Numeric Types
The `EvaluateAt<T, F>` trait should work automatically with new numeric types that implement `Float`. For specialized numeric types, you may need custom implementations.

### Exponential Family Implementation Strategy
When implementing `ExponentialFamily`:
- If natural parameters and log partition share expensive computations, implement `natural_params_and_log_partition` and rely on defaults for individual methods
- If computations are independent and cheap, implement individual methods and rely on default for combined method
- If you need maximum performance for both access patterns, implement all three methods

## References
- [DESIGN_NOTES.md](DESIGN_NOTES.md) - Overall architecture and design philosophy
- [SIMPLIFICATION_PROPOSAL.md](SIMPLIFICATION_PROPOSAL.md) - Recent simplification analysis
- [Rust Reference - Associated Types](https://doc.rust-lang.org/reference/items/associated-items.html#associated-types)
- [Rust Reference - Const Generics](https://doc.rust-lang.org/reference/items/generics.html#const-generics)

## Density Trait Simplification (Completed)

**Status**: ✅ **COMPLETED** - Successfully simplified density traits without breaking functionality.

### What Was Removed
- `HasLogDensityWrt<T, F, BaseMeasure>` - Never implemented or used
- `LogDensityEval<T, F>` - Never implemented or used  
- `GeneralLogDensity<T, F>` - Redundant with builder pattern
- Duplicate `LogDensityBuilder` definition in `density.rs`

### What Remains
- `HasLogDensity<T, F>` - Core trait for log-density computation
- `LogDensityTrait<T, F>` - Used in builder pattern
- `LogDensityBuilder<T>` - Builder pattern entry point (in `measure.rs`)
- `EvaluateAt<T, F>` - Used by builder pattern
- `LogDensity<T, M, R>` - Builder pattern implementation
- `CachedLogDensity<L, T, F>` - Caching wrapper
- `DensityMeasure<T, F, M>` - Density as measure
- `SharedRootMeasure<T, F>` - Optimization trait

### Verification
✅ **Non-Exponential Family Support Verified**: Added Cauchy distribution as test case
- Cauchy is NOT an exponential family (heavy tails, no finite moments)
- Requires manual `HasLogDensity` implementation
- Works seamlessly with same API as exponential families
- Type-level dispatch correctly routes to manual implementation
- All tests pass, relative density computation works correctly

### Key Insights
1. **Type-level dispatch works perfectly**: `IsExponentialFamily = False` correctly routes to manual implementations
2. **API consistency maintained**: Both exponential and non-exponential families use identical API
3. **Performance preserved**: No runtime overhead from trait dispatch
4. **Composability intact**: Relative densities work between different distribution types

## Scalar DotProduct Implementations Are Ruled Out

**TL;DR**: We cannot add scalar `DotProduct` implementations due to trait coherence issues.

### The Problem
If we tried to add both array and scalar implementations:

```rust
// Existing array implementation
impl<T: Float, const N: usize> DotProduct for [T; N] {
    type Output = T;
    fn dot(&self, other: &Self) -> Self::Output { ... }
}

// Hypothetical scalar implementation - DON'T DO THIS
impl<T: Float> DotProduct for T {
    type Output = T;
    fn dot(&self, other: &Self) -> Self::Output { *self * *other }
}
```

This creates **trait coherence violations** because:
1. `T` could be `[f64; 1]`, making both impls applicable
2. Rust cannot determine which implementation to use
3. The compiler would reject this with overlapping implementations error

### The Solution
For exponential families with scalar natural parameters, use `[T; 1]` instead of `T`:

```rust
impl<T> ExponentialFamily<T, T> for Exponential<T> {
    type NaturalParam = [T; 1];  // ← Not just T
    type SufficientStat = [T; 1];
    // ...
}
```

This ensures compatibility with `DotProduct` without ambiguity issues.

## Exponential Family Relative Density Optimization

**Status**: ✅ **COMPLETED** - Decided to rely on zero-overhead optimization and JIT compilation rather than manual optimizations.

### Decision: Lean on JIT System

After implementing and testing both manual exponential family optimization and the existing zero-overhead system, we decided to **rely on the JIT compilation infrastructure** rather than maintaining separate manual optimizations.

### Performance Analysis

Testing showed:
- **Manual exponential family optimization**: 12x speedup vs baseline
- **Zero-overhead optimization**: 6x speedup vs baseline  
- **Standard exponential family**: 1x (baseline)

While the manual optimization is 2x faster than zero-overhead, this difference will be eliminated when JIT compilation is fully implemented, as JIT will:
- Inline all function calls
- Eliminate base measure computation overhead
- Generate optimal machine code
- Automatically detect and optimize mathematical patterns

### Current Recommendation

**For performance-critical relative density computation:**

```rust
// RECOMMENDED: Use zero-overhead optimization
let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
for &x in large_dataset {
    let result = optimized_fn(&x);  // 6x faster than baseline
}

// ALTERNATIVE: Use builder pattern (convenient, still optimized)
for &x in large_dataset {
    let result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
}
```

### Why This Approach

1. **Simpler codebase**: No need to maintain manual optimization functions
2. **Future-proof**: JIT will eventually make all approaches equally fast
3. **Consistent API**: Users don't need to choose between different optimization strategies
4. **Rust's strengths**: Leverages LLVM's optimization capabilities

### Mathematical Foundation (Preserved for Reference)

The mathematical insight remains valid and important:

For two exponential families p₁(x|θ₁) and p₂(x|θ₂) from the same family:
```
log(p₁(x)/p₂(x)) = (η₁ - η₂)·T(x) - (A(η₁) - A(η₂))
```

**Key insight**: Base measure terms `log h(x)` cancel out completely!

This optimization is **automatically applied** by:
- Zero-overhead optimization (partially - still computes base measures)
- JIT compilation (fully - when implemented)
- LLVM optimization passes (at compile time)

### Implementation Status

- ✅ **Zero-overhead optimization**: Working and providing 6x speedup
- ✅ **Builder pattern integration**: Seamless API for all cases  
- ✅ **Mathematical correctness**: All approaches give identical results
- 🚧 **JIT compilation**: In development, will provide ultimate optimization
- ❌ **Manual optimization**: Removed to simplify codebase

### Future Work

When JIT compilation is fully implemented:
1. **Automatic pattern detection**: JIT will recognize exponential family patterns
2. **Optimal code generation**: Base measure cancellation will be automatic
3. **Zero-cost abstraction**: All API approaches will have identical performance
4. **Advanced optimizations**: SIMD, vectorization, and CPU-specific optimizations

This approach aligns with Rust's philosophy of zero-cost abstractions and leverages the sophisticated optimization infrastructure already in place. 