# Log-Density Design Architecture

## Repository Organization

The repository has been reorganized for maximum clarity and intuitive navigation:

```
src/
├── lib.rs                           # 🎯 Public API with comprehensive examples
├── core/                            # 🏗️  Fundamental abstractions
│   ├── density.rs                   # LogDensity traits and implementations
│   ├── measure.rs                   # Measure trait and PrimitiveMeasure
│   ├── types.rs                     # Type-level programming helpers
│   └── mod.rs                       # Clean re-exports
├── measures/                        # 📏 Concrete measure implementations
│   ├── primitive/                   # Basic building blocks
│   │   ├── lebesgue.rs             # Lebesgue measure (continuous)
│   │   └── counting.rs             # Counting measure (discrete)
│   └── derived/                     # Constructed measures
│       ├── dirac.rs                # Point masses
│       └── weighted.rs             # Weighted measures
├── distributions/                   # 📊 Probability distributions
│   ├── continuous/                  # Normal, etc.
│   ├── discrete/                    # Poisson, etc.
│   └── multivariate/               # Multivariate Normal, etc.
├── exponential_family/              # 📈 Exponential family specializations
└── traits/                          # 🔧 Domain-specific computational traits
    ├── dot_product.rs              # Vector operations
    └── exponential_family.rs       # Exponential family helpers
```

### Key Organizational Principles

1. **`core/`** - Mathematical foundations that everything builds on
2. **`measures/primitive/`** - Fundamental measures (Lebesgue, Counting)  
3. **`measures/derived/`** - Measures built from primitives (Dirac, Weighted)
4. **`distributions/`** - User-facing probability distributions
5. **`traits/`** - Domain-specific computational traits (not core abstractions)

This organization eliminates the previous duplicate definitions and creates a clear hierarchy from mathematical foundations to user-facing APIs.

## Design Philosophy: Why Split the Trait?

You asked whether all functionality should be in `LogDensityTrait` or if there are advantages to splitting up `wrt` and other functions. After implementing both approaches and correcting initial confusion, the **split design is definitively better** for several key reasons:

**Important Note**: I initially implemented a contradictory design that mixed both approaches. This has been corrected to implement the true split design.

## Critical Insight: Separating Evaluation from Mathematical Structure

A crucial insight emerged during the design process: **the output type needs to be separated from the log-density trait itself**. This enables powerful capabilities like automatic differentiation:

```rust
let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Same log-density, different numeric types
let f64_result: f64 = ld.at(&x);           // Regular evaluation  
let f32_result: f32 = ld.at(&(x as f32));  // Lower precision
let dual_result: Dual64 = ld.at(&dual_x);  // Forward autodiff with dual numbers
```

This separation is essential for modern scientific computing where you want to evaluate the same mathematical object with different number systems.

## True Split Design vs Unified Design

### Split Design (Implemented & Recommended)
- **`LogDensityTrait<T>`**: Minimal trait with only `measure()`, `base_measure()` - **no `at()` method**
- **`EvaluateAt<T, F>`**: Generic evaluation trait that works with any numeric type `F`
- **`LogDensity<T, M, B>`**: Builder type with `wrt()`, `at()`, `cached()`, etc.


### Unified Design (Rejected)
- **`LogDensityTrait<T>`**: Contains everything including fixed output type
- Less type safety, harder to optimize, **impossible to use with different numeric types**

## Why the Split Design is Superior

### 1. **Zero-Cost Type-Level Tracking**

The split allows the type system to track exactly which measures are being used at compile time:

```rust
LogDensity<f64, Normal<f64>, LebesgueMeasure<f64>>   // vs
LogDensity<f64, Normal<f64>, CountingMeasure<f64>>   // Different types = different optimizations
```

With everything in one trait, this information would be erased, preventing compile-time optimizations.

### 2. **Static Dispatch & Monomorphization**

Each specific combination gets its own optimized implementation:

```rust
// These become completely different monomorphized functions:
normal.log_density().wrt(lebesgue).at(&x)    // One optimized path
normal.log_density().wrt(counting).at(&x)    // Different optimized path  
```

A single trait would force dynamic dispatch or less specific optimization.

### 3. **Generic Numeric Types for Autodiff**

The separated evaluation enables automatic differentiation:

```rust
// Forward-mode autodiff with dual numbers
use dual::Dual64;

let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

let x = Dual64::new(1.5, 1.0);  // value=1.5, derivative=1.0
let result: Dual64 = ld.at(&x); // result.derivative is ∂/∂x log_density(x)
```

This is impossible with a fixed output type in the trait.

### 4. **Ergonomic Fluent Interface**

The builder pattern creates a natural, discoverable API:

```rust
measure.log_density()           // Start: LogDensity<T, M, M::RootMeasure>  
      .wrt(other_measure)       // Transform: LogDensity<T, M, OtherMeasure>
      .at(&x)                   // Evaluate: F (inferred from context)
```

### 5. **Minimal Trait Implementation**

Implementing the trait is simpler - you only need to provide the core mathematical operations:

```rust
impl<T> LogDensityTrait<T> for MyLogDensity<T> {
    type Measure = MyMeasure;
    type BaseMeasure = MyBaseMeasure;
    
    fn measure(&self) -> &Self::Measure { &self.measure }
    fn base_measure(&self) -> &Self::BaseMeasure { &self.base_measure }
    // That's it! No need to implement at(), wrt(), caching, etc.
    // Evaluation is handled separately via EvaluateAt<T, F>
}
```

## Core Design Components

### 1. **`LogDensityTrait<T>`** - The Mathematical Interface (Minimal)

Defines what a log-density *is* mathematically:
- `measure()` - what measure this density represents
- `base_measure()` - with respect to what base measure  

**Notably missing**: `at()` - this is intentionally separated into `EvaluateAt<T, F>`.

### 2. **`EvaluateAt<T, F>`** - Generic Evaluation Interface

Allows evaluation with any numeric type `F`:
- `at(&T) -> F` - evaluate at a point, returning type `F`

This enables the same log-density to work with:
- `f64` for regular computation
- `f32` for lower precision  
- `Dual64` for forward-mode autodiff
- `Complex<f64>` for complex analysis
- Custom number types for specialized computation

### 3. **`LogDensity<T, M, B>`** - The Builder Type (Feature-Rich)

The concrete implementation and entry point:
- `T` - the space type (f64, Vector, etc.)
- `M` - the measure type (Normal, Poisson, etc.)  
- `B` - the base measure type (defaults to M::RootMeasure)
- `wrt()` - change base measure (returns new `LogDensity` with different `B`)
- `at()` - evaluate at a point (generic over return type)
- `cached()` - wrap in caching (via extension trait)
  
### 4. **Algebraic Combinators**

Type-level implementations of log-density algebra:
- `CachedLogDensity<L, T, F>` - memoization wrapper (includes numeric type)
- Additional operations (addition, composition, etc.) can be added later as needed

## Key Requirements Satisfied

✅ **`l.wrt(new_base_measure)`** - Via builder pattern (only on `LogDensity<T,M,B>`)  
✅ **`l.at(x)` repeatability** - Pure functions, deterministic results  
✅ **Caching support** - Via `CachedLogDensity` wrapper  
✅ **Static/stack allocation** - No heap allocation in hot paths  
✅ **Generic numeric types** - **NEW**: Works with ANY numeric type via `EvaluateAt<T, F>`  
✅ **Default base measure** - **NEW**: Base measure defaults to root measure when not specified
✅ **Log-density algebra** - Via type-level combinators with operator overloading  
✅ **Autodiff support** - **NEW**: Can evaluate with dual numbers for automatic differentiation
✅ **Automatic shared-root computation** - **NEW**: When measures share root, automatically uses subtraction formula

## Automatic Computation for Shared Root Measures

A powerful feature of the design is automatic computation when measures share the same root measure. The type system automatically detects when two measures have the same `RootMeasure` and provides a default implementation using the mathematical relationship:

**`log(dm1/dm2) = log(dm1/root) - log(dm2/root)`**

### Implementation

This is implemented via the `HasLogDensity<T, F>` trait and a constraint-based default implementation:

```rust
/// Default implementation when measure and base measure share the same root measure
impl<T, M1, M2, F> EvaluateAt<T, F> for LogDensity<T, M1, M2>
where
    T: Clone,
    M1: Measure<T> + HasLogDensity<T, F> + Clone,
    M2: Measure<T, RootMeasure = M1::RootMeasure> + HasLogDensity<T, F> + Clone,
    F: std::ops::Sub<Output = F>,
{
    fn at(&self, x: &T) -> F {
        self.measure.log_density_wrt_root(x) - self.base_measure.log_density_wrt_root(x)
    }
}
```

### Usage Example

```rust
// Assume both Normal distributions have LebesgueMeasure as root
let normal1 = Normal::new(0.0, 1.0);  // RootMeasure = LebesgueMeasure
let normal2 = Normal::new(1.0, 2.0);  // RootMeasure = LebesgueMeasure

// This automatically uses the subtraction formula
let ld = normal1.log_density().wrt(normal2);
let value: f64 = ld.at(&x);  
// Equivalent to: normal1.log_density().at(&x) - normal2.log_density().at(&x)
```

### Benefits

1. **Automatic optimization**: No need to manually implement each pair of measures
2. **Mathematical correctness**: Enforced at the type level via `RootMeasure` equality
3. **Generic over numeric types**: Works with f64, f32, dual numbers, etc.
4. **Zero runtime cost**: All checks happen at compile time

## Usage Examples

```rust
use measures::{Normal, LebesgueMeasure, Measure};
use dual::Dual64; // Example autodiff library

let normal = Normal::new(0.0, 1.0);
let x = 0.5;

// Same log-density, different numeric types
let ld = normal.log_density();
let f64_result: f64 = ld.at(&x);           // Regular evaluation
let f32_result: f32 = ld.at(&(x as f32));  // Lower precision
let dual_result: Dual64 = ld.at(&Dual64::new(x, 1.0)); // Autodiff

// With different base measure  
let lebesgue = LebesgueMeasure::new();
let ld2 = normal.log_density().wrt(lebesgue);  // wrt() is on the builder type
let value2: f64 = ld2.at(&x);

// Algebraic operations work with any numeric type
let ld_neg = -ld;              // Negated log-density

// Caching for specific numeric types
let ld_cached = normal.log_density().cached_for::<f64>(); // Cache f64 results
let ld_cached_dual = normal.log_density().cached_for::<Dual64>(); // Cache Dual64 results

for &xi in &[0.1, 0.2, 0.1, 0.3, 0.1] {  // 0.1 computed only once
    let _val: f64 = ld_cached.at(&xi);
    let _dual_val: Dual64 = ld_cached_dual.at(&Dual64::new(xi, 1.0));
}
```

## Benefits of This Architecture

1. **Performance**: Zero-cost abstractions with compile-time optimization
2. **Type Safety**: Measure compatibility checked at compile time
3. **Mathematical Correctness**: Type system enforces valid log-density operations
4. **Ergonomics**: Natural fluent interface that guides correct usage
5. **Extensibility**: Easy to add new measure types and evaluation strategies
6. **Flexibility**: Works with any numeric type, supports caching per type
7. **Simple Implementation**: Trait implementers only need core methods
8. **Autodiff Ready**: Can be used with dual numbers, complex numbers, etc.

## Conclusion

The split design provides significant advantages over a monolithic trait approach. By separating the mathematical structure (`LogDensityTrait`) from evaluation (`EvaluateAt<T, F>`), we enable powerful capabilities like automatic differentiation while maintaining type safety and performance. The type-level tracking of measure relationships enables optimizations and correctness guarantees that would be impossible with a unified trait approach.

**Key insight**: Separating evaluation from mathematical structure is essential for modern scientific computing where the same mathematical object needs to be evaluated with different number systems (real, complex, dual, etc.).

Additional algebraic operations (like addition with proper chain rule constraints) can be added incrementally as needed, without disrupting the core design.