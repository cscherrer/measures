# Architecture Guide

## Mathematical Foundation and Design Rationale

### The Measure-Density Separation Problem

Traditional probability libraries conflate probability distributions with their canonical density representations. This creates limitations when working with:

1. **Relative densities**: Computing how one distribution relates to another
2. **Change of measure**: Transforming between different probability spaces  
3. **Bayesian inference**: Working with prior and posterior measures
4. **Numerical integration**: Using different base measures for quadrature

### Measure-Theoretic Solution

This library implements the fundamental measure-theoretic relationship:

```
dμ/dν(x) = Radon-Nikodym derivative of measure μ with respect to measure ν
```

For measures sharing a common root measure ρ:
```
dμ₁/dμ₂ = (dμ₁/dρ) / (dμ₂/dρ)
```

This mathematical identity enables efficient computation of relative densities without explicitly constructing intermediate representations.

## Core Type System Architecture

### Three-Layer Trait Design

The library separates mathematical structure from computational concerns through a layered trait hierarchy:

```rust
// Layer 1: Mathematical Structure (minimal interface)
trait LogDensityTrait<T> {
    type Measure;
    type BaseMeasure;
    fn measure(&self) -> &Self::Measure;
    fn base_measure(&self) -> &Self::BaseMeasure;
}

// Layer 2: Generic Evaluation (supports any numeric type)
trait EvaluateAt<T, F> {
    fn at(&self, x: &T) -> F;
}

// Layer 3: Builder Interface (fluent API)
struct LogDensity<T, M, B> {
    measure: M,
    base_measure: B,
}
```

### Design Benefits

**Type-Level Measure Tracking**: The type system encodes measure relationships at compile time:

```rust
// Different types for different mathematical objects
LogDensity<f64, Normal<f64>, LebesgueMeasure<f64>>   // Standard normal density
LogDensity<f64, Normal<f64>, Normal<f64>>            // Relative density between normals
LogDensity<f64, Poisson<f64>, CountingMeasure<f64>>  // Discrete distribution
```

**Separation of Concerns**: Mathematical structure is independent of evaluation strategy:
- Same mathematical object works with different numeric types (f64, f32, dual numbers)
- Compile-time optimization based on measure type relationships
- Static dispatch eliminates runtime overhead

**Automatic Mathematical Optimization**: The type system enables automatic application of mathematical identities:
```rust
// When measures share root measure, automatically uses:
// log(dμ₁/dμ₂) = log(dμ₁/dρ) - log(dμ₂/dρ)
normal1.log_density().wrt(normal2).at(&x)
```

## Module Organization and Dependencies

```
measures/
├── measures-core/              # Mathematical foundations
│   ├── src/core/              # Core abstractions
│   │   ├── density.rs         # LogDensity traits and builders
│   │   ├── measure.rs         # Measure trait hierarchy
│   │   └── types.rs           # Type-level programming utilities
│   ├── src/primitive/         # Primitive measures
│   │   ├── lebesgue.rs        # Continuous base measure
│   │   └── counting.rs        # Discrete base measure
│   └── src/traits/            # Shared trait definitions
├── measures-combinators/       # Measure construction
│   ├── src/measures/primitive/ # Basic measure implementations
│   ├── src/measures/derived/   # Constructed measures (Dirac, weighted)
│   └── src/measures/combinators/ # Measure operations (product, transform)
├── measures-distributions/     # Probability distributions
│   ├── src/distributions/continuous/ # Normal, Gamma, Beta, etc.
│   ├── src/distributions/discrete/   # Poisson, Binomial, etc.
│   └── src/distributions/multivariate/ # Multivariate distributions
├── measures-exponential-family/ # Exponential family specializations
│   ├── src/exponential_family/ # Core exponential family traits
│   └── src/implementations/    # Distribution-specific implementations
├── measures-bayesian/          # Bayesian modeling tools
└── symbolic-math/              # Expression system and compilation
    ├── src/expr.rs            # AST-based expressions
    ├── src/final_tagless.rs   # Final tagless approach
    └── src/jit.rs             # JIT compilation
```

### Architectural Principles

1. **Mathematical Foundations First**: Core abstractions in `measures-core` establish the mathematical framework
2. **Primitive Before Derived**: Basic measures (Lebesgue, counting) before constructed ones (Dirac, weighted)
3. **Separation of Concerns**: Mathematical structure separate from computational optimization
4. **Incremental Complexity**: Simple cases work without advanced features, complex cases build naturally

## Key Abstractions

### Measure Hierarchy

```rust
// Root trait for all measures
trait Measure<T> {
    type RootMeasure: Measure<T>;
    fn root_measure(&self) -> Self::RootMeasure;
}

// Primitive measures are their own root
trait PrimitiveMeasure<T>: Measure<T> {
    // Default implementation: Self as root measure
}

// Derived measures reference their root
trait DerivedMeasure<T>: Measure<T> {
    // Must specify non-Self root measure
}
```

**Concrete Examples**:
```rust
// Primitive measures
impl Measure<f64> for LebesgueMeasure<f64> {
    type RootMeasure = Self;  // Self-rooted
}

impl Measure<u64> for CountingMeasure<u64> {
    type RootMeasure = Self;  // Self-rooted
}

// Derived measures
impl Measure<f64> for Normal<f64> {
    type RootMeasure = LebesgueMeasure<f64>;  // Rooted in Lebesgue
}

impl Measure<u64> for Poisson<f64> {
    type RootMeasure = CountingMeasure<u64>;  // Rooted in counting
}
```

### Log-Density Construction

```rust
trait HasLogDensity<T> {
    type LogDensity: LogDensityTrait<T>;
    fn log_density(&self) -> Self::LogDensity;
}
```

**Builder Pattern Flow**:
```rust
// Start with a measure
let normal = Normal::new(0.0, 1.0);

// Get log-density w.r.t. root measure
let ld_canonical = normal.log_density();  // LogDensity<f64, Normal, Lebesgue>

// Change base measure
let ld_relative = ld_canonical.wrt(other_normal);  // LogDensity<f64, Normal, Normal>

// Evaluate at point
let result: f64 = ld_relative.at(&x);
```

### Exponential Family Framework

```rust
trait ExponentialFamily<T>: HasLogDensity<T> {
    type NaturalParameters;
    type SufficientStatistic;
    
    // Convert to natural parameterization: θ
    fn to_natural(&self) -> Self::NaturalParameters;
    
    // Compute sufficient statistics: T(x)
    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStatistic;
    
    // Log-partition function: A(θ)
    fn log_partition(&self) -> f64;
}
```

**Mathematical Form**: Exponential family distributions have the form:
```
p(x|θ) = exp(θᵀT(x) - A(θ)) h(x)
```
where:
- θ: natural parameters
- T(x): sufficient statistics  
- A(θ): log-partition function
- h(x): base measure density

**IID Extension**:
```rust
trait IIDExtension<T>: ExponentialFamily<T> {
    fn iid(&self) -> IIDExponentialFamily<Self>;
}

// For n independent samples: x₁, ..., xₙ
// Sufficient statistic becomes: Σᵢ T(xᵢ)
// Log-partition becomes: n·A(θ)
```

## Automatic Differentiation Integration

The split trait design enables seamless automatic differentiation:

```rust
// Mathematical structure remains the same
let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Different numeric types for different purposes
let value: f64 = ld.at(&x);                    // Function value
let gradient: Dual64 = ld.at(&dual_x);         // Forward-mode AD
let hessian: HyperDual64 = ld.at(&hyperdual_x); // Second-order AD
```

This works because:
1. `LogDensityTrait<T>` defines mathematical structure independent of numeric type
2. `EvaluateAt<T, F>` is generic over output type `F`
3. AD types implement the same arithmetic operations as standard floats

## Performance Architecture

### Static Dispatch and Monomorphization

The type system enables complete static dispatch:

```rust
// These become different monomorphized functions at compile time
fn evaluate_standard(normal: &Normal<f64>, x: f64) -> f64 {
    normal.log_density().at(&x)  // Optimized for standard evaluation
}

fn evaluate_relative(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) -> f64 {
    normal1.log_density().wrt(normal2).at(&x)  // Optimized for relative density
}
```

### Zero-Cost Abstractions

- **No heap allocation** in density evaluation hot paths
- **Compile-time measure compatibility** checking prevents runtime errors
- **Automatic mathematical optimization** through type-level dispatch
- **Inlined trait method calls** eliminate function call overhead

### Optimization Strategy Hierarchy

1. **Standard Evaluation**: Direct trait method dispatch
   - Uses Rust's native floating-point operations
   - Suitable for general-purpose computation
   - Baseline performance reference

2. **Zero-Overhead Optimization**: Compile-time constant folding
   - Pre-computes distribution parameters
   - Generates specialized closures
   - Eliminates redundant calculations

3. **JIT Compilation**: Runtime code generation (experimental)
   - Compiles expressions to native machine code
   - Optimizes for repeated evaluation patterns
   - Currently limited by placeholder transcendental function implementations

## Extension Mechanisms

### Custom Measures

Implement the measure trait hierarchy:

```rust
struct CustomMeasure {
    parameter: f64,
}

impl Measure<f64> for CustomMeasure {
    type RootMeasure = LebesgueMeasure<f64>;
    
    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}

impl HasLogDensity<f64> for CustomMeasure {
    type LogDensity = CustomLogDensity;
    
    fn log_density(&self) -> Self::LogDensity {
        CustomLogDensity::new(self.parameter)
    }
}
```

### Custom Numeric Types

Extend evaluation to new numeric types:

```rust
// Custom numeric type (e.g., interval arithmetic)
struct Interval { lower: f64, upper: f64 }

impl EvaluateAt<f64, Interval> for NormalLogDensity {
    fn at(&self, x: &f64) -> Interval {
        // Interval arithmetic implementation
        let lower_bound = /* ... */;
        let upper_bound = /* ... */;
        Interval { lower: lower_bound, upper: upper_bound }
    }
}
```

### Custom Exponential Families

Implement the exponential family interface:

```rust
impl ExponentialFamily<f64> for CustomDistribution {
    type NaturalParameters = (f64, f64);  // Two-parameter family
    type SufficientStatistic = (f64, f64); // Two sufficient statistics
    
    fn to_natural(&self) -> Self::NaturalParameters {
        // Convert from standard to natural parameterization
    }
    
    fn sufficient_statistic(&self, x: &f64) -> Self::SufficientStatistic {
        // Compute T(x) = (T₁(x), T₂(x))
    }
    
    fn log_partition(&self) -> f64 {
        // Compute A(θ) = log ∫ exp(θᵀT(x)) h(x) dx
    }
}
```

## Mathematical Correctness Guarantees

The architecture maintains mathematical correctness through:

### Type Safety
- **Measure compatibility**: Incompatible measure operations are compile-time errors
- **Dimension consistency**: Type system prevents dimension mismatches
- **Parameter validity**: Invalid parameter combinations caught at construction

### Numerical Stability
- **Log-space computation**: All density calculations in log space to prevent underflow
- **Numerically stable algorithms**: Implementations use stable computational methods
- **Precision preservation**: Careful ordering of operations to minimize floating-point error

### Mathematical Identity Preservation
- **Automatic optimization**: Mathematical identities applied automatically when beneficial
- **Consistency checking**: Relative density computations verified against direct calculation
- **Invariant maintenance**: Mathematical properties preserved across transformations

This architecture provides a mathematically sound and computationally efficient foundation for statistical computing while maintaining the flexibility needed for advanced applications. 