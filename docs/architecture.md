# Architecture Guide

## Design Philosophy

The `measures` library is built on a fundamental separation between **measures** and **densities**. This design enables computing densities with respect to arbitrary base measures, not just canonical ones.

### Mathematical Foundation

In measure theory, a density function describes how one measure relates to another:

```
dμ/dν(x) = density of measure μ with respect to measure ν at point x
```

Traditional probability libraries only compute densities with respect to the canonical base measure (Lebesgue for continuous, counting for discrete). This library generalizes to arbitrary base measures.

## Core Architecture

### Type System Design

The library uses a three-layer trait design:

```rust
// 1. Mathematical structure (minimal interface)
trait LogDensityTrait<T> {
    type Measure;
    type BaseMeasure;
    fn measure(&self) -> &Self::Measure;
    fn base_measure(&self) -> &Self::BaseMeasure;
}

// 2. Generic evaluation (supports any numeric type)
trait EvaluateAt<T, F> {
    fn at(&self, x: &T) -> F;
}

// 3. Builder pattern (fluent interface)
struct LogDensity<T, M, B> {
    // Implementation details...
}
```

### Why This Design?

**Separation of Concerns**: Mathematical structure is separate from evaluation, enabling:
- Different numeric types (f64, f32, dual numbers for autodiff)
- Compile-time optimization based on measure types
- Static dispatch with zero-cost abstractions

**Type-Level Tracking**: The type system tracks measure relationships:
```rust
LogDensity<f64, Normal<f64>, LebesgueMeasure<f64>>   // Standard density
LogDensity<f64, Normal<f64>, Normal<f64>>            // Relative density
```

**Automatic Optimization**: When measures share the same root measure, the library automatically uses the mathematical identity:
```
log(dμ₁/dμ₂) = log(dμ₁/dν) - log(dμ₂/dν)
```

## Module Organization

```
src/
├── core/                    # Mathematical foundations
│   ├── density.rs          # LogDensity traits and types
│   ├── measure.rs          # Measure trait and primitives
│   └── types.rs            # Type-level programming helpers
├── measures/               # Concrete measure implementations
│   ├── primitive/          # Lebesgue, counting measures
│   └── derived/            # Dirac, weighted measures
├── distributions/          # Probability distributions
│   ├── continuous/         # Normal, Gamma, etc.
│   └── discrete/           # Poisson, Binomial, etc.
├── exponential_family/     # Exponential family specializations
├── symbolic-math/          # Expression system and JIT (separate crate)
└── bayesian/              # Bayesian modeling tools
```

### Design Principles

1. **Core before derived**: Fundamental abstractions in `core/`, concrete implementations build on them
2. **Primitive before derived**: Basic measures (Lebesgue, counting) before constructed ones (Dirac, weighted)
3. **Mathematical before computational**: Measure theory foundations before optimization techniques

## Key Abstractions

### Measures

```rust
trait Measure<T> {
    type RootMeasure: Measure<T>;
    fn root_measure(&self) -> Self::RootMeasure;
}

trait PrimitiveMeasure<T>: Measure<T> {
    // Primitive measures are their own root
}
```

**Examples**:
- `LebesgueMeasure`: Primitive measure for continuous distributions
- `CountingMeasure`: Primitive measure for discrete distributions
- `Normal`: Derived measure with `LebesgueMeasure` as root

### Log-Densities

```rust
trait HasLogDensity<T> {
    type LogDensity: LogDensityTrait<T>;
    fn log_density(&self) -> Self::LogDensity;
}
```

**Builder Pattern**:
```rust
measure.log_density()           // LogDensity<T, M, M::RootMeasure>
      .wrt(other_measure)       // LogDensity<T, M, OtherMeasure>
      .at(&x)                   // F (inferred from context)
```

### Exponential Families

```rust
trait ExponentialFamily<T>: HasLogDensity<T> {
    type NaturalParameters;
    type SufficientStatistic;
    
    fn to_natural(&self) -> Self::NaturalParameters;
    fn sufficient_statistic(&self, x: &T) -> Self::SufficientStatistic;
}
```

**IID Extension**:
```rust
trait IIDExtension<T>: ExponentialFamily<T> {
    fn iid(&self) -> IIDExponentialFamily<Self>;
}
```

## Automatic Differentiation Support

The split design enables automatic differentiation by separating evaluation from mathematical structure:

```rust
let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Same log-density, different numeric types
let f64_result: f64 = ld.at(&x);           // Standard evaluation
let dual_result: Dual64 = ld.at(&dual_x);  // Forward-mode autodiff
```

This works because `EvaluateAt<T, F>` is generic over the output type `F`.

## Performance Considerations

### Static Dispatch

The type system enables complete static dispatch:
```rust
// These become different monomorphized functions
normal.log_density().wrt(lebesgue).at(&x)    // Optimized for Lebesgue
normal.log_density().wrt(counting).at(&x)    // Optimized for counting
```

### Zero-Cost Abstractions

- No heap allocation in hot paths
- Compile-time measure compatibility checking
- Automatic optimization for shared root measures

### Optimization Strategies

1. **Standard evaluation**: Direct trait method calls
2. **Zero-overhead optimization**: Pre-compute constants, generate closures
3. **JIT compilation**: Runtime code generation (experimental)

## Extension Points

### Custom Measures

Implement the `Measure` trait:
```rust
struct CustomMeasure;

impl Measure<f64> for CustomMeasure {
    type RootMeasure = LebesgueMeasure<f64>;
    fn root_measure(&self) -> Self::RootMeasure {
        LebesgueMeasure::new()
    }
}
```

### Custom Distributions

Implement `HasLogDensity`:
```rust
impl HasLogDensity<f64> for CustomDistribution {
    type LogDensity = CustomLogDensity;
    fn log_density(&self) -> Self::LogDensity {
        CustomLogDensity::new(self)
    }
}
```

### Custom Numeric Types

Implement `EvaluateAt<T, F>` for your log-density type:
```rust
impl EvaluateAt<f64, MyNumericType> for MyLogDensity {
    fn at(&self, x: &f64) -> MyNumericType {
        // Custom evaluation logic
    }
}
```

## Mathematical Correctness

The library maintains mathematical correctness through:

1. **Type safety**: Incompatible operations are compile-time errors
2. **Measure compatibility**: Automatic checking of measure relationships
3. **Numerical stability**: Log-space computation throughout
4. **Identity preservation**: Automatic use of mathematical identities for optimization

This architecture provides a solid foundation for statistical computing while maintaining flexibility for advanced use cases. 