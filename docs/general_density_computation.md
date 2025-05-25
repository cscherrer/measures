# General Density Computation

This document describes the measures framework's capability to compute log-densities with respect to any base measure, not just the root measure. This is a powerful feature that enables advanced statistical computing applications.

## Mathematical Foundation

### Standard Density Computation

Traditionally, probability densities are computed with respect to a canonical base measure:
- Continuous distributions use Lebesgue measure
- Discrete distributions use counting measure

For a measure ν with root measure μ, we compute: `log(dν/dμ)(x)`

### General Density Computation

The measures framework extends this to compute densities with respect to **any** base measure. For measures ν₁ and ν₂ sharing a computational root μ, we can compute:

```
log(dν₁/dν₂)(x) = log(dν₁/dμ)(x) - log(dν₂/dμ)(x)
```

This leverages the mathematical relationship that log-densities form a vector space under addition.

## API Overview

### Core Interfaces

1. **`.wrt()` method**: Change the base measure for density computation
2. **Optimization support**: Zero-overhead and JIT compilation for custom base measures

### Basic Usage

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);  // Standard normal
let normal2 = Normal::new(1.0, 2.0);  // Different parameters
let x = 0.5;

// Standard density (wrt Lebesgue measure)
let density1: f64 = normal1.log_density().at(&x);

// Relative density (normal1 wrt normal2)
let relative_density = normal1.log_density().wrt(normal2).at(&x);
```

## Performance Optimization

### Zero-Overhead Optimization

The framework provides zero-overhead optimization for computing densities with respect to custom base measures:

```rust
use measures::exponential_family::jit::ZeroOverheadOptimizer;

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Create optimized function for repeated evaluations
let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());

// Use the optimized function
let result: f64 = optimized_fn(&x);
```

### Compile-Time Macro Optimization

For maximum performance when parameters are known at compile time:

```rust
use measures::optimized_exp_fam;

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Macro with custom base measure
let macro_fn = optimized_exp_fam!(normal1, wrt: normal2);
let result = macro_fn(&x);
```

## Practical Applications

### 1. Importance Sampling

Compute importance weights when using one distribution as a proposal for another:

```rust
fn importance_weight<D1, D2>(
    target: &D1,
    proposal: &D2, 
    x: &f64
) -> f64 
where
    D1: LogDensityBuilder<f64> + Clone,
    D2: Measure<f64> + Clone,
{
    target.log_density().wrt(proposal.clone()).at(x).exp()
}

let target = Normal::new(0.0, 1.0);
let proposal = Normal::new(0.5, 1.2);
let weight = importance_weight(&target, &proposal, &x);
```

### 2. Model Comparison

Compute Bayes factors and likelihood ratios:

```rust
fn log_bayes_factor<M1, M2>(
    model1: &M1,
    model2: &M2,
    data: &f64
) -> f64
where
    M1: LogDensityBuilder<f64> + Clone,
    M2: Measure<f64> + Clone,
{
    model1.log_density().wrt(model2.clone()).at(data)
}

let model1 = Normal::new(0.0, 1.0);
let model2 = Normal::new(1.0, 2.0);
let log_bf = log_bayes_factor(&model1, &model2, &x);
let bayes_factor = log_bf.exp();
```

### 3. Change of Measure

Compute Radon-Nikodym derivatives for measure transformations:

```rust
fn radon_nikodym_derivative<P, Q>(
    measure_p: &P,
    measure_q: &Q,
    x: &f64
) -> f64
where
    P: LogDensityBuilder<f64> + Clone,
    Q: Measure<f64> + Clone,
{
    measure_p.log_density().wrt(measure_q.clone()).at(x).exp()
}
```

### 4. Variational Inference

Compute KL divergence terms and ELBO components:

```rust
fn elbo_density_term<Q, P>(
    variational: &Q,
    prior: &P,
    x: &f64
) -> f64
where
    Q: LogDensityBuilder<f64> + Clone,
    P: Measure<f64> + Clone,
{
    // log q(x) - log p(x) term in ELBO
    -variational.log_density().wrt(prior.clone()).at(x)
}
```

## Advanced Features

### Automatic Shared Root Detection

When measures share the same root measure, the framework automatically uses the efficient subtraction formula:

```rust
// Both Normal distributions have LebesgueMeasure as root
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Automatically computed as: normal1.log_density() - normal2.log_density()
let relative = normal1.log_density().wrt(normal2).at(&x);
```

### Generic Numeric Types

The same density computation works with different numeric types:

```rust
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);
let ld = normal1.log_density().wrt(normal2);

let f64_result: f64 = ld.at(&x);
let f32_result: f32 = ld.at(&(x as f32));
// For autodiff: let dual_result: Dual64 = ld.at(&dual_x);
```

### Caching Support

For repeated evaluations with the same base measure:

```rust
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

let cached_ld = normal1.log_density().wrt(normal2).cached();

for &xi in &[0.1, 0.2, 0.1, 0.3, 0.1] {
    let _val: f64 = cached_ld.at(&xi);  // 0.1 computed only once
}
```

## Implementation Details

### Type Safety

The type system ensures that only compatible measures can be used together:

```rust
// This compiles - both are continuous distributions
let continuous_relative = normal1.log_density().wrt(normal2).at(&x);

// This would cause a compilation error if attempted:
// let mixed = normal.log_density().wrt(poisson).at(&x);  // Type mismatch
```

### Zero-Cost Abstractions

All density computations use static dispatch and compile-time optimization:

- No heap allocation in hot paths
- Function calls are inlined by LLVM
- Type information is erased at runtime
- Performance equivalent to hand-written code

### Mathematical Correctness

The implementation ensures mathematical correctness through:

- Automatic handling of the chain rule
- Proper treatment of measure relationships
- Numerical stability for log-space computation
- Consistent handling of edge cases

## Best Practices

### 1. Choose the Right Method

- **`.wrt()`**: For fluent API and type safety
- **Optimization methods**: For repeated evaluations

### 2. Performance Considerations

- Use optimization for >100 evaluations with the same measures
- Consider caching for repeated evaluations with different points
- JIT compilation for >88,000 evaluations

### 3. Numerical Stability

- Always work in log-space for probability computations
- Be aware of potential overflow/underflow in extreme cases
- Use appropriate numeric types for your precision requirements

## Examples

See `examples/general_density_computation.rs` for a complete demonstration of all capabilities, including:

- Basic relative density computation
- Performance optimization techniques
- Practical applications in statistical computing
- Verification of mathematical correctness

## Future Extensions

Planned enhancements include:

- Support for measures with different root measures
- Automatic detection of conjugate relationships
- Integration with automatic differentiation frameworks
- GPU acceleration for batch computations 

## Approaches to General Density Computation

The measures library provides several approaches for computing log-densities with respect to different base measures:

1. **Builder pattern with `.wrt()`**: Fluent API for changing the base measure
2. **Direct computation**: Manual computation using individual log-densities

### 1. Builder Pattern Approach

```rust
use measures::{LogDensityBuilder, Normal};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Compute log-density of normal1 with respect to normal2
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

This approach provides a clean, fluent API for computing densities with respect to any base measure. 