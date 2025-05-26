# Measures

A Rust library for statistical computing built on measure theory foundations. This library provides a mathematically rigorous approach to probability distributions and density computation through the separation of measures and densities.

## Technical Motivation

Traditional probability libraries conflate probability distributions with their canonical representations, limiting flexibility in density computation. This library addresses this limitation by:

1. **Separating measures from densities**: Enables computing densities with respect to arbitrary base measures
2. **Measure-theoretic foundations**: Provides mathematically sound abstractions for probability computation
3. **Exponential family unification**: Offers a consistent interface across exponential family distributions
4. **Type-safe optimization**: Leverages Rust's type system for compile-time guarantees and performance

## Core Architecture

### Mathematical Foundation

The library implements the mathematical relationship:
```
dμ₁/dμ₂ = (dμ₁/dν) / (dμ₂/dν)
```
where μ₁ and μ₂ are measures sharing a common root measure ν.

### Component Structure

- **Measures**: Mathematical objects that assign "size" to measurable sets
- **Densities**: Radon-Nikodym derivatives describing how one measure relates to another  
- **Distributions**: Concrete probability measures (Normal, Poisson, etc.)
- **Exponential Families**: Unified parameterization using natural parameters and sufficient statistics

### Type System Design

The library uses a split trait design for flexibility and optimization:

```rust
// Core traits
trait LogDensityTrait<T>     // Mathematical structure (minimal interface)
trait EvaluateAt<T, F>       // Generic evaluation with any numeric type F
struct LogDensity<T, M, B>   // Builder with fluent interface
```

This design enables the same mathematical object to work with different numeric types:

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Same mathematical object, different numeric types
let f64_result: f64 = ld.at(&x);           // Standard double precision
let f32_result: f32 = ld.at(&(x as f32));  // Single precision
// let dual_result: Dual64 = ld.at(&dual_x); // Automatic differentiation (with AD feature)
```

## Key Features

### General Density Computation

Compute densities with respect to arbitrary base measures, not just canonical ones:

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Standard: density w.r.t. Lebesgue measure
let density = normal1.log_density().at(&0.5);

// General: density of normal1 w.r.t. normal2 as base measure
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

The library automatically handles the mathematical requirements:
- Type safety prevents invalid measure combinations (e.g., continuous w.r.t. discrete)
- Efficient computation using the identity above when measures share root measures
- Proper handling of measure-theoretic edge cases

### Exponential Family Framework

Unified interface for exponential family distributions using natural parameterization:

```rust
use measures::{Normal, Poisson, exponential_family::ExponentialFamily};

let normal = Normal::new(2.0, 1.5);
let poisson = Poisson::new(3.0);

// Access natural parameters θ
let normal_params = normal.to_natural();
let poisson_params = poisson.to_natural();

// Compute sufficient statistics T(x)
let normal_stats = normal.sufficient_statistic(&x);
let poisson_stats = poisson.sufficient_statistic(&k);

// Exponential family form: exp(θᵀT(x) - A(θ))
```

### Independent and Identically Distributed (IID) Collections

Efficient computation for collections of independent samples:

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

let samples = vec![0.5, -0.3, 1.2];
let joint_log_density = iid_normal.iid_log_density(&samples);
```

### Symbolic Computation

Mathematical expression building and manipulation:

```rust
use symbolic_math::{Expr, final_tagless::{DirectEval, PrettyPrint, MathExpr}};

// Traditional AST approach
let expr = Expr::add(
    Expr::mul(Expr::constant(2.0), Expr::variable("x")),
    Expr::constant(1.0)
);

// Final tagless approach (polymorphic over interpreters)
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
where E::Repr<f64>: Clone,
{
    let a = E::constant(2.0);
    let b = E::constant(3.0);
    let c = E::constant(1.0);
    E::add(E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)), c)
}

// Direct evaluation
let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));

// Pretty printing
let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
```

## Optimization Strategies

The library provides multiple approaches for performance optimization:

### Standard Evaluation
Direct computation using Rust's native operations. Suitable for general-purpose use.

### Zero-Overhead Optimization
Pre-computes constants and eliminates redundant calculations at compile time:

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    let optimized_fn = normal.zero_overhead_optimize();
    let result = optimized_fn(&x);
}
```

### JIT Compilation (Experimental)
Native code generation for repeated evaluation:

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    if let Ok(jit_fn) = normal.auto_jit() {
        let result = jit_fn.call(x);
    }
}
```

**Note**: JIT compilation is experimental. Current implementations use placeholder functions for transcendental operations and may not provide performance benefits over standard evaluation.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
measures = "0.1"

# For experimental JIT features
measures = { version = "0.1", features = ["jit"] }

# For automatic differentiation support
measures = { version = "0.1", features = ["autodiff"] }

# For final tagless symbolic computation
measures = { version = "0.1", features = ["final-tagless"] }
```

## Examples

The `examples/` directory contains comprehensive demonstrations:

### Core Functionality
- [`general_density_computation.rs`](examples/general_density_computation.rs): Computing densities with custom base measures
- [`measure_theory_showcase.rs`](examples/measure_theory_showcase.rs): Mathematical foundations in practice
- [`structured_log_density_showcase.rs`](examples/structured_log_density_showcase.rs): Advanced density computation patterns

### Performance and Optimization  
- [`benchmark_comparison.rs`](examples/benchmark_comparison.rs): Performance comparison across optimization strategies
- [`runtime_vs_compile_time_optimization.rs`](examples/runtime_vs_compile_time_optimization.rs): Optimization technique comparison
- [`iid_optimization_demo.rs`](examples/iid_optimization_demo.rs): IID collection performance

### Advanced Features
- [`exponential_family_ir_example.rs`](examples/exponential_family_ir_example.rs): Exponential family intermediate representation
- [`autodiff_example.rs`](examples/autodiff_example.rs): Automatic differentiation integration
- [`jit_compilation_demo.rs`](examples/jit_compilation_demo.rs): JIT compilation demonstration

## Documentation

### Guides
- [Architecture Guide](docs/architecture.md): Mathematical foundations and design principles
- [Performance Guide](docs/performance.md): Optimization strategies and benchmarking methodology
- [API Reference](docs/api.md): Comprehensive API documentation with usage examples

### Additional Resources
- [Examples README](examples/README.md): Overview of all example programs
- [Autodiff Integration](examples/README_autodiff.md): Automatic differentiation framework integration

## Current Status and Limitations

### Implemented Features
- ✅ Core measure theory abstractions
- ✅ Exponential family framework
- ✅ Standard probability distributions
- ✅ IID collection optimization
- ✅ Basic symbolic computation
- ✅ Zero-overhead optimization

### Experimental Features
- ⚠️ **JIT compilation**: Functional but uses placeholder implementations for transcendental functions
- ⚠️ **Bayesian module**: Basic expression building only, JIT compilation not implemented

### Known Limitations
- **Multivariate distributions**: Limited support, primarily univariate focus
- **Transcendental function accuracy**: JIT implementations use approximations
- **Memory usage**: Some optimization strategies increase memory footprint

## Development

```bash
# Run test suite
cargo test

# Performance benchmarks  
cargo bench

# Build documentation
cargo doc --open

# Run specific example
cargo run --example general_density_computation

# Run with features
cargo run --features="jit,autodiff" --example benchmark_comparison
```

## Contributing

This library follows mathematical rigor and type safety principles. When contributing:

1. Ensure mathematical correctness of implementations
2. Maintain type safety guarantees
3. Include comprehensive tests for new features
4. Document mathematical foundations in code comments

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option. 