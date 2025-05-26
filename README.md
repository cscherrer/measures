# Measures

A Rust library for statistical computing with measure theory foundations and exponential family distributions.

## Architecture

The library separates **measures** from **densities** to enable flexible probability computation. This design allows computing densities with respect to any base measure, not just the canonical one.

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Standard: density w.r.t. Lebesgue measure
let density = normal1.log_density().at(&0.5);

// General: density of normal1 w.r.t. normal2
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

### Core Components

- **Measures**: Mathematical objects that assign "size" to sets
- **Densities**: Functions that describe how one measure relates to another
- **Distributions**: Concrete probability measures (Normal, Poisson, etc.)
- **Exponential Families**: Unified interface for exponential family distributions

### Type System Design

The library uses a split trait design for type safety and optimization:

```rust
// LogDensityTrait<T>: Mathematical structure (minimal)
// EvaluateAt<T, F>: Generic evaluation with any numeric type F
// LogDensity<T, M, B>: Builder with fluent interface

let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Same mathematical object, different numeric types
let f64_result: f64 = ld.at(&x);           // Standard evaluation
let f32_result: f32 = ld.at(&(x as f32));  // Lower precision
// let dual_result: Dual64 = ld.at(&dual_x); // Automatic differentiation
```

## Key Features

### General Density Computation

Compute densities with respect to arbitrary base measures:

```rust
use measures::{Normal, Poisson, LogDensityBuilder};

// Continuous distributions
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(2.0, 0.5);
let continuous_ratio = normal1.log_density().wrt(normal2).at(&1.0);

// Mixed measure types (when mathematically valid)
let discrete = Poisson::new(3.0);
// Note: normal.wrt(discrete) would be a compile-time error
```

**Mathematical foundation**: When measures share the same root measure, the library automatically uses the identity `log(dμ₁/dμ₂) = log(dμ₁/dν) - log(dμ₂/dν)` for efficient computation.

### Exponential Family Framework

Unified interface for exponential family distributions:

```rust
use measures::{Normal, Poisson, exponential_family::ExponentialFamily};

let normal = Normal::new(2.0, 1.5);
let poisson = Poisson::new(3.0);

// Access natural parameters and sufficient statistics
let normal_params = normal.to_natural();
let poisson_params = poisson.to_natural();

// Compute sufficient statistics
let normal_stats = normal.sufficient_statistic(&x);
let poisson_stats = poisson.sufficient_statistic(&k);
```

### IID Collections

Efficient computation for independent samples:

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

let samples = vec![0.5, -0.3, 1.2];
let joint_log_density = iid_normal.iid_log_density(&samples);
```

## Performance

The library provides multiple optimization strategies:

| Method | Time per call | Use Case |
|--------|---------------|----------|
| Standard evaluation | ~154 ns | General purpose |
| Zero-overhead optimization | ~106 ns | Pre-computed constants |
| JIT compilation | Experimental | Placeholder implementations |

```rust
// Standard evaluation
let result = normal.log_density().at(&x);

// Zero-overhead optimization (pre-computes constants)
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    let optimized_fn = normal.zero_overhead_optimize();
    let result = optimized_fn(&x);
}

// Experimental JIT compilation
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    if let Ok(jit_fn) = normal.auto_jit() {
        let result = jit_fn.call(x);
    }
}
```

**Note**: JIT compilation is experimental and currently slower than standard evaluation due to placeholder implementations for mathematical functions.

## Symbolic Computation

Basic symbolic computation for mathematical expressions:

```rust
use symbolic_math::{Expr, jit::GeneralJITCompiler};

// Build expressions programmatically
let expr = Expr::add(
    Expr::mul(Expr::constant(2.0), Expr::variable("x")),
    Expr::constant(1.0)
);

// Basic simplification
let simplified = expr.simplify();
```

## Installation

```toml
[dependencies]
measures = "0.1"

# For experimental JIT features
measures = { version = "0.1", features = ["jit"] }
```

## Examples

See the `examples/` directory for complete examples:

- `general_density_computation.rs`: Computing densities with custom base measures
- `optimization_comparison.rs`: Performance comparison of optimization strategies
- `iid_exponential_family_theory.rs`: Mathematical foundations and usage

## Documentation

- [Architecture Guide](docs/architecture.md): Core design principles and mathematical foundations
- [Performance Guide](docs/performance.md): Optimization strategies and benchmarking
- [API Reference](docs/api.md): Complete API documentation with examples

## Current Limitations

- **JIT compilation**: Uses placeholder implementations for transcendental functions
- **Bayesian module**: Basic expression building only, JIT not implemented
- **Multivariate distributions**: Limited support

## Development

```bash
cargo test                    # Run test suite
cargo bench                   # Performance benchmarks
cargo doc --open             # Build documentation
cargo run --example <name>   # Run examples
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option. 