# Measures

A Rust library for statistical computing with measure theory foundations, exponential family distributions, and experimental JIT compilation.

## Core Features

### General Density Computation
Compute densities with respect to any base measure, not just the root measure:

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Compute density of normal1 with respect to normal2
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

### Exponential Family Framework
Unified interface for exponential family distributions with automatic optimization:

```rust
use measures::{Normal, exponential_family::jit::ZeroOverheadOptimizer};

let normal = Normal::new(2.0, 1.5);

// Standard evaluation
let result1 = normal.log_density().at(&1.0);

// Optimized evaluation (pre-computes constants)
let optimized_fn = normal.zero_overhead_optimize();
let result2 = optimized_fn(&1.0);
```

### IID Collections
Efficient computation for independent and identically distributed samples:

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

let samples = vec![0.5, -0.3, 1.2];
let joint_log_density = iid_normal.iid_log_density(&samples);
```

## Performance Optimization

The library provides multiple optimization strategies with different trade-offs:

### Standard Evaluation
Traditional trait-based evaluation using Rust's type system:

```rust
let result = normal.log_density().at(&x);  // ~414 ps/call
```

### Zero-Overhead Optimization
Pre-compute constants and generate optimized closures:

```rust
// Pre-compute constants once
let optimized_fn = normal.zero_overhead_optimize();
let result = optimized_fn(&x);  // ~515 ps/call
```

**Note**: "Zero-overhead" refers to reduced function call overhead, not elimination of all computational overhead.

### Experimental JIT Compilation
Runtime compilation to native machine code (experimental, not production-ready):

```rust
#[cfg(feature = "jit")]
{
    let jit_function = normal.compile_custom_jit()?;
    let result = jit_function.call(x);  // Currently slower due to implementation limitations
}
```

**Current JIT Status**: 
- Infrastructure exists but uses placeholder implementations for mathematical functions
- Performance overhead compared to standard evaluation
- Suitable for experimentation only

## Performance Results

Based on actual benchmarks (not estimates):

| Method | Time per call | Performance vs Standard |
|--------|---------------|------------------------|
| Standard evaluation | 414.49 ps | 1.0x (baseline) |
| Zero-overhead optimization | 515.45 ps | 0.8x (slower) |
| JIT compilation | 1,309.4 ps | 0.32x (3x slower) |

**Note**: Performance varies by hardware and use case. Zero-overhead optimization may provide benefits for specific scenarios despite average overhead.

## Symbolic IR and Expression System

Basic symbolic computation for mathematical expressions:

```rust
use measures::symbolic_ir::expr::Expr;

// Build expressions programmatically
let expr = Expr::add(
    Expr::mul(Expr::constant(2.0), Expr::variable("x")),
    Expr::constant(1.0)
);

// Basic simplification
let simplified = expr.simplify(); // Constant folding, identity elimination
```

## Bayesian Modeling (Experimental)

Basic infrastructure for Bayesian inference:

```rust
use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};

// Build Bayesian model expressions
let likelihood = normal_likelihood("x", "mu", "sigma");
let prior = normal_prior("mu", 0.0, 1.0);
let posterior = posterior_log_density(likelihood, prior);
```

**Status**: Expression building works, but JIT compilation uses placeholder implementations.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
measures = "0.1"

# For experimental JIT features
measures = { version = "0.1", features = ["jit"] }
```

## Examples

- **[General Density Computation](examples/general_density_computation.rs)**: Computing densities with respect to custom base measures
- **[Optimization Comparison](examples/optimization_comparison.rs)**: Performance comparison of different optimization strategies
- **[IID Exponential Family](examples/iid_exponential_family_theory.rs)**: Mathematical foundations and practical usage

## Documentation

- **[General Density Computation](docs/general_density_computation.md)**: Complete guide to computing densities with respect to any base measure
- **[Performance Optimization Guide](docs/performance_optimization.md)**: Optimization techniques and performance analysis
- **[Design Notes](DESIGN_NOTES.md)**: Architectural decisions and mathematical rationale

## Current Limitations

### JIT Compilation
- Uses placeholder implementations for ln(), exp(), sin(), cos() functions
- Performance overhead compared to standard evaluation
- Experimental status, not suitable for production

### Zero-Overhead Optimization
- Still computes base measures in some cases
- May have overhead for simple operations
- Benefits depend on specific use patterns

### Bayesian Module
- Basic expression building only
- JIT compilation not implemented (uses todo!() placeholders)
- Suitable for experimentation and development

## Future Work

### Planned Improvements
- Complete libm integration for JIT compilation
- Performance optimization for zero-overhead techniques
- Extended distribution family support
- Multivariate distribution implementations

### Research Directions
- Variational inference automation
- Information geometry integration
- GPU acceleration exploration
- Probabilistic programming language integration

## Development

```bash
cargo test                    # Run test suite
cargo bench                   # Performance benchmarks  
cargo doc --open             # Build and view documentation
cargo run --example <name>   # Run specific examples
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 