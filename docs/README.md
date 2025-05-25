# Measures Documentation

Technical documentation for the `measures` crate - a Rust library for statistical computing with measure theory foundations.

## Documentation Structure

- **[Architecture Guide](architecture.md)** - Core design principles and mathematical foundations
- **[Performance Guide](performance.md)** - Optimization strategies and benchmarking
- **[API Reference](api.md)** - Complete API documentation with examples

## Quick Start

The `measures` crate separates measures from densities to enable flexible probability computation:

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);

// Standard density (w.r.t. Lebesgue measure)
let density = normal.log_density().at(&0.5);

// Relative density (w.r.t. another measure)
let other_normal = Normal::new(1.0, 2.0);
let relative_density = normal.log_density().wrt(other_normal).at(&0.5);
```

## Key Concepts

### Measure Theory Foundation

Traditional probability libraries compute densities only with respect to canonical base measures (Lebesgue for continuous, counting for discrete). This library generalizes to arbitrary base measures, enabling:

- **Importance sampling**: Compute densities w.r.t. proposal distributions
- **Model comparison**: Direct density ratios between models
- **Variational inference**: Densities w.r.t. variational approximations

### Type System Design

The library uses a split trait design for type safety and performance:

```rust
// Mathematical structure (minimal interface)
trait LogDensityTrait<T> { /* ... */ }

// Generic evaluation (any numeric type)
trait EvaluateAt<T, F> { /* ... */ }

// Builder pattern (fluent interface)
struct LogDensity<T, M, B> { /* ... */ }
```

This enables:
- Automatic differentiation support
- Compile-time optimization
- Zero-cost abstractions

### Exponential Family Framework

Unified interface for exponential family distributions:

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);

// Access natural parameters and sufficient statistics
let params = normal.to_natural();
let stats = normal.sufficient_statistic(&x);

// Efficient IID computation
let iid_normal = normal.iid();
let joint_density = iid_normal.iid_log_density(&samples);
```

## Performance

Multiple optimization strategies with different trade-offs:

| Method | Time per call | Use case |
|--------|---------------|----------|
| Standard evaluation | ~154 ns | General purpose |
| Zero-overhead optimization | ~106 ns | Pre-computed constants |
| JIT compilation | Experimental | Placeholder implementations |

See the [Performance Guide](performance.md) for detailed analysis and best practices.

## Examples

Complete examples are available in the repository's `examples/` directory:

- `general_density_computation.rs` - Computing densities with custom base measures
- `optimization_comparison.rs` - Performance comparison of optimization strategies  
- `iid_exponential_family_theory.rs` - Mathematical foundations and usage

## Mathematical Correctness

The library maintains correctness through:

1. **Type safety**: Incompatible operations are compile-time errors
2. **Measure compatibility**: Automatic checking of measure relationships  
3. **Numerical stability**: Log-space computation throughout
4. **Identity preservation**: Automatic use of mathematical identities

For detailed architectural information, see the [Architecture Guide](architecture.md). 