# Measures: Type-Safe Measure Theory for Rust

A Rust library for measure theory and probability distributions with a focus on type safety, performance, and automatic differentiation support.

## 🚀 Key Features

- **Type-safe measure theory**: Clear separation between measures and densities
- **Generic numeric types**: Works with `f64`, `f32`, dual numbers for autodiff, etc.
- **Zero-cost abstractions**: Compile-time optimization and static dispatch
- **Automatic shared-root computation**: Efficient log-density calculations
- **Exponential family support**: Specialized implementations for common distributions
- **Fluent API**: Natural, discoverable interface

## 📖 Quick Start

```rust
use measures::{Normal, Measure};

let normal = Normal::new(0.0, 1.0);
let x = 0.5;

// Compute log-density with respect to root measure (Lebesgue)
let ld = normal.log_density();
let log_density_value: f64 = ld.at(&x);

// Compute log-density with respect to different base measure
let other_normal = Normal::new(1.0, 2.0);
let ld_wrt = normal.log_density().wrt(other_normal);
let relative_log_density: f64 = ld_wrt.at(&x);

// Same log-density, different numeric types (autodiff ready!)
let f32_x = x as f32;
let f32_result: f32 = normal.log_density().at(&f32_x);
// let dual_result: Dual64 = ld.at(&dual_x);  // With autodiff library
```

## 🏗️ Architecture

The library is organized around a clean separation of concerns:

### Core Abstractions (`src/core/`)
- **`Measure<T>`**: Fundamental measure trait
- **`LogDensityTrait<T>`**: Mathematical log-density interface  
- **`EvaluateAt<T, F>`**: Generic evaluation for any numeric type
- **`LogDensity<T, M, B>`**: Builder for fluent log-density computation

### Measures (`src/measures/`)
- **Primitive**: `LebesgueMeasure`, `CountingMeasure` (building blocks)
- **Derived**: `Dirac`, `WeightedMeasure` (constructed from primitives)

### Distributions (`src/distributions/`)
- **Continuous**: `Normal`, `StdNormal`
- **Discrete**: `Poisson`  
- **Multivariate**: `MultivariateNormal`

## 🎯 Design Highlights

### Automatic Shared-Root Optimization

When measures share the same root measure, log-densities are automatically computed using the efficient formula:

```rust
// Both normals have LebesgueMeasure as root
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Automatically computed as: normal1.log_density() - normal2.log_density()
let ld = normal1.log_density().wrt(normal2);
```

### Generic Numeric Types

The same log-density works with different numeric types:

```rust
let normal = Normal::new(0.0, 1.0);
let x = 0.5;

let f64_result: f64 = normal.log_density().at(&x);           // Regular computation
let f32_result: f32 = normal.log_density().at(&(x as f32));  // Lower precision  
// let dual_result: Dual64 = normal.log_density().at(&dual_x); // Forward-mode autodiff
```

### Type-Level Safety

The type system enforces mathematical correctness:

```rust
// ✅ Valid: measures with same root
let ld = normal1.log_density().wrt(normal2);

// ❌ Compile error: incompatible measures  
// let ld = normal.log_density().wrt(discrete_measure);
```

## 📚 Documentation

- **[Design Notes](DESIGN_NOTES.md)**: Detailed architectural decisions and rationale
- **API Documentation**: Run `cargo doc --open` for full API docs
- **Examples**: See `examples/` directory for usage patterns

## 🔬 Advanced Features

### Caching
```rust
let ld_cached = normal.log_density().cached();
for &xi in &[0.1, 0.2, 0.1, 0.3, 0.1] {  // 0.1 computed only once
    let _val: f64 = ld_cached.at(&xi);
}
```

### Algebraic Operations
```rust
let ld_neg = -ld;              // Negated log-density
// let ld_sum = ld1 + ld2;     // Chain rule (when valid)
```

### Exponential Families
```rust
// Specialized implementations for exponential family distributions
// with natural parameter computations and sufficient statistics
```

## 🛠️ Development

```bash
# Run tests
cargo test

# Check compilation
cargo check

# Build documentation
cargo doc --open

# Run benchmarks
cargo bench
```

## 📄 License

[Add your license here] 