# Measures: Type-Safe Measure Theory for Rust

A Rust library implementing measure theory foundations for probability distributions with type safety and automatic differentiation support.

## Mathematical Foundation

This library implements measure theory concepts for statistical computing:

- **Measure separation**: Clear distinction between measures μ and their densities dν/dμ
- **Log-density computation**: Numerically stable evaluation of log(dν/dμ)(x)  
- **Base measure transformations**: Change of measure via log(dν₁/dμ) - log(dν₂/dμ) = log(dν₁/dν₂)
- **Exponential families**: Distributions of the form f(x|θ) = h(x)exp(η(θ)·T(x) - A(η(θ)))
- **IID collections**: Joint distributions for independent samples maintaining exponential family structure

## Architecture

### Core Abstractions (`src/core/`)
- `Measure<T>`: Fundamental measure trait defining support and root measure
- `LogDensityTrait<T>`: Mathematical interface for log-density computation
- `EvaluateAt<T, F>`: Generic evaluation enabling different numeric types
- `LogDensity<T, M, B>`: Builder type for fluent computation with type-level tracking

### Measure Implementations (`src/measures/`)
- **Primitive**: `LebesgueMeasure<T>`, `CountingMeasure<T>` 
- **Derived**: `Dirac<T>`, `FactorialMeasure<F>`

### Distributions (`src/distributions/`)
- **Continuous**: `Normal<T>`, `StdNormal<T>`
- **Discrete**: `Poisson<F>`
- **Exponential Family**: Unified interface via `ExponentialFamily<X, F>` trait

### Type System Features
- **Static dispatch**: Zero-cost abstractions with compile-time optimization
- **Generic numeric types**: Works with f64, f32, dual numbers for autodiff
- **Type-level safety**: Incompatible measures cause compilation errors

## Usage

### Basic Log-Density Computation

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);
let x = 0.5;

// Log-density with respect to Lebesgue measure
let log_density: f64 = normal.log_density().at(&x);

// Log-density with respect to different base measure  
let other_normal = Normal::new(1.0, 2.0);
let relative_log_density: f64 = normal.log_density().wrt(other_normal).at(&x);
```

### General Density Computation

The framework supports computing densities with respect to **any** base measure, enabling advanced statistical applications:

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Compute density of normal1 with respect to normal2
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

**Applications:**
- **Importance Sampling**: Compute weights when using different proposal distributions
- **Model Comparison**: Calculate Bayes factors and likelihood ratios
- **Variational Inference**: Compute ELBO terms and KL divergences
- **Change of Measure**: Calculate Radon-Nikodym derivatives

### Automatic Shared-Root Optimization

When measures share the same root measure, computation automatically uses:
`log(dν₁/dν₂) = log(dν₁/dμ) - log(dν₂/dμ)`

```rust
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Automatically computed via subtraction (both use LebesgueMeasure)
let relative_density = normal1.log_density().wrt(normal2).at(&x);
```

### Generic Numeric Types

```rust
// Same mathematical object, different numeric evaluations
let f64_result: f64 = normal.log_density().at(&x);
let f32_result: f32 = normal.log_density().at(&(x as f32));
// Autodiff: let dual_result: Dual64 = normal.log_density().at(&dual_x);
```

### IID Collections

For independent and identically distributed samples:

```rust
use measures::IIDExtension;

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();
let samples = vec![0.5, -0.3, 1.2];

// Efficient computation using exponential family structure
let joint_log_density: f64 = iid_normal.iid_log_density(&samples);
```

### Exponential Family Interface

```rust
use measures::exponential_family::{compute_exp_fam_log_density, ExponentialFamily};

let poisson = Poisson::new(3.0);

// Direct exponential family computation
let log_density = compute_exp_fam_log_density(&poisson, &2u64);

// Access natural parameters and sufficient statistics
let natural_params = poisson.to_natural();
let sufficient_stats = poisson.sufficient_statistic(&2u64);
```

### Caching for Performance

```rust
// Cache exponential family parameters for repeated evaluations
let cached_normal = normal.log_density().cached();
for &xi in &[0.1, 0.2, 0.1, 0.3, 0.1] {
    let _val: f64 = cached_normal.at(&xi);  // 0.1 computed only once
}
```

## Performance Optimization

The measures framework includes sophisticated performance optimization techniques that can achieve **significant speedups** over standard implementations:

```rust
use measures::exponential_family::jit::ZeroOverheadOptimizer;

let normal = Normal::new(2.0, 1.5);

// Zero-overhead runtime optimization (44% faster)
let optimized_fn = normal.zero_overhead_optimize();
let result = optimized_fn(&x);

// Compile-time macro optimization (47% faster)  
let macro_fn = measures::optimized_normal!(2.0, 1.5);
let result = macro_fn(x);

// JIT compilation for >88k calls
let jit_fn = normal.compile_jit()?;
let result = jit_fn.call(x);
```

### Performance Results

| Method | Time per call | Speedup |
|--------|---------------|---------|
| Zero-overhead (generic) | 0.55 ns | 1.02x |
| Zero-overhead (specific) | 0.39 ns | **1.44x** |
| Compile-time macro | 0.38 ns | **1.47x** |
| Standard framework | 0.56 ns | 1.0x (baseline) |
| Box<dyn Fn> | 2.13 ns | 0.26x (380% slower) |

### Key Features

- **🚀 Zero-overhead runtime codegen**: Beat the compiler with `impl Fn` closures
- **⚡ Generic optimization**: Works for ANY exponential family distribution  
- **🎯 JIT compilation**: Native machine code generation with Cranelift
- **📊 Comprehensive analysis**: Detailed overhead and amortization studies
- **🛠️ Best practices**: Decision trees for choosing optimization strategies

**See [Performance Optimization Guide](docs/performance_optimization.md) for complete details.**

### Examples

- **[Normal Optimization Techniques](examples/normal_optimization_techniques.rs)**: Complete demonstration of all optimization strategies for Normal distributions
- **[JIT Compilation](examples/jit_compilation.rs)**: Just-in-time compilation with Cranelift
- **[Symbolic Optimization](examples/symbolic_log_density.rs)**: Symbolic expression optimization

## Future Work

### Planned Extensions
- **Additional distributions**: Binomial, Beta, Gamma families
- **Multivariate support**: Complete multivariate normal implementation  
- **SIMD optimization**: Vectorized operations for batch computations
- **GPU support**: CUDA/OpenCL backends for large-scale computation

### Research Directions
- **Variational inference**: Automatic differentiation for gradient-based optimization
- **Hamiltonian Monte Carlo**: Integration with NUTS and other samplers
- **Information geometry**: Fisher information and natural gradients
- **Conjugate priors**: Automatic Bayesian updates for exponential families

### API Evolution
- **Algebraic operations**: Addition, composition of log-densities via operator overloading
- **Symbolic computation**: Compile-time evaluation of simple expressions
- **Custom base measures**: Extension framework for domain-specific measures

## Documentation

- **[General Density Computation](docs/general_density_computation.md)**: Complete guide to computing densities with respect to any base measure
- **[Performance Optimization Guide](docs/performance_optimization.md)**: JIT compilation, zero-overhead optimization, and performance analysis
- **[Design Notes](DESIGN_NOTES.md)**: Architectural decisions and mathematical rationale
- **[Exponential Family Implementation](IID_EXPONENTIAL_FAMILY_IMPLEMENTATION.md)**: IID mathematics and implementation details
- **API Documentation**: Run `cargo doc --open` for complete API reference
- **Examples**: See `examples/` directory for usage patterns and performance demonstrations

## Development

```bash
cargo test    # Run test suite
cargo bench   # Performance benchmarks  
cargo doc --open  # Build and view documentation
``` 