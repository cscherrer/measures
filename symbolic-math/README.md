# Symbolic Math

A high-performance symbolic mathematics library with Just-In-Time (JIT) compilation for Rust.

## Overview

This crate provides a general-purpose symbolic representation system for mathematical expressions with JIT compilation capabilities. It's designed to be domain-agnostic and can be used for any mathematical computation, not just probability distributions.

## Features

- **Symbolic Expression Building**: Create and manipulate mathematical expressions programmatically
- **Expression Simplification**: Basic algebraic simplification with excellent performance
- **Advanced Optimization**: Deep optimization using egglog for mathematical correctness
- **JIT Compilation**: Convert expressions to native machine code using Cranelift
- **Multiple Evaluation Modes**: Interpreted, JIT-compiled, and zero-overhead evaluation
- **Builder Patterns**: Convenient builders for common mathematical expressions

## 🚀 Performance

The symbolic-math crate is designed for high-performance symbolic computation with multiple optimization strategies:

### Overhead Reduction Optimizations ⚡

**NEW**: Significant overhead reduction optimizations achieve **2-6x performance improvements**:

| Expression Type | Original | Optimized | **Speedup** | Method |
|----------------|----------|-----------|-------------|---------|
| **Linear (2x + 3)** | 31.8 ns | 5.9 ns | **5.4x** | Specialized evaluation |
| **Quadratic (x² + 2x + 1)** | 56.3 ns | 14.6 ns | **3.9x** | Pattern recognition |
| **Complex polynomial** | 434.8 ns | 65.8 ns | **6.6x** | HashMap elimination |
| **Batch processing** | 57.6 ns/item | 25.0 ns/item | **2.3x** | Optimized allocation |
| **Transcendental functions** | 49.7 ns | 27.2 ns | **1.8x** | Single-variable path |

**Key Features**:
- 🎯 **Specialized evaluation** for linear and polynomial patterns
- 🗂️ **HashMap elimination** for single-variable expressions  
- 🔄 **Smart evaluation** that automatically chooses the fastest method
- 📦 **Optimized batch processing** with allocation reuse
- ⚡ **Constant folding** during evaluation

### Core Performance Features

- **Expression Creation**: < 100 ns for all types (30.88 ns simple, 99.81 ns complex)
- **Evaluation Throughput**: Consistent 8-15 Mitem/s across complexity levels
- **Grid Evaluation**: Excellent scaling at 15.6 Mitem/s maintained across all grid sizes
- **Simplification**: Fast pattern matching (108.6 ns trigonometric, 330.8 ns polynomial)
- **Caching**: 30% speedup (470.3 ns warm vs 610.8 ns cold)

### JIT Compilation 🔥

- **Compilation Time**: 24.7 µs median
- **Execution Speedup**: 16-30x faster than interpreted (3.9-6.5 ns/call vs 109-118 ns/call)
- **Break-even Point**: ~230-500 evaluations
- **Memory Scaling**: 19 Mitem/s maintained across expression sizes

### Advanced Optimization 🧠

- **Egglog Integration**: 1.1-1.7 ms optimization time
- **Mathematical Identities**: Correctly simplifies sin²+cos²=1 to constant 1
- **Complexity Reduction**: 71-100% reduction in expression complexity
- **Sub-linear Scaling**: Performance maintained with nesting depth

## Execution Overhead Analysis

Understanding the performance characteristics is crucial for optimal usage. Our detailed analysis reveals:

### Framework vs Raw Computation

| Expression Type | Framework Time | Raw Rust | Overhead Factor |
|----------------|----------------|----------|-----------------|
| Simple polynomial (x² + 2x + 1) | 42.1 ns | 0.17 ns | **247x** |
| Complex polynomial (degree 4) | 106.8 ns | 0.35 ns | **305x** |
| Transcendental functions | 79.8 ns | 29.8 ns | **2.7x** |

**Key Insights:**
- **Framework overhead dominates** for simple algebraic operations (95-99% of execution time)
- **Transcendental functions** have the best overhead ratio because computation cost is higher
- **Overhead is consistent** and predictable across expression types

### When to Use Each Approach

#### Framework Overhead Acceptable (Recommended)
- **Transcendental functions**: Only 2.7x overhead
- **Complex mathematical modeling**: Flexibility outweighs cost
- **Batch processing**: 15-18 Mitem/s sustained throughput
- **Applications with >100ns latency tolerance**

#### Consider Alternatives
- **Simple arithmetic in tight loops**: 247-305x overhead may be excessive
- **Sub-nanosecond performance requirements**: Use raw Rust
- **High-frequency trading**: Framework overhead too high

### Optimization Strategies

| Strategy | When to Use | Performance Gain |
|----------|-------------|------------------|
| **Batch processing** | Multiple evaluations | 20-40% improvement |
| **JIT compilation** | >500 repeated evaluations | 13-84x speedup |
| **Caching** | Repeated expressions | 30% improvement |
| **Expression simplification** | Complex expressions | Reduces evaluation cost |

For complete overhead analysis, see [EXECUTION_OVERHEAD_ANALYSIS.md](docs/EXECUTION_OVERHEAD_ANALYSIS.md).

## Quick Start

### Basic Expression Building

```rust
use symbolic_math::Expr;

// Build expression: x² + 2x + 1
let x = Expr::variable("x");
let expr = Expr::add(
    Expr::add(
        Expr::pow(x.clone(), Expr::constant(2.0)),
        Expr::mul(Expr::constant(2.0), x)
    ),
    Expr::constant(1.0)
);

// Simplify
let simplified = expr.simplify();
println!("Simplified: {}", simplified);
```

### Using Macros for Natural Syntax

```rust
use symbolic_math::expr;

// More natural syntax
let x = expr!(x);
let quadratic = expr!(x^2 + 2*x + 1);
let trig = expr!(sin(2*x) + cos(x));
```

### JIT Compilation

```rust
#[cfg(feature = "jit")]
{
    use symbolic_math::GeneralJITCompiler;
    use std::collections::HashMap;

    let expr = expr!(x^2 + 2*x + 1);
    
    let compiler = GeneralJITCompiler::new()?;
    let jit_function = compiler.compile_expression(
        &expr,
        &["x".to_string()], // data variables
        &[],                // parameter variables
        &HashMap::new(),    // constants
    )?;

    // JIT execution is 13-84x faster than interpreted
    let result = jit_function.call_single(2.0);
}
```

### Advanced Optimization

```rust
#[cfg(feature = "optimization")]
{
    use symbolic_math::EgglogOptimize;

    // Create expression with trigonometric identity
    let expr = expr!(sin(x)^2 + cos(x)^2);
    
    // Advanced optimization correctly simplifies to 1
    let optimized = expr.optimize_with_egglog()?;
    println!("Optimized: {}", optimized); // Should be constant 1
}
```

### Builder Patterns

```rust
use symbolic_math::builders;

// Normal distribution log-PDF
let normal_logpdf = builders::normal_log_pdf("x", 0.0, 1.0);

// Polynomial
let poly = builders::polynomial("x", &[1.0, -2.0, 3.0, -1.0]);

// Gaussian kernel
let kernel = builders::gaussian_kernel("x", 0.0, 1.0);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
symbolic-math = "0.1"

# For JIT compilation
symbolic-math = { version = "0.1", features = ["jit"] }

# For advanced optimization
symbolic-math = { version = "0.1", features = ["optimization"] }

# For all features
symbolic-math = { version = "0.1", features = ["jit", "optimization"] }
```

## Performance Recommendations

### When to Use JIT Compilation
- **Repeated evaluations**: >500 calls per expression
- **Performance-critical paths**: Where every nanosecond counts
- **Complex expressions**: Higher complexity = better speedup ratios

### When to Use Basic Simplification
- **One-time evaluations**: Development and debugging
- **Interactive use**: Fast feedback loops
- **Simple expressions**: Already well-optimized

### When to Use Advanced Optimization
- **Mathematical correctness**: Finding deep mathematical identities
- **Complex symbolic manipulation**: Where basic simplification isn't enough
- **Offline optimization**: When compilation time isn't critical

## Examples

Run the profiling examples to see performance characteristics:

```bash
# Comprehensive profiling across all areas
cargo run --example profiling_benchmark --features="jit,optimization"

# Execution overhead analysis
cargo run --example overhead_demonstration --features="jit,optimization"

# Focused micro-benchmarks
cargo run --example micro_benchmarks --features="jit,optimization"

# Basic profiling without optional features
cargo run --example simple_profile
```

## Architecture

The crate is organized into several modules:

- **`expr`**: Core symbolic expression representation
- **`jit`**: JIT compilation using Cranelift (optional)
- **`optimization`**: Advanced optimization using egglog (optional)
- **`builders`**: Convenient expression builders
- **`macros`**: Natural syntax macros

## Performance Scaling

| Expression Type | Complexity | JIT Speedup | Best Use Case |
|----------------|------------|-------------|---------------|
| Linear         | Low        | 28-37x      | Simple math, loops |
| Polynomial     | Medium     | Up to 84x   | Scientific computing |
| Trigonometric  | Medium     | 25-68x      | Signal processing |
| Exponential    | High       | 25-52x      | Statistics, ML |

## Current Limitations

- **JIT compilation**: Uses Taylor series approximations for transcendental functions
- **Advanced optimization**: Can be slow for very complex expressions
- **Memory usage**: JIT functions keep compiled code in memory

## Contributing

We welcome contributions! Areas for improvement:

1. **Better transcendental function implementations** in JIT
2. **Vectorized operations** for batch processing
3. **More optimization rules** in egglog
4. **Additional builder patterns** for common expressions

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option. 