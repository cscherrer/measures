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

## Performance

The symbolic-math crate delivers excellent performance across all operations. Comprehensive benchmarks using [Divan](https://crates.io/crates/divan) show:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Expression creation | < 100 ns | All expression types |
| Evaluation throughput | 8-15 Mitem/s | Consistent across complexity |
| Grid evaluation | 15.6 Mitem/s | Excellent scaling |
| Basic simplification | ~100-300 ns | Fast pattern matching |
| Caching speedup | 1.3x | 30% improvement |
| JIT compilation | ~25 µs | Cranelift-based |
| Egglog optimization | ~1-2 ms | Powerful but expensive |

**Key Highlights:**
- Sub-linear scaling with expression complexity
- Minimal memory allocation overhead  
- Excellent performance for both simple and complex expressions
- Effective caching for repeated operations

For detailed benchmark results and methodology, see [BENCHMARKS.md](docs/BENCHMARKS.md).

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
- **Repeated evaluations**: >1000 calls per expression
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