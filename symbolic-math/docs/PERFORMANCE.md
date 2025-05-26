# Performance Guide

This document provides comprehensive performance analysis and optimization strategies for the symbolic-math crate.

## Performance Overview

The symbolic-math crate offers multiple evaluation strategies, each optimized for different use cases:

1. **Interpreted Evaluation**: General-purpose, good for development
2. **JIT Compilation**: High-performance native code generation
3. **Advanced Optimization**: Deep mathematical simplification

## Detailed Performance Metrics

### Expression Creation Performance

| Operation Type | Time per Expression | Use Case |
|---------------|-------------------|----------|
| Simple expressions | 59-62 ns | Basic arithmetic, variables |
| Complex expressions | 135-175 ns | Nested operations, functions |
| Builder patterns | 285-448 ns | Convenient API, readability |

**Example:**
```rust
// Simple: ~60 ns
let simple = Expr::add(Expr::variable("x"), Expr::constant(1.0));

// Complex: ~170 ns  
let complex = Expr::exp(Expr::neg(Expr::pow(Expr::variable("x"), Expr::constant(2.0))));

// Builder: ~420 ns
let builder = builders::normal_log_pdf("x", 0.0, 1.0);
```

### Simplification Performance

#### Basic Simplification
- **Speed**: 0-7 μs for most expressions
- **Effectiveness**: 71-100% complexity reduction
- **Best for**: Development, interactive use, simple expressions

| Pattern | Original Ops | Simplified Ops | Reduction | Time |
|---------|-------------|---------------|-----------|------|
| `x + 0` | 1 | 0 | 100% | 0-3 μs |
| `x * 1` | 1 | 0 | 100% | 0 μs |
| `x * 0` | 1 | 0 | 100% | 0 μs |
| `x^1` | 1 | 0 | 100% | 0 μs |
| `ln(exp(x))` | 2 | 0 | 100% | 0-1 μs |
| Complex redundant | 7 | 2 | 71% | 1 μs |

#### Advanced Optimization (egglog)
- **Speed**: 8-232 ms (much slower but more thorough)
- **Effectiveness**: Can find deep mathematical identities
- **Best for**: Offline optimization, mathematical correctness

| Expression | Original | Basic | Advanced | Time |
|-----------|----------|-------|----------|------|
| Redundant | 7 ops | 2 ops | 1 op | 232 ms |
| Polynomial | 12 ops | 11 ops | 10 ops | 26 ms |
| `sin²(x) + cos²(x)` | 5 ops | 5 ops | 0 ops (→1) | 8.7 ms |

### JIT Compilation Performance

#### Compilation Overhead
- **Compiler creation**: 23-89 μs (one-time cost)
- **Expression compilation**: 356-1578 μs per expression
- **Code generation**: ~128 bytes, ~11 CLIF instructions

#### Break-even Analysis
The compilation overhead pays off after a certain number of evaluations:

| Expression Type | Compilation Time | Interpreted Time/Call | JIT Time/Call | Break-even Point |
|----------------|------------------|---------------------|---------------|------------------|
| Linear | 1152 μs | 143 ns | 3.9 ns | ~8 calls |
| Quadratic | 666 μs | 158 ns | 6.3 ns | ~4 calls |
| Exponential | 837 μs | 178 ns | 7.1 ns | ~5 calls |
| Trigonometric | 500 μs | 127 ns | 3.9 ns | ~4 calls |

### Execution Performance

#### Interpreted Execution
- **Simple expressions**: 109-164 ns/call
- **Complex expressions**: 319-962 ns/call
- **Scaling**: Linear with expression complexity

#### JIT Execution
- **Simple expressions**: 3.9-8.6 ns/call
- **Complex expressions**: 6.3-11.4 ns/call
- **Scaling**: Nearly constant time regardless of complexity

#### Speedup Analysis

| Expression Type | Interpreted (ns) | JIT (ns) | Speedup | Best Use Case |
|----------------|------------------|----------|---------|---------------|
| Linear | 143 | 3.9 | 36.7x | Loops, simple math |
| Quadratic | 158 | 6.3 | 25.1x | Scientific computing |
| Exponential | 178 | 7.1 | 25.0x | Statistics, ML |
| Logarithmic | 118 | 8.6 | 13.9x | Information theory |
| Trigonometric | 127 | 3.9 | 32.3x | Signal processing |
| Polynomial (deg 5) | 962 | 11.4 | 84.3x | Numerical analysis |

## Optimization Strategies

### 1. Choose the Right Evaluation Method

```rust
// For development and one-time evaluation
let result = expr.evaluate(&vars)?;

// For repeated evaluation (>100 calls)
#[cfg(feature = "jit")]
{
    let jit_fn = compiler.compile_expression(&expr, &vars, &[], &HashMap::new())?;
    for x in data {
        let result = jit_fn.call_single(x);
    }
}
```

### 2. Simplify Before JIT Compilation

```rust
// Always simplify first to reduce compilation time and improve performance
let simplified = expr.simplify();
let jit_fn = compiler.compile_expression(&simplified, &vars, &[], &HashMap::new())?;
```

### 3. Use Builders for Common Patterns

```rust
// Instead of manual construction
let manual = Expr::add(
    Expr::neg(Expr::mul(Expr::constant(0.5), Expr::ln(Expr::mul(Expr::constant(2.0), Expr::constant(std::f64::consts::PI))))),
    Expr::neg(Expr::div(Expr::pow(Expr::sub(Expr::variable("x"), Expr::constant(0.0)), Expr::constant(2.0)), Expr::mul(Expr::constant(2.0), Expr::pow(Expr::constant(1.0), Expr::constant(2.0)))))
);

// Use builders
let builder = builders::normal_log_pdf("x", 0.0, 1.0);
```

### 4. Batch Processing for Multiple Evaluations

```rust
// Inefficient: multiple JIT compilations
for expr in expressions {
    let jit_fn = compiler.compile_expression(&expr, &vars, &[], &HashMap::new())?;
    let result = jit_fn.call_single(x);
}

// Efficient: compile once, evaluate many times
let jit_fn = compiler.compile_expression(&expr, &vars, &[], &HashMap::new())?;
for x in data {
    let result = jit_fn.call_single(x);
}
```

## Performance Tuning Guidelines

### When to Use JIT Compilation

✅ **Good candidates:**
- Expressions evaluated >1000 times
- Performance-critical inner loops
- Complex expressions (high speedup ratios)
- Batch processing scenarios

❌ **Poor candidates:**
- One-time evaluations
- Development/debugging
- Very simple expressions (overhead not worth it)
- Memory-constrained environments

### When to Use Advanced Optimization

✅ **Good candidates:**
- Offline preprocessing
- Mathematical correctness required
- Complex symbolic expressions
- Finding mathematical identities

❌ **Poor candidates:**
- Real-time applications
- Simple expressions
- Interactive development
- When basic simplification is sufficient

### Memory Considerations

- **JIT functions**: Keep compiled code in memory (~128 bytes typical)
- **Expression trees**: Memory usage scales with complexity
- **Optimization**: egglog can use significant memory for complex expressions

## Benchmarking Your Use Case

Use the provided examples to benchmark your specific use case:

```bash
# Run comprehensive profiling
cargo run --example profiling_benchmark --features="jit,optimization"

# Run micro-benchmarks for specific operations
cargo run --example micro_benchmarks --features="jit,optimization"

# Test without optional features
cargo run --example simple_profile
```

## Performance Regression Testing

To ensure performance doesn't regress:

1. **Baseline measurements**: Run benchmarks on known expressions
2. **Automated testing**: Include performance tests in CI
3. **Profile regularly**: Use profiling tools to identify bottlenecks

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_jit_speedup_regression() {
        let expr = builders::polynomial("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Measure interpreted performance
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = expr.evaluate(&vars);
        }
        let interpreted_time = start.elapsed();
        
        // Measure JIT performance
        let jit_fn = compiler.compile_expression(&expr, &["x".to_string()], &[], &HashMap::new()).unwrap();
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = jit_fn.call_single(1.0);
        }
        let jit_time = start.elapsed();
        
        let speedup = interpreted_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
        assert!(speedup > 10.0, "JIT speedup regression: {:.2}x", speedup);
    }
}
```

## Future Performance Improvements

Areas for potential optimization:

1. **SIMD vectorization**: Batch evaluation of multiple points
2. **Better transcendental functions**: More accurate and faster implementations
3. **Constant folding**: More aggressive compile-time optimization
4. **Memory pooling**: Reduce allocation overhead
5. **Parallel evaluation**: Multi-threaded expression evaluation 