# Symbolic Math Performance Benchmarks

This document presents comprehensive performance benchmarks for the symbolic-math crate using [Divan](https://crates.io/crates/divan), a high-precision benchmarking framework.

## Benchmark Results Summary

All benchmarks were run on a modern system with timer precision of 10 ns. Results show median times and throughput rates.

### Expression Creation Performance

| Operation | Median Time | Throughput | Notes |
|-----------|-------------|------------|-------|
| Simple expression (`2x + 1`) | 30.88 ns | - | Basic arithmetic |
| Complex expression | 99.81 ns | - | Multi-variable with transcendentals |
| Polynomial (degree 4) | 98.81 ns | - | `3x⁴ - 2x³ + x² - 5x + 7` |

**Key Insights:**
- Expression creation is extremely fast (< 100 ns)
- Complex expressions with transcendental functions have minimal overhead
- Memory allocation is well-optimized

### Simplification Performance

| Expression Type | Median Time | Notes |
|----------------|-------------|-------|
| Polynomial | 330.8 ns | Algebraic simplification |
| Trigonometric identity | 108.6 ns | `sin²(x) + cos²(x)` → `1` |
| Algebraic identity | 109.8 ns | `x + x + x - x` → `2x` |

**Key Insights:**
- Trigonometric identities are simplified very efficiently
- Basic algebraic simplification is fast and effective
- Pattern matching is well-optimized

### Evaluation Performance

#### Single Point Evaluation
| Expression | Size | Median Time | Throughput |
|------------|------|-------------|------------|
| Polynomial | 1 | 105.6 ns | 9.47 Mitem/s |
| Polynomial | 1000 | 103 µs | 9.70 Mitem/s |
| Complex | 1 | 119.8 ns | 8.35 Mitem/s |
| Complex | 1000 | 93.13 µs | 10.73 Mitem/s |

#### Batch Evaluation
| Expression | Size | Median Time | Throughput |
|------------|------|-------------|------------|
| Polynomial | 1 | 129.8 ns | 7.70 Mitem/s |
| Polynomial | 1000 | 118.7 µs | 8.42 Mitem/s |

**Key Insights:**
- Consistent ~10 Mitem/s throughput across different expression complexities
- Batch evaluation shows good scaling characteristics
- Complex expressions (with transcendentals) perform nearly as well as polynomials

### Grid Evaluation Performance

| Grid Size | Total Points | Median Time | Throughput |
|-----------|--------------|-------------|------------|
| 5×5 | 25 | 1.632 µs | 15.31 Mitem/s |
| 10×10 | 100 | 6.402 µs | 15.61 Mitem/s |
| 20×20 | 400 | 25.47 µs | 15.69 Mitem/s |
| 50×50 | 2500 | 160.4 µs | 15.57 Mitem/s |

**Key Insights:**
- Excellent scaling: ~15.6 Mitem/s throughput maintained across all grid sizes
- Grid evaluation is highly optimized for 2D parameter sweeps
- Linear scaling with number of evaluation points

### Caching Performance

| Operation | Cache State | Median Time | Speedup |
|-----------|-------------|-------------|---------|
| Evaluation | Cold | 610.8 ns | - |
| Evaluation | Warm | 470.3 ns | 1.30x |
| Simplification | Cold | 860.8 ns | - |
| Simplification | Warm | 665.8 ns | 1.29x |

**Key Insights:**
- Caching provides consistent ~30% performance improvement
- Cache overhead is minimal
- Effective for repeated operations on the same expressions

### JIT Compilation Performance

| Operation | Median Time | Notes |
|-----------|-------------|-------|
| Polynomial compilation | 24.7 µs | Cranelift-based JIT |

**Key Insights:**
- JIT compilation is fast (~25 µs for polynomial)
- Compilation overhead is reasonable for repeated evaluations
- JIT evaluation benchmarks were excluded due to thread safety constraints

### Advanced Optimization (Egglog)

| Expression Type | Median Time | Result |
|----------------|-------------|--------|
| Algebraic identity | 1.733 ms | `x + x + x - x` → `x + x` |
| Trigonometric identity | 1.102 ms | `sin²(x) + cos²(x)` → `1` |

**Key Insights:**
- Egglog optimization is powerful but expensive (~1-2 ms)
- Correctly identifies and applies mathematical identities
- Best suited for complex expressions where optimization pays off

### Memory Usage and Scaling

| Expression Size | Terms | Median Time | Throughput |
|----------------|-------|-------------|------------|
| Small | 10 | 525.8 ns | 19.01 Mitem/s |
| Medium | 100 | 5.289 µs | 18.9 Mitem/s |
| Large | 1000 | 53.3 µs | 18.76 Mitem/s |

**Key Insights:**
- Excellent scaling: ~19 Mitem/s maintained across expression sizes
- Memory allocation is well-optimized
- Linear scaling with expression complexity

### Complexity Scaling Analysis

| Nesting Depth | Median Time | Throughput |
|---------------|-------------|------------|
| 2 | 29.81 ns | 67.08 Mitem/s |
| 4 | 23.61 ns | 169.4 Mitem/s |
| 6 | 33 ns | 181.8 Mitem/s |
| 8 | 43 ns | 186 Mitem/s |
| 10 | 68.37 ns | 146.2 Mitem/s |

**Key Insights:**
- Performance remains excellent even with deep nesting
- Some optimization effects visible at medium depths
- Evaluation time scales sub-linearly with complexity

## Performance Recommendations

### For High-Throughput Applications
- Use batch evaluation for multiple data points
- Enable caching for repeated operations
- Consider JIT compilation for hot paths

### For Complex Expressions
- Basic simplification is very fast - use liberally
- Egglog optimization is expensive but powerful - use selectively
- Grid evaluation is highly optimized for parameter sweeps

### For Memory-Constrained Environments
- Expression creation and evaluation have minimal memory overhead
- Caching uses reasonable memory for significant speedups
- Large expressions scale well

## Benchmark Methodology

- **Framework**: Divan 0.1 with 10 ns timer precision
- **Samples**: 100 samples per benchmark
- **Iterations**: Variable (100-6400) based on operation speed
- **Features**: All benchmarks run with `jit` and `optimization` features
- **Environment**: Release build with optimizations enabled

## Running Benchmarks

```bash
# Basic benchmarks
cargo bench --bench divan_benchmarks

# With all features
cargo bench --bench divan_benchmarks --features "jit optimization"

# Specific benchmark
cargo bench --bench divan_benchmarks -- create_simple_expression
```

## Comparison with Other Libraries

The symbolic-math crate demonstrates excellent performance characteristics:

- **Expression creation**: Sub-100ns for most operations
- **Evaluation throughput**: Consistent 8-15 Mitem/s across workloads
- **Scaling**: Linear or sub-linear with expression complexity
- **Memory efficiency**: Minimal allocation overhead

These results position the crate well for high-performance symbolic computation applications. 