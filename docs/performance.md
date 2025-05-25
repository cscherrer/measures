# Performance Guide

## Optimization Strategies

The library provides three optimization approaches with different trade-offs:

| Method | Time per call | Compilation overhead | Use case |
|--------|---------------|---------------------|----------|
| Standard evaluation | ~154 ns | None | General purpose |
| Zero-overhead optimization | ~106 ns | Minimal | Pre-computed constants |
| JIT compilation | Experimental | High | Placeholder implementations |

**Note**: These numbers are from realistic benchmarks (100,000 evaluations of Normal distribution log-density). Sub-nanosecond claims in some micro-benchmarks are artifacts of compiler optimizations eliminating actual work.

## Standard Evaluation

Direct trait method calls using Rust's type system:

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);
let result = normal.log_density().at(&x);
```

**Characteristics**:
- No compilation overhead
- Consistent performance across different distributions
- Full mathematical function support
- Recommended for most use cases

## Zero-Overhead Optimization

Pre-computes constants and generates optimized closures:

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    
    let normal = Normal::new(2.0, 1.5);
    let optimized_fn = normal.zero_overhead_optimize();
    let result = optimized_fn(&x);
}
```

**Characteristics**:
- Minimal compilation overhead (~microseconds)
- Pre-computes distribution parameters
- Currently has slight overhead due to implementation details
- May benefit specific use patterns despite average overhead

**When to use**:
- Many evaluations with the same distribution parameters
- Performance-critical inner loops
- When compilation time is not a concern

## JIT Compilation

Runtime compilation to native machine code:

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    
    let normal = Normal::new(0.0, 1.0);
    if let Ok(jit_fn) = normal.auto_jit() {
        let result = jit_fn.call(x);
    }
}
```

**Current limitations**:
- Uses placeholder implementations for `ln()`, `exp()`, `sin()`, `cos()`
- Performance overhead compared to standard evaluation
- Experimental status, not production-ready
- May produce incorrect results for distributions requiring transcendental functions

**Future potential**:
- Could provide significant speedups once mathematical functions are properly implemented
- Useful for complex expressions with many operations
- Enables runtime specialization based on data patterns

## Amortization Analysis

The break-even point for JIT compilation depends on the number of evaluations:

```rust
// Benchmark different call counts to find break-even points
let call_counts = vec![1, 10, 100, 1000, 10000];

for count in call_counts {
    // Include compilation time in measurement
    let start = Instant::now();
    let jit_fn = normal.auto_jit()?;
    for _ in 0..count {
        jit_fn.call(x);
    }
    let total_time = start.elapsed();
}
```

**Current results** (from realistic benchmarks):
- Standard evaluation: ~154 ns/call (100,000 evaluations of Normal distribution)
- Zero-overhead optimization: ~106 ns/call (31% improvement)
- JIT compilation: Currently experimental with placeholder mathematical functions

**Benchmark methodology**: Results from `cargo run --example optimization_comparison` measuring 100,000 evaluations of Normal distribution log-density computation. This provides realistic performance estimates for actual workloads.

## Performance Best Practices

### Choose the Right Strategy

1. **Default to standard evaluation** for most use cases
2. **Consider zero-overhead optimization** for repeated evaluations with fixed parameters
3. **Avoid JIT compilation** until mathematical functions are properly implemented

### Measure Performance

Always benchmark your specific use case:

```rust
use criterion::{black_box, Criterion};

fn benchmark_approaches(c: &mut Criterion) {
    let normal = Normal::new(0.0, 1.0);
    let x = 1.5;
    
    c.bench_function("standard", |b| {
        let ld = normal.log_density();
        b.iter(|| black_box(ld.at(&black_box(x))));
    });
    
    #[cfg(feature = "jit")]
    c.bench_function("zero_overhead", |b| {
        let optimized = normal.zero_overhead_optimize();
        b.iter(|| black_box(optimized(&black_box(x))));
    });
}
```

### Avoid Common Pitfalls

1. **Don't recompile in hot loops**:
   ```rust
   // Bad: recompiles every iteration
   for x in values {
       let optimized = normal.zero_overhead_optimize();
       result = optimized(&x);
   }
   
   // Good: compile once, use many times
   let optimized = normal.zero_overhead_optimize();
   for x in values {
       result = optimized(&x);
   }
   ```

2. **Consider compilation overhead**:
   - Zero-overhead: ~microseconds
   - JIT compilation: ~milliseconds
   - Only worthwhile for many evaluations

3. **Profile your specific workload**:
   - Performance varies by distribution type
   - Hardware differences affect results
   - Compiler optimizations may change trade-offs

## Automatic Optimizations

The library automatically applies several optimizations:

### Shared Root Measure Optimization

When computing relative densities with shared root measures:

```rust
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Automatically uses: log(dμ₁/dμ₂) = log(dμ₁/dν) - log(dμ₂/dν)
let relative_density = normal1.log_density().wrt(normal2).at(&x);
```

This avoids redundant computation of the shared base measure.

### Static Dispatch

The type system enables complete static dispatch:

```rust
// These become different optimized functions at compile time
normal.log_density().wrt(lebesgue).at(&x);    // Lebesgue-specific
normal.log_density().wrt(counting).at(&x);    // Counting-specific
```

### Constant Folding

Distribution parameters are embedded as constants when possible:

```rust
// Parameters (2.0, 1.5) become compile-time constants
let normal = Normal::new(2.0, 1.5);
let optimized = normal.zero_overhead_optimize();
```

## Future Optimizations

### Planned Improvements

1. **Proper JIT mathematical functions**: Replace placeholders with correct implementations
2. **SIMD vectorization**: Batch evaluation of multiple points
3. **GPU acceleration**: Parallel evaluation for large datasets
4. **Advanced symbolic optimization**: Algebraic simplification before compilation

### Research Directions

1. **Adaptive optimization**: Choose strategy based on runtime patterns
2. **Profile-guided optimization**: Use execution profiles to guide compilation
3. **Specialized numeric types**: Custom number types for specific domains
4. **Distributed computation**: Scale across multiple cores/machines

The performance characteristics will evolve as these optimizations are implemented. Always benchmark your specific use case with the current version. 