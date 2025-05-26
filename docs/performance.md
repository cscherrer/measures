# Performance Guide

## Optimization Strategy Overview

The library provides multiple optimization approaches, each with specific use cases and trade-offs. Performance characteristics depend heavily on workload patterns, distribution types, and evaluation frequency.

### Available Optimization Methods

1. **Standard Evaluation**: Direct trait method dispatch using Rust's type system
2. **Zero-Overhead Optimization**: Compile-time constant folding and closure generation  
3. **JIT Compilation**: Runtime native code generation (experimental)

### Benchmark Methodology

Performance measurements in this guide come from:
- **Workload**: 100,000 evaluations of Normal distribution log-density computation
- **Hardware**: Results may vary significantly across different systems
- **Measurement**: Total time including all overhead, not isolated operation timing
- **Tool**: Criterion.rs benchmarking framework with proper statistical analysis

**Important**: Always benchmark your specific use case. Performance characteristics vary significantly based on distribution type, parameter values, and usage patterns.

## Standard Evaluation

Direct computation using Rust's native trait system and floating-point operations.

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);
let result = normal.log_density().at(&x);
```

### Characteristics

- **Compilation overhead**: None
- **Runtime overhead**: Minimal trait dispatch
- **Mathematical accuracy**: Full precision using standard library functions
- **Compatibility**: Works with all distributions and numeric types
- **Memory usage**: No additional allocations in evaluation path

### Performance Profile

- Consistent performance across different distribution types
- Predictable execution time for capacity planning
- No warm-up period or compilation delays
- Suitable for interactive applications and real-time systems

### When to Use

- **Default choice** for most applications
- Applications requiring predictable performance
- Interactive systems where compilation delays are unacceptable
- Mixed workloads with varying distribution types

## Zero-Overhead Optimization

Pre-computes distribution parameters and generates specialized evaluation closures.

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    
    let normal = Normal::new(2.0, 1.5);
    let optimized_fn = normal.zero_overhead_optimize();
    
    // Use optimized function for repeated evaluations
    for x in data_points {
        let result = optimized_fn(&x);
    }
}
```

### Optimization Techniques

- **Parameter pre-computation**: Distribution parameters calculated once at optimization time
- **Constant folding**: Mathematical constants embedded in generated code
- **Specialized closures**: Type-specific evaluation paths without dynamic dispatch
- **Redundancy elimination**: Common subexpressions computed once

### Performance Characteristics

- **Compilation time**: Microseconds for simple distributions
- **Memory overhead**: Additional closure storage
- **Evaluation speed**: Potential improvement for parameter-heavy distributions
- **Scalability**: Benefits increase with evaluation frequency

### Limitations

- **Current implementation overhead**: May not always provide speedup due to implementation details
- **Distribution-specific**: Benefits vary significantly by distribution type
- **Parameter fixation**: Optimized for specific parameter values only
- **Memory usage**: Additional storage for pre-computed values

### When to Use

- High-frequency evaluation with fixed distribution parameters
- Performance-critical inner loops where microsecond compilation time is acceptable
- Batch processing of large datasets with the same distribution
- Applications where memory usage for optimization is not a concern

## JIT Compilation (Experimental)

Runtime compilation of mathematical expressions to native machine code.

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    
    let normal = Normal::new(0.0, 1.0);
    match normal.auto_jit() {
        Ok(jit_fn) => {
            // Use compiled function
            let result = jit_fn.call(x);
        }
        Err(e) => {
            // Fall back to standard evaluation
            let result = normal.log_density().at(&x);
        }
    }
}
```

### Current Implementation Status

**Functional Components**:
- Expression compilation to Cranelift IR
- Basic arithmetic operations (add, subtract, multiply, divide)
- Control flow and variable handling
- Memory management for compiled functions

**Experimental Components**:
- **Transcendental functions**: Uses placeholder implementations for `ln()`, `exp()`, `sin()`, `cos()`
- **Mathematical accuracy**: May produce incorrect results for distributions requiring transcendental functions
- **Performance**: Currently slower than standard evaluation due to placeholder overhead
- **Stability**: Experimental status, not recommended for production use

### Technical Architecture

- **Compiler backend**: Cranelift code generator
- **IR generation**: Direct translation from expression trees
- **Function signatures**: Supports single variable and multi-parameter functions
- **Memory management**: Automatic cleanup of compiled code

### Future Development

**Planned Improvements**:
- Accurate transcendental function implementations
- LLVM backend integration for advanced optimizations
- Vectorization for batch evaluation
- Symbolic optimization passes

**Potential Benefits** (once fully implemented):
- Significant speedup for complex mathematical expressions
- Runtime specialization based on data patterns
- Advanced optimization techniques not available at compile time

### When to Avoid (Currently)

- Production applications requiring mathematical accuracy
- Distributions heavily dependent on transcendental functions
- Applications where compilation overhead is unacceptable
- Systems requiring deterministic performance characteristics

## Performance Analysis and Benchmarking

### Measuring Performance

Always benchmark your specific use case with realistic data:

```rust
use criterion::{black_box, Criterion, BenchmarkId};
use std::time::Instant;

fn benchmark_optimization_strategies(c: &mut Criterion) {
    let normal = Normal::new(0.0, 1.0);
    let test_values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
    
    // Standard evaluation
    c.bench_function("standard_evaluation", |b| {
        let ld = normal.log_density();
        b.iter(|| {
            for &x in &test_values {
                black_box(ld.at(&black_box(x)));
            }
        });
    });
    
    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    c.bench_function("zero_overhead", |b| {
        let optimized = normal.zero_overhead_optimize();
        b.iter(|| {
            for &x in &test_values {
                black_box(optimized(&black_box(x)));
            }
        });
    });
}
```

### Amortization Analysis

Understanding when optimization overhead pays off:

```rust
fn amortization_analysis() {
    let normal = Normal::new(0.0, 1.0);
    let evaluation_counts = vec![1, 10, 100, 1000, 10000, 100000];
    
    for count in evaluation_counts {
        // Measure total time including compilation
        let start = Instant::now();
        
        #[cfg(feature = "jit")]
        {
            let optimized = normal.zero_overhead_optimize();
            for i in 0..count {
                let x = i as f64 * 0.01;
                black_box(optimized(&x));
            }
        }
        
        let total_time = start.elapsed();
        let per_call = total_time / count as u32;
        
        println!("Count: {}, Total: {:?}, Per call: {:?}", 
                 count, total_time, per_call);
    }
}
```

### Performance Best Practices

#### 1. Avoid Recompilation in Hot Paths

```rust
// Inefficient: recompiles every iteration
for data_batch in batches {
    let optimized = distribution.zero_overhead_optimize();
    for x in data_batch {
        let result = optimized(&x);
    }
}

// Efficient: compile once, use repeatedly
let optimized = distribution.zero_overhead_optimize();
for data_batch in batches {
    for x in data_batch {
        let result = optimized(&x);
    }
}
```

#### 2. Consider Compilation Overhead

- **Zero-overhead optimization**: Microseconds compilation time
- **JIT compilation**: Milliseconds compilation time
- **Break-even point**: Depends on evaluation frequency and distribution complexity

#### 3. Profile Distribution-Specific Performance

Different distributions have different optimization characteristics:

```rust
// Simple distributions may not benefit from optimization
let uniform = Uniform::new(0.0, 1.0);

// Complex distributions may show larger improvements
let gamma = Gamma::new(shape, rate);
let beta = Beta::new(alpha, beta);
```

## Automatic Optimizations

The library automatically applies several mathematical and computational optimizations.

### Shared Root Measure Optimization

When computing relative densities between measures with the same root measure:

```rust
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Automatically applies: log(dμ₁/dμ₂) = log(dμ₁/dν) - log(dμ₂/dν)
// Avoids redundant computation of shared Lebesgue measure
let relative_density = normal1.log_density().wrt(normal2).at(&x);
```

### Static Dispatch Optimization

The type system enables complete static dispatch without runtime overhead:

```rust
// These become different monomorphized functions at compile time
fn evaluate_lebesgue(normal: &Normal<f64>, x: f64) -> f64 {
    normal.log_density().at(&x)  // Optimized for Lebesgue base measure
}

fn evaluate_relative(normal1: &Normal<f64>, normal2: &Normal<f64>, x: f64) -> f64 {
    normal1.log_density().wrt(normal2).at(&x)  // Optimized for relative density
}
```

### Compile-Time Constant Folding

Distribution parameters become compile-time constants when possible:

```rust
// Parameters (2.0, 1.5) embedded as constants in optimized code
let normal = Normal::new(2.0, 1.5);
let optimized = normal.zero_overhead_optimize();
```

## Memory and Resource Considerations

### Memory Usage Patterns

- **Standard evaluation**: No additional memory allocation in hot paths
- **Zero-overhead optimization**: Additional storage for pre-computed values and closures
- **JIT compilation**: Memory for compiled native code and metadata

### Resource Management

```rust
// Automatic cleanup of JIT resources
{
    let jit_fn = normal.auto_jit()?;
    // Use jit_fn...
} // Compiled code automatically freed here
```

### Scalability Considerations

- **Single-threaded performance**: Focus of current optimizations
- **Multi-threaded usage**: Thread-safe by design, no shared mutable state
- **Memory scaling**: Linear with number of optimized distributions

## Future Performance Roadmap

### Near-term Improvements

1. **Accurate JIT mathematical functions**: Replace placeholder implementations
2. **Vectorization support**: SIMD operations for batch evaluation
3. **Memory optimization**: Reduce overhead of optimization structures
4. **Benchmark suite expansion**: More comprehensive performance testing

### Long-term Research Directions

1. **GPU acceleration**: Parallel evaluation for large datasets
2. **Adaptive optimization**: Runtime strategy selection based on usage patterns
3. **Advanced symbolic optimization**: Algebraic simplification before compilation
4. **Distributed computation**: Scaling across multiple cores and machines

### Performance Evolution

Performance characteristics will continue to evolve as optimizations are implemented. The current focus is on correctness and mathematical accuracy, with performance improvements being added incrementally.

**Recommendation**: Always benchmark with your specific use case and the current library version. Performance characteristics may change significantly between releases as optimizations are refined and new features are added. 