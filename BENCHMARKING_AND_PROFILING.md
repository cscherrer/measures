# Benchmarking and Profiling Guide for Rust Performance Analysis

This document provides a technical guide to performance analysis using the benchmarking and profiling infrastructure implemented in the measures crate. It covers methodology, tools, and result interpretation.

## Overview

Performance analysis in Rust requires a multi-faceted approach combining CPU profiling, memory analysis, and algorithmic complexity measurement. This guide demonstrates practical techniques using real examples from optimizing mathematical computations.

## Tool Selection and Setup

### CPU Performance Analysis

**Criterion** - Statistical benchmarking framework
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

**pprof** - Flamegraph generation for hotspot identification
```toml
pprof = { version = "0.13", features = ["flamegraph", "criterion"] }
```

**Inferno** - Enhanced flamegraph visualization
```bash
cargo install inferno
```

### Memory Profiling

**dhat** - Heap allocation tracking
```toml
[dependencies]
dhat = "0.3"

[features]
dhat-heap = []
```

**Configuration**:
```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
```

### Instrumentation

**profiling crate** - Lightweight scope tracking
```toml
profiling = "1.0"
```

Strategic placement on hot paths:
```rust
#[profiling::function]
fn critical_computation() {
    profiling::scope!("specific_operation");
    // computation here
}
```

## Benchmark Design Methodology

### 1. Isolation Testing

Test individual components to understand baseline costs:

```rust
fn bench_individual_operations(c: &mut Criterion) {
    let normal = Normal::new(0.0, 1.0);
    let x = 0.5;
    
    c.bench_function("direct_computation", |b| {
        b.iter(|| normal.log_density_wrt_root(&x))
    });
    
    c.bench_function("wrapped_computation", |b| {
        b.iter(|| normal.log_density().at(&x))
    });
}
```

**Interpretation**: Direct comparison reveals wrapper overhead. Expect wrapped versions to show 10-50% overhead depending on complexity.

### 2. Scaling Analysis

Test performance across different input sizes to identify algorithmic complexity:

```rust
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial_scaling");
    
    for k in [1, 5, 10, 20, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("O(k)_approach", k), &k, |b, &k| {
            b.iter(|| {
                let mut log_factorial = 0.0;
                for i in 1..=k {
                    log_factorial += (i as f64).ln();
                }
                log_factorial
            })
        });
        
        group.bench_with_input(BenchmarkId::new("O(1)_approach", k), &k, |b, &k| {
            b.iter(|| factorial_lookup_or_stirling(k))
        });
    }
    group.finish();
}
```

**Interpretation**: 
- O(k) approach: Linear time growth `time = base_cost * k`
- O(1) approach: Constant time regardless of k
- Plot results to visualize algorithmic differences

### 3. Pattern Comparison

Compare different usage patterns to guide API design:

```rust
fn bench_usage_patterns(c: &mut Criterion) {
    let normal = Normal::new(0.0, 1.0);
    let points: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    
    c.bench_function("clone_per_evaluation", |b| {
        b.iter(|| {
            for &x in &points {
                let ld = normal.log_density(); // Clone each time
                black_box(ld.at(&x));
            }
        })
    });
    
    c.bench_function("reuse_cloned_density", |b| {
        b.iter(|| {
            let ld = normal.log_density(); // Clone once
            for &x in &points {
                black_box(ld.at(&x));
            }
        })
    });
}
```

**Interpretation**: Quantifies the cost of different API usage patterns. Helps users understand performance implications of their code structure.

## Memory Profiling Methodology

### 1. Allocation Pattern Analysis

```rust
// examples/memory_profiling.rs
fn test_allocation_patterns() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    
    // Test different patterns
    test_clone_heavy_pattern();
    test_efficient_pattern();
    test_allocation_scaling();
}
```

### 2. Running Memory Analysis

```bash
# Generate memory profile
cargo run --features dhat-heap --example memory_profiling

# This creates a file called dhat-heap.json
```

### 3. Viewing dhat Profiles - Step by Step

**Option 1: Online Viewer (Easiest)**

1. Generate your profile:
   ```bash
   cargo run --features dhat-heap --example memory_profiling
   # This creates dhat-heap.json
   ```

2. Open the online DHAT viewer:
   **https://nnethercote.github.io/dh_view/dh_view.html**

3. Click "Load" button and select your `dhat-heap.json` file

That's it! No downloads or setup required.

**Option 2: Local Valgrind Viewer (Alternative)**

If you prefer a local viewer:
```bash
# Clone Valgrind repository  
git clone git://sourceware.org/git/valgrind.git
# Open valgrind/dhat/dh_view.html in your browser
```

**Important Version Note**: You need Valgrind 3.17 or later for the viewer to work with Rust dhat crate output. Earlier versions will show "data file is missing a field" errors.

### 4. What You'll See in the Viewer

The DHAT viewer shows a hierarchical tree of memory allocations with:

- **Total allocations**: Bytes and blocks allocated during execution
- **Peak memory usage**: Maximum heap size and when it occurred  
- **Allocation lifetimes**: How long allocations lived
- **Access patterns**: Read/write ratios for allocated memory
- **Call stacks**: Where allocations originated from

**Key metrics for optimization**:
- High allocation counts indicate cloning hotspots
- Short lifetimes suggest unnecessary temporary allocations
- Low read/write ratios indicate wasted memory
- Large allocations at peak show memory pressure points

## Flamegraph Analysis

### 1. Generating Flamegraphs

```bash
# With criterion integration
cargo bench --bench my_benchmark

# With cargo-flamegraph
cargo flamegraph --example profiling_workload
```

### 2. Reading Flamegraphs

**Width Interpretation**:
- Wide bars = High cumulative CPU time
- Focus optimization efforts on widest bars
- Narrow bars usually not worth optimizing

**Height Analysis**:
- Tall stacks = Deep call chains
- May indicate inlining opportunities
- Can reveal abstraction overhead

**Pattern Recognition**:
- Repeated similar patterns = Potential for batching
- Many tiny bars = High function call overhead
- Flat profiles = Well-optimized code

### 3. Identifying Optimization Targets

**CPU Hotspots**: Functions consuming >5% of total time
**Unexpected Overhead**: Wrapper functions taking significant time
**Algorithmic Issues**: Functions with time proportional to input size

## Case Study: Factorial Optimization

### Problem Identification

Original implementation:
```rust
fn slow_log_factorial(k: u64) -> f64 {
    let mut result = 0.0;
    for i in 1..=k {
        result += (i as f64).ln();
    }
    result
}
```

Benchmark revealed O(k) scaling:
```
k=100:  ~250ns
k=500:  ~1250ns  
k=1000: ~2500ns
```

### Solution Development

Hybrid approach with complexity analysis:
```rust
fn optimized_log_factorial(k: u64) -> f64 {
    if k <= 20 {
        // O(1): Precomputed lookup
        LOG_FACTORIAL_TABLE[k as usize]
    } else {
        // O(1): Stirling's approximation
        stirling_approximation(k)
    }
}
```

### Validation Methodology

1. **Performance Testing**: Verify O(1) complexity across input range
2. **Accuracy Testing**: Compare against reference implementations
3. **Integration Testing**: Ensure no regression in dependent code

Results:
```
k=100:  ~2.5ns (100x improvement)
k=500:  ~2.5ns (500x improvement)
k=1000: ~2.5ns (1000x improvement)
```

## Best Practices

### Benchmark Design

1. **Use `black_box()`** to prevent compiler optimizations from eliminating computations
2. **Warm up appropriately** for consistent timing
3. **Test realistic input ranges** that match actual usage
4. **Include both micro and macro benchmarks**

### Profiling Workflow

1. **Start with high-level profiling** to identify bottlenecks
2. **Drill down with targeted benchmarks** on hot functions
3. **Validate optimizations** with before/after comparisons
4. **Monitor for regressions** with continuous benchmarking

### Result Interpretation

1. **Focus on relative improvements** rather than absolute numbers
2. **Consider statistical significance** of benchmark results
3. **Validate real-world impact** with integration testing
4. **Document optimization trade-offs** (accuracy, complexity, maintainability)

## Common Pitfalls

### Measurement Issues
- **Insufficient iterations**: Results with high variance
- **Cold cache effects**: First iteration anomalies
- **Background processes**: Interfering with timing

### Optimization Mistakes
- **Premature optimization**: Optimizing non-bottlenecks
- **Micro-optimizations**: Ignoring algorithmic improvements
- **Breaking correctness**: Trading accuracy for speed inappropriately

### Profiling Artifacts
- **Debug mode profiling**: Unrepresentative performance
- **Profiler overhead**: Affecting measurement accuracy
- **Sampling bias**: Missing short-duration hotspots

## Continuous Performance Monitoring

### CI Integration
```bash
# In CI pipeline
cargo bench --bench performance_regression -- --output-format json
# Parse results to detect regressions
```

### Regression Detection
- Set thresholds for acceptable performance changes
- Alert on significant regressions
- Maintain performance history for trend analysis

## Tools Reference

### Command Reference
```bash
# Run all benchmarks
cargo bench

# Memory profiling
cargo run --features dhat-heap --example memory_profiling

# Generate flamegraph
cargo flamegraph --example workload

# Profile with custom options
cargo bench --profile profiling
```

### Configuration Files
- `Cargo.toml`: Benchmark dependencies and features
- `benches/`: Benchmark source files
- `examples/`: Profiling examples and workloads

This methodology provides a systematic approach to identifying, measuring, and optimizing performance bottlenecks in Rust applications. 