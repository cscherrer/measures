# 🔥 Measures Performance Profiling Infrastructure

This document describes the comprehensive profiling infrastructure built for the measures crate to identify and optimize performance bottlenecks, particularly around cloning overhead and factorial computations.

## 🎯 Overview

The profiling setup addresses the user's concern about "all the clones" by providing:

1. **CPU Performance Benchmarking** - Detailed timing analysis with flamegraphs
2. **Memory Allocation Profiling** - Heap usage tracking with dhat
3. **Clone Impact Analysis** - Specific benchmarks comparing cloning patterns
4. **Factorial Computation Analysis** - Poisson-specific performance testing
5. **Real-world Workload Simulation** - Monte Carlo, ML, and inference patterns

## 🛠️ Tools and Dependencies

### Core Profiling Stack
- **criterion** - CPU benchmarking with statistical analysis
- **pprof** - Flamegraph generation for CPU hotspot identification  
- **dhat** - Memory allocation profiling and heap analysis
- **inferno** - Enhanced flamegraph visualization
- **profiling** - Lightweight instrumentation framework

### Optional Advanced Tools
- **tracy-client** - Real-time profiling (feature: `tracy`)
- **iai-callgrind** - Instruction-level analysis (feature: `callgrind`)

## 📊 Benchmark Categories

### 1. Single Density Evaluations
Tests individual function call performance:
- `normal_log_density` - Full LogDensity wrapper
- `normal_exp_fam_log_density` - Exponential family method
- `normal_log_density_wrt_root` - Direct computation
- `poisson_*` variants - Factorial computation analysis

### 2. Batch Evaluations
Compares cloning strategies across different batch sizes (10, 100, 1000):
- `normal_repeated_clone` - Clone LogDensity per evaluation (worst case)
- `normal_reused_log_density` - Clone once, reuse (better)
- `normal_direct_computation` - No wrapper overhead (best)

### 3. Poisson Factorial Analysis
Stress tests factorial computation scaling:
- `poisson_full_computation` - Complete Poisson log-density
- `factorial_only` - Isolated factorial computation
- Tests k values: 0, 1, 5, 10, 20, 50

### 4. Exponential Family Components
Measures component creation overhead:
- `normal_to_natural` - Natural parameter conversion
- `normal_sufficient_statistic` - Sufficient statistic computation
- `normal_base_measure` - Base measure creation
- `poisson_base_measure_creation` - FactorialMeasure creation

### 5. Allocation Patterns
Memory-focused benchmarks (run with dhat for detailed analysis):
- `normal_minimal_allocations` - Direct computation path
- `normal_medium_allocations` - Exponential family path
- `normal_heavy_allocations` - Full LogDensity construction
- `batch_clone_per_evaluation` - Worst-case cloning pattern
- `batch_single_clone` - Optimized cloning pattern

### 6. Component Creation
Analyzes exponential family component overhead:
- Natural parameter creation
- Base measure instantiation
- Sufficient statistic computation
- Component-wise vs integrated computation

## 🧠 Memory Profiling

### dhat Integration
The `examples/memory_profiling.rs` provides comprehensive memory analysis:

```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
```

### Test Patterns
1. **Cloning Patterns** - Compare clone-per-evaluation vs reuse strategies
2. **Efficient Patterns** - Direct vs wrapped computation paths
3. **Factorial Allocations** - Poisson computation memory scaling
4. **Usage Patterns** - Monte Carlo, ML optimization, statistical inference

### Analysis Workflow
1. Run: `cargo run --features dhat-heap --example memory_profiling`
2. View `dhat-heap.json` with DHAT viewer (search for "dhat viewer" online or use Valgrind's `dh_view.html`)
3. Identify allocation hotspots and cloning overhead

## 🎲 Workload Simulation

### Real-world Patterns
`examples/profiling_workload.rs` simulates realistic usage:

1. **Monte Carlo Simulation** - 100k density evaluations
2. **ML Optimization** - Multiple distribution fitting
3. **Statistical Inference** - Relative density computations
4. **Factorial Stress Test** - Large k values for Poisson

### Performance Insights
- Tests both efficient and inefficient patterns
- Measures cumulative impact of design decisions
- Provides realistic performance baselines

## 🔧 Instrumentation

### Profiling Annotations
Key functions are instrumented with `#[profiling::function]`:
- `LogDensity::at()` - Main evaluation entry point
- `exp_fam_log_density()` - Exponential family computation
- `factorial_computation` - Factorial loop with k-specific scopes
- `EvaluateAt` implementations - All density evaluation paths

### Scope Tracking
Detailed scope analysis for:
- `exp_fam_computation` - Overall exponential family timing
- `dot_product` - Natural parameter · sufficient statistic
- `chain_rule` - Base measure log-density computation
- `factorial_loop` - Per-k factorial computation timing

## 🚀 Usage

### Quick Start
```bash
# Run all profiling
./profile.sh

# Individual components
cargo bench --bench density_computation
cargo run --features dhat-heap --example memory_profiling
cargo run --example profiling_workload --profile profiling
```

### Profile Configuration
```toml
[profile.profiling]
inherits = "release"
debug = true  # Enable debug symbols for profiling
```

### Feature Flags
- `dhat-heap` - Enable heap profiling
- `tracy` - Real-time profiling (optional)
- `callgrind` - Instruction-level analysis (optional)

## 📈 Results Analysis

### CPU Performance
1. Open `target/criterion/reports/index.html` for detailed benchmark results
2. View flamegraphs in `target/criterion/*/profile/flamegraph.svg`
3. Look for functions with high sample counts

### Memory Analysis
1. Load `dhat-heap.json` in DHAT viewer
2. Identify allocation patterns and peak usage
3. Focus on clone-heavy operations

### Key Metrics to Monitor
- **Clone overhead**: Compare `repeated_clone` vs `reused_log_density`
- **Factorial scaling**: Poisson performance vs k value
- **Wrapper cost**: Direct computation vs LogDensity wrapper
- **Component creation**: Exponential family instantiation overhead

## 🎯 Optimization Targets

Based on benchmark results, focus optimization efforts on:

1. **High-frequency cloning** - LogDensity construction patterns
2. **Factorial computation** - Poisson k-scaling bottlenecks  
3. **Component creation** - Exponential family instantiation
4. **Memory allocations** - Heap usage patterns in real workloads

## 🔍 Advanced Analysis

### Flamegraph Interpretation
- **Wide bars** = High CPU time (optimization targets)
- **Tall stacks** = Deep call chains (potential inlining opportunities)
- **Repeated patterns** = Cloning or allocation hotspots

### Memory Profile Insights
- **Total allocations** = Overall memory pressure
- **Peak usage** = Maximum heap consumption
- **Allocation patterns** = Clone frequency and size

### Performance Regression Detection
Criterion automatically detects performance changes:
- **Green** = Performance improvement
- **Red** = Performance regression  
- **Yellow** = Within noise threshold

This infrastructure provides comprehensive visibility into the measures crate's performance characteristics, enabling data-driven optimization decisions to address cloning overhead and computational bottlenecks. 