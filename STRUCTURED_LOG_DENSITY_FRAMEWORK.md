# Structured Log-Density Decomposition Framework

## Overview

This document describes the unified structured log-density decomposition framework implemented in the measures library. This framework provides a consistent API for both exponential families and non-exponential families, enabling efficient computation for IID samples and parameter optimization.

## Key Insight

**All log-densities can be decomposed into structured components**, regardless of whether the distribution is an exponential family or not:

```
log p(x|θ) = f_data(x) + f_param(θ) + f_mixed(x,θ) + f_const
```

Where:
- `f_data(x)`: Terms that depend only on data
- `f_param(θ)`: Terms that depend only on parameters  
- `f_mixed(x,θ)`: Terms that depend on both data and parameters
- `f_const`: Fixed constants

## Framework Components

### Core Types

```rust
pub struct LogDensityDecomposition<X, Theta, F> {
    pub data_terms: Vec<DataTerm<X, F>>,
    pub param_terms: Vec<ParamTerm<Theta, F>>,
    pub mixed_terms: Vec<MixedTerm<X, Theta, F>>,
    pub constant_terms: Vec<F>,
}

pub trait HasLogDensityDecomposition<X, Theta, F: Float> {
    fn log_density_decomposition(&self) -> LogDensityDecomposition<X, Theta, F>;
}
```

### Builder Pattern

```rust
let decomp = DecompositionBuilder::new()
    .constant(-0.5 * (2.0 * PI).ln())  // -0.5*log(2π)
    .param_term(|sigma| -sigma.ln(), "negative log scale")
    .mixed_term(|x, (mu, sigma)| {
        let z = (x - mu) / sigma;
        -0.5 * z * z
    }, "negative squared error")
    .build();
```

## Distribution Examples

### Normal Distribution (Exponential Family)

```
log p(x|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
             = f_const + f_param(σ) + f_mixed(x,μ,σ)
```

- **Constant**: `-0.5*log(2π)`
- **Parameter-only**: `-log(σ)`
- **Mixed**: `-0.5*(x-μ)²/σ²`

### Cauchy Distribution (Non-Exponential Family)

```
log p(x|x₀,γ) = -log(π) - log(γ) - log(1 + ((x-x₀)/γ)²)
              = f_const + f_param(γ) + f_mixed(x,x₀,γ)
```

- **Constant**: `-log(π)`
- **Parameter-only**: `-log(γ)`
- **Mixed**: `-log(1 + ((x-x₀)/γ)²)`

### Student's t Distribution (Non-Exponential Family)

```
log p(x|ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - (ν+1)/2*log(1 + x²/ν)
           = f_param(ν) + f_mixed(x,ν)
```

- **Parameter-only**: `log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ)`
- **Mixed**: `-(ν+1)/2*log(1 + x²/ν)`

## Computational Benefits

### 1. Efficient IID Sample Computation

For n IID samples x₁, x₂, ..., xₙ:

```
Σᵢ log p(xᵢ|θ) = Σᵢ f_data(xᵢ) + n·f_param(θ) + Σᵢ f_mixed(xᵢ,θ) + n·f_const
```

This avoids redundant computation of parameter-dependent terms.

### 2. Parameter Optimization

When optimizing parameters with fixed data:
- Cache data-dependent terms: `Σᵢ f_data(xᵢ)`
- Cache constants: `n·f_const`
- Only recompute: `n·f_param(θ) + Σᵢ f_mixed(xᵢ,θ)`

### 3. Incremental Updates

For streaming data, incrementally update:
- Add new data terms as they arrive
- Scale parameter terms by current sample size
- Maintain running sums efficiently

## Usage Examples

### Basic Usage

```rust
use measures::{Normal, Cauchy, StudentT, LogDensityBuilder};

// All distributions work with the same API
let normal = Normal::new(0.0, 1.0);
let cauchy = Cauchy::new(0.0, 1.0);
let student_t = StudentT::new(3.0);

let x = 1.5;

// Direct log-density computation
let normal_density: f64 = normal.log_density().at(&x);
let cauchy_density: f64 = cauchy.log_density().at(&x);
let t_density: f64 = student_t.log_density().at(&x);

// Structured decomposition
let normal_decomp = normal.log_density_decomposition();
let cauchy_decomp = cauchy.log_density_decomposition();
let t_decomp = student_t.log_density_decomposition();
```

### IID Sample Computation

```rust
let samples = vec![0.1, -0.5, 1.2, -0.8, 0.3];

// Efficient IID computation
let cauchy_ll = cauchy.log_density_iid(&samples);

// Manual verification
let manual_ll: f64 = samples.iter()
    .map(|&x| cauchy.log_density().at(&x))
    .sum();

assert!((cauchy_ll - manual_ll).abs() < 1e-10);
```

### Parameter Optimization

```rust
let samples = vec![0.1, -0.5, 1.2, -0.8, 0.3];
let decomp = cauchy.log_density_decomposition();

// Cache data-dependent terms (if any)
let data_cache: f64 = samples.iter()
    .map(|&x| decomp.evaluate_data_terms(&x))
    .sum();

// Test different parameters
for &scale in &[0.5, 1.0, 1.5, 2.0] {
    let params = (0.0, scale);
    
    // Only recompute parameter and mixed terms
    let param_terms = decomp.evaluate_param_terms(&params);
    let mixed_terms: f64 = samples.iter()
        .map(|&x| decomp.mixed_terms.iter()
            .map(|term| (term.compute)(&x, &params))
            .sum::<f64>())
        .sum();
    
    let log_likelihood = data_cache + 
                        param_terms * (samples.len() as f64) + 
                        mixed_terms;
    
    println!("Scale {}: LL = {:.6}", scale, log_likelihood);
}
```

### Caching Strategies

```rust
// Strategy 1: Fixed parameters, changing data
let param_terms = decomp.evaluate_param_terms(&params);
let constants = decomp.constant_sum();

for new_sample in new_data {
    let mixed_terms: f64 = decomp.mixed_terms.iter()
        .map(|term| (term.compute)(&new_sample, &params))
        .sum();
    
    // Incremental update
    running_total += mixed_terms + param_terms + constants;
}

// Strategy 2: Fixed data, changing parameters  
let data_cache: f64 = samples.iter()
    .map(|&x| decomp.evaluate_data_terms(&x))
    .sum();

for new_params in param_grid {
    let param_contribution = decomp.evaluate_param_terms(&new_params) 
                           * (samples.len() as f64);
    let mixed_contribution: f64 = samples.iter()
        .map(|&x| decomp.mixed_terms.iter()
            .map(|term| (term.compute)(&x, &new_params))
            .sum::<f64>())
        .sum();
    
    let total = data_cache + param_contribution + mixed_contribution;
}
```

## Framework Benefits

✅ **Unified API**: Same interface for exponential families and non-exponential families

✅ **Efficient IID Computation**: Avoid redundant parameter term computation

✅ **Optimized Parameter Optimization**: Cache data-dependent terms

✅ **Flexible Caching**: Support various caching strategies

✅ **Incremental Updates**: Efficient streaming data processing

✅ **Type Safety**: Compile-time guarantees about decomposition structure

✅ **Extensible**: Easy to add new distributions

## Implementation Notes

### Type Safety

The framework uses Rust's type system to ensure correctness:

```rust
impl<T: Float + FloatConst> HasLogDensityDecomposition<T, CauchyParams<T>, T> for Cauchy<T>
```

This ensures that:
- Data type `T` matches across all terms
- Parameter type `CauchyParams<T>` is consistent
- Float type `T` is used for all computations

### Performance

The decomposition framework adds minimal overhead:
- Function pointers for term computation
- Structured evaluation that can be optimized
- Caching opportunities that often provide net speedup

### Extensibility

Adding a new distribution requires:

1. Define parameter type: `pub type MyDistParams<T> = ...;`
2. Implement `HasLogDensityDecomposition`
3. Use `DecompositionBuilder` to specify terms
4. Add `log_density_iid` method for convenience

## Future Directions

1. **Automatic Differentiation**: Extend to work with dual numbers
2. **JIT Compilation**: Compile decompositions for maximum performance  
3. **Parallel Computation**: Parallelize IID sample computation
4. **GPU Support**: Extend to GPU computation
5. **More Distributions**: Add more non-exponential family distributions
6. **Advanced Caching**: Implement more sophisticated caching strategies

## Conclusion

The structured log-density decomposition framework provides a powerful, unified approach to probability density computation that works seamlessly across exponential families and non-exponential families. It enables significant performance optimizations while maintaining type safety and extensibility.

This framework demonstrates that **measure theory provides the right abstractions** for building efficient, general-purpose probabilistic computing systems. 