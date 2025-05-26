# API Reference

## Core Traits

### LogDensityBuilder

Entry point for computing log-densities:

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);

// Standard density (w.r.t. root measure)
let density = normal.log_density().at(&0.5);

// Relative density (w.r.t. another measure)
let other_normal = Normal::new(1.0, 2.0);
let relative_density = normal.log_density().wrt(other_normal).at(&0.5);
```

### EvaluateAt

Generic evaluation with different numeric types:

```rust
let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Different numeric types
let f64_result: f64 = ld.at(&1.0);
let f32_result: f32 = ld.at(&1.0_f32);
// let dual_result: Dual64 = ld.at(&dual_x);  // With autodiff library
```

## Distributions

### Continuous Distributions

```rust
use measures::{Normal, Gamma, Beta, Exponential, Cauchy, ChiSquared};

// Normal distribution
let normal = Normal::new(0.0, 1.0);  // mean, std_dev

// Gamma distribution  
let gamma = Gamma::new(2.0, 1.5);    // shape, rate

// Beta distribution
let beta = Beta::new(2.0, 3.0);      // alpha, beta

// Exponential distribution
let exponential = Exponential::new(1.5);  // rate

// Cauchy distribution
let cauchy = Cauchy::new(0.0, 1.0);  // location, scale

// Chi-squared distribution
let chi_squared = ChiSquared::new(3.0);  // degrees of freedom
```

### Discrete Distributions

```rust
use measures::{Poisson, Binomial, Bernoulli, Geometric, NegativeBinomial, Categorical};

// Poisson distribution
let poisson = Poisson::new(3.0);     // rate

// Binomial distribution
let binomial = Binomial::new(10, 0.3);  // n, p

// Bernoulli distribution
let bernoulli = Bernoulli::new(0.7); // p

// Geometric distribution
let geometric = Geometric::new(0.2); // p

// Negative binomial distribution
let neg_binomial = NegativeBinomial::new(5, 0.3);  // r, p

// Categorical distribution
let categorical = Categorical::new(vec![0.2, 0.3, 0.5]);  // probabilities
```

## Exponential Family Interface

### Natural Parameters and Sufficient Statistics

```rust
use measures::{Normal, Poisson};
use measures::exponential_family::ExponentialFamily;

let normal = Normal::new(2.0, 1.5);
let poisson = Poisson::new(3.0);

// Access natural parameters
let normal_params = normal.to_natural();
let poisson_params = poisson.to_natural();

// Compute sufficient statistics
let x = 1.0;
let k = 2;
let normal_stats = normal.sufficient_statistic(&x);
let poisson_stats = poisson.sufficient_statistic(&k);
```

### IID Collections

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

// Compute joint log-density for independent samples
let samples = vec![0.5, -0.3, 1.2, 0.8];
let joint_density = iid_normal.iid_log_density(&samples);

// Equivalent to sum of individual log-densities
let manual_sum: f64 = samples.iter()
    .map(|&x| normal.log_density().at(&x))
    .sum();
assert!((joint_density - manual_sum).abs() < 1e-10);
```

## Performance Optimization

### Zero-Overhead Optimization

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    
    let normal = Normal::new(2.0, 1.5);
    
    // Pre-compute constants and generate optimized closure
    let optimized_fn = normal.zero_overhead_optimize();
    
    // Use the optimized function
    let result = optimized_fn(&1.0);
}
```

### JIT Compilation (Experimental)

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    
    let normal = Normal::new(0.0, 1.0);
    
    // Compile to native machine code
    match normal.auto_jit() {
        Ok(jit_fn) => {
            let result = jit_fn.call(1.0);
        }
        Err(e) => {
            println!("JIT compilation failed: {}", e);
        }
    }
}
```

## Symbolic Computation

### Expression Building

```rust
use symbolic_math::Expr;

// Build expressions programmatically
let x = Expr::variable("x");
let two = Expr::constant(2.0);
let one = Expr::constant(1.0);

// 2*x + 1
let expr = Expr::add(
    Expr::mul(two, x),
    one
);

// Basic simplification
let simplified = expr.simplify();
```

### JIT Compilation of Expressions

```rust
#[cfg(feature = "jit")]
{
    use symbolic_math::{Expr, jit::GeneralJITCompiler};
    use std::collections::HashMap;
    
    // Create expression: x^2 + 2*x + 1
    let expr = Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Var("x".to_string()))
            ))
        )),
        Box::new(Expr::Const(1.0))
    );
    
    let compiler = GeneralJITCompiler::new()?;
    let jit_function = compiler.compile_expression(
        &expr,
        &["x".to_string()],  // variables
        &[],                 // parameters
        &HashMap::new(),     // constants
    )?;
    
    // Evaluate: (3^2 + 2*3 + 1) = 16
    let result = jit_function.call_single(3.0);
}
```

## Bayesian Modeling

### Expression Building

```rust
use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};

// Build Bayesian model expressions
let likelihood = normal_likelihood("x", "mu", "sigma");
let prior = normal_prior("mu", 0.0, 1.0);
let posterior = posterior_log_density(likelihood, prior);

// Note: JIT compilation not yet implemented for Bayesian expressions
```

## Error Handling

### JIT Compilation Errors

```rust
#[cfg(feature = "jit")]
{
    use measures::JITError;
    
    match normal.auto_jit() {
        Ok(jit_fn) => { /* use jit_fn */ }
        Err(JITError::CompilationFailed(msg)) => {
            println!("Compilation failed: {}", msg);
        }
        Err(JITError::UnsupportedOperation(op)) => {
            println!("Unsupported operation: {}", op);
        }
        Err(e) => {
            println!("Other JIT error: {}", e);
        }
    }
}
```

## Type Safety

### Compile-Time Measure Compatibility

```rust
use measures::{Normal, Poisson};

let normal = Normal::new(0.0, 1.0);
let poisson = Poisson::new(3.0);

// This works: both are continuous distributions with Lebesgue root measure
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);

// This would be a compile-time error: incompatible measure types
// let invalid = normal.log_density().wrt(poisson).at(&0.5);  // Error!
```

### Generic Numeric Types

```rust
// Same API works with different numeric types
fn compute_density<T, F>(distribution: T, x: F) -> F 
where
    T: LogDensityBuilder<F>,
    F: Copy,
{
    distribution.log_density().at(&x)
}

let normal_f64 = Normal::new(0.0, 1.0);
let normal_f32 = Normal::new(0.0_f32, 1.0_f32);

let result_f64: f64 = compute_density(normal_f64, 1.0);
let result_f32: f32 = compute_density(normal_f32, 1.0_f32);
```

## Common Patterns

### Caching Results

```rust
use std::collections::HashMap;

// Manual caching for expensive computations
let mut cache: HashMap<i32, f64> = HashMap::new();
let normal = Normal::new(0.0, 1.0);

for x in data {
    let key = (x * 1000.0) as i32;  // Discretize for caching
    let density = cache.entry(key)
        .or_insert_with(|| normal.log_density().at(&(key as f64 / 1000.0)));
}
```

### Batch Evaluation

```rust
// Efficient batch evaluation
let normal = Normal::new(0.0, 1.0);
let log_density = normal.log_density();

let results: Vec<f64> = data.iter()
    .map(|&x| log_density.at(&x))
    .collect();
```

### Parameter Estimation

```rust
// Maximum likelihood estimation pattern
fn log_likelihood(params: &[f64], data: &[f64]) -> f64 {
    let normal = Normal::new(params[0], params[1]);
    let log_density = normal.log_density();
    
    data.iter()
        .map(|&x| log_density.at(&x))
        .sum()
}
``` 