# API Reference

## Core Traits and Interfaces

### LogDensityBuilder

The primary interface for constructing and evaluating log-density functions. This trait provides the entry point for all density computations.

```rust
use measures::{Normal, LogDensityBuilder};

let normal = Normal::new(0.0, 1.0);

// Standard density computation (w.r.t. root measure)
let density = normal.log_density().at(&0.5);

// Relative density computation (w.r.t. another measure)
let other_normal = Normal::new(1.0, 2.0);
let relative_density = normal.log_density().wrt(other_normal).at(&0.5);
```

**Key Methods**:
- `log_density()`: Creates a log-density object for the distribution
- `wrt(base_measure)`: Changes the base measure for relative density computation
- `at(point)`: Evaluates the log-density at a specific point

### EvaluateAt<T, F>

Generic evaluation trait that enables the same mathematical object to work with different numeric types.

```rust
let normal = Normal::new(0.0, 1.0);
let ld = normal.log_density();

// Standard floating-point evaluation
let f64_result: f64 = ld.at(&1.0);
let f32_result: f32 = ld.at(&1.0_f32);

// Automatic differentiation (with appropriate feature flags)
// let dual_result: Dual64 = ld.at(&dual_x);
// let hyperdual_result: HyperDual64 = ld.at(&hyperdual_x);
```

**Type Parameters**:
- `T`: Input type (the point at which to evaluate)
- `F`: Output type (the numeric type for the result)

### Measure<T>

Fundamental trait defining the measure hierarchy and root measure relationships.

```rust
trait Measure<T> {
    type RootMeasure: Measure<T>;
    fn root_measure(&self) -> Self::RootMeasure;
}
```

**Implementation Examples**:
```rust
// Primitive measures are self-rooted
impl Measure<f64> for LebesgueMeasure<f64> {
    type RootMeasure = Self;
}

// Derived measures reference their root
impl Measure<f64> for Normal<f64> {
    type RootMeasure = LebesgueMeasure<f64>;
}
```

## Probability Distributions

### Continuous Distributions

All continuous distributions are rooted in the Lebesgue measure and support the full API.

```rust
use measures::{Normal, Gamma, Beta, Exponential, Cauchy, ChiSquared, Uniform};

// Normal distribution: N(μ, σ²)
let normal = Normal::new(0.0, 1.0);  // mean, standard_deviation

// Gamma distribution: Γ(α, β) with shape α and rate β
let gamma = Gamma::new(2.0, 1.5);    // shape, rate

// Beta distribution: Beta(α, β)
let beta = Beta::new(2.0, 3.0);      // alpha, beta

// Exponential distribution: Exp(λ) with rate λ
let exponential = Exponential::new(1.5);  // rate

// Cauchy distribution: Cauchy(x₀, γ)
let cauchy = Cauchy::new(0.0, 1.0);  // location, scale

// Chi-squared distribution: χ²(k) with k degrees of freedom
let chi_squared = ChiSquared::new(3.0);  // degrees_of_freedom

// Uniform distribution: U(a, b)
let uniform = Uniform::new(0.0, 1.0);  // lower_bound, upper_bound
```

**Parameter Validation**: All constructors validate parameters and may panic on invalid inputs (e.g., negative variance, zero scale parameters).

### Discrete Distributions

Discrete distributions are rooted in the counting measure and work with integer-valued data.

```rust
use measures::{Poisson, Binomial, Bernoulli, Geometric, NegativeBinomial, Categorical};

// Poisson distribution: Pois(λ) with rate λ
let poisson = Poisson::new(3.0);     // rate

// Binomial distribution: Bin(n, p) with n trials and success probability p
let binomial = Binomial::new(10, 0.3);  // n_trials, success_probability

// Bernoulli distribution: Ber(p) with success probability p
let bernoulli = Bernoulli::new(0.7); // success_probability

// Geometric distribution: Geom(p) with success probability p
let geometric = Geometric::new(0.2); // success_probability

// Negative binomial distribution: NB(r, p)
let neg_binomial = NegativeBinomial::new(5, 0.3);  // r, success_probability

// Categorical distribution with k categories
let categorical = Categorical::new(vec![0.2, 0.3, 0.5]);  // probabilities
```

**Data Types**: Discrete distributions typically work with `u64` for count data and `usize` for categorical data.

## Exponential Family Framework

### ExponentialFamily<T>

Unified interface for exponential family distributions using natural parameterization.

```rust
use measures::{Normal, Poisson};
use measures::exponential_family::ExponentialFamily;

let normal = Normal::new(2.0, 1.5);
let poisson = Poisson::new(3.0);

// Access natural parameters θ
let normal_params = normal.to_natural();
let poisson_params = poisson.to_natural();

// Compute sufficient statistics T(x)
let x = 1.0;
let k = 2;
let normal_stats = normal.sufficient_statistic(&x);
let poisson_stats = poisson.sufficient_statistic(&k);

// Access log-partition function A(θ)
let normal_log_partition = normal.log_partition();
let poisson_log_partition = poisson.log_partition();
```

**Mathematical Form**: Exponential family distributions have the canonical form:
```
p(x|θ) = exp(θᵀT(x) - A(θ)) h(x)
```

### IIDExtension<T>

Extension trait for efficient computation with independent and identically distributed samples.

```rust
use measures::{Normal, IIDExtension};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

// Compute joint log-density for independent samples
let samples = vec![0.5, -0.3, 1.2, 0.8];
let joint_density = iid_normal.iid_log_density(&samples);

// Mathematical equivalence verification
let manual_sum: f64 = samples.iter()
    .map(|&x| normal.log_density().at(&x))
    .sum();
assert!((joint_density - manual_sum).abs() < 1e-10);
```

**Optimization**: IID collections use optimized algorithms that avoid redundant parameter computations.

## Performance Optimization

### Zero-Overhead Optimization

Compile-time optimization that pre-computes distribution parameters and generates specialized evaluation closures.

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::jit::ZeroOverheadOptimizer;
    
    let normal = Normal::new(2.0, 1.5);
    
    // Pre-compute constants and generate optimized closure
    let optimized_fn = normal.zero_overhead_optimize();
    
    // Use the optimized function for repeated evaluations
    let results: Vec<f64> = data_points.iter()
        .map(|&x| optimized_fn(&x))
        .collect();
}
```

**Use Cases**:
- High-frequency evaluation with fixed distribution parameters
- Performance-critical inner loops
- Batch processing of large datasets

### JIT Compilation (Experimental)

Runtime compilation of mathematical expressions to native machine code.

```rust
#[cfg(feature = "jit")]
{
    use measures::exponential_family::AutoJITExt;
    
    let normal = Normal::new(0.0, 1.0);
    
    // Attempt JIT compilation with fallback
    match normal.auto_jit() {
        Ok(jit_fn) => {
            let result = jit_fn.call(1.0);
            println!("JIT result: {}", result);
        }
        Err(e) => {
            println!("JIT compilation failed: {}, using standard evaluation", e);
            let result = normal.log_density().at(&1.0);
        }
    }
}
```

**Current Limitations**:
- Experimental status with placeholder transcendental function implementations
- May produce incorrect results for distributions requiring `ln()`, `exp()`, `sin()`, `cos()`
- Performance overhead compared to standard evaluation
- Not recommended for production use

## Symbolic Computation

### Traditional AST Approach

Expression building using abstract syntax trees.

```rust
use symbolic_math::Expr;

// Build expressions programmatically
let x = Expr::variable("x");
let two = Expr::constant(2.0);
let one = Expr::constant(1.0);

// Construct: 2*x + 1
let expr = Expr::add(
    Expr::mul(two, x),
    one
);

// Apply basic simplification rules
let simplified = expr.simplify();
```

### Final Tagless Approach

Polymorphic expressions that work with multiple interpreters.

```rust
use symbolic_math::final_tagless::{DirectEval, PrettyPrint, MathExpr};

// Define expression polymorphically
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
where E::Repr<f64>: Clone,
{
    let a = E::constant(2.0);
    let b = E::constant(3.0);
    let c = E::constant(1.0);
    E::add(E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)), c)
}

// Direct evaluation
let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));

// Pretty printing
let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
println!("Expression: {}", pretty);

// JIT compilation (experimental)
#[cfg(feature = "jit")]
{
    use symbolic_math::final_tagless::JITEval;
    let jit_expr = quadratic::<JITEval>(JITEval::var("x"));
    let compiled = JITEval::compile_single_var(jit_expr, "x")?;
    let jit_result = compiled.call_single(2.0);
}
```

### JIT Compilation of Expressions

```rust
#[cfg(feature = "jit")]
{
    use symbolic_math::{Expr, jit::GeneralJITCompiler};
    use std::collections::HashMap;
    
    // Create expression: x² + 2x + 1
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
        &["x".to_string()],  // data variables
        &[],                 // parameter variables
        &HashMap::new(),     // embedded constants
    )?;
    
    // Evaluate: (3² + 2×3 + 1) = 16
    let result = jit_function.call_single(3.0);
    assert_eq!(result, 16.0);
}
```

## Bayesian Modeling

### Expression Building

Basic expression construction for Bayesian models.

```rust
use measures::bayesian::expressions::{normal_likelihood, normal_prior, posterior_log_density};

// Build Bayesian model expressions
let likelihood = normal_likelihood("x", "mu", "sigma");
let prior = normal_prior("mu", 0.0, 1.0);
let posterior = posterior_log_density(likelihood, prior);

// Note: Currently supports expression building only
// JIT compilation not yet implemented for Bayesian expressions
```

**Current Status**: The Bayesian module provides basic expression building capabilities but does not yet support JIT compilation or advanced optimization.

## Error Handling

### JIT Compilation Errors

Comprehensive error handling for JIT compilation failures.

```rust
#[cfg(feature = "jit")]
{
    use measures::JITError;
    
    match normal.auto_jit() {
        Ok(jit_fn) => {
            // Successful compilation
            let result = jit_fn.call(1.0);
        }
        Err(JITError::CompilationFailed(msg)) => {
            eprintln!("Compilation failed: {}", msg);
            // Fall back to standard evaluation
        }
        Err(JITError::UnsupportedOperation(op)) => {
            eprintln!("Unsupported operation: {}", op);
            // Use alternative approach
        }
        Err(JITError::InvalidParameters(msg)) => {
            eprintln!("Invalid parameters: {}", msg);
            // Check parameter validity
        }
        Err(e) => {
            eprintln!("Other JIT error: {}", e);
            // Generic error handling
        }
    }
}
```

### Parameter Validation

Distribution constructors validate parameters and provide clear error messages.

```rust
// These will panic with descriptive error messages
// let invalid_normal = Normal::new(0.0, -1.0);  // Negative standard deviation
// let invalid_gamma = Gamma::new(-1.0, 1.0);    // Negative shape parameter
// let invalid_beta = Beta::new(0.0, 1.0);       // Zero alpha parameter

// Proper error handling for user input
fn create_normal_safe(mean: f64, std_dev: f64) -> Result<Normal<f64>, String> {
    if std_dev <= 0.0 {
        Err(format!("Standard deviation must be positive, got {}", std_dev))
    } else {
        Ok(Normal::new(mean, std_dev))
    }
}
```

## Type Safety and Compile-Time Guarantees

### Measure Compatibility Checking

The type system prevents incompatible measure operations at compile time.

```rust
use measures::{Normal, Poisson};

let normal = Normal::new(0.0, 1.0);
let poisson = Poisson::new(3.0);

// Valid: both continuous distributions with Lebesgue root measure
let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);

// Compile-time error: incompatible measure types
// let invalid = normal.log_density().wrt(poisson).at(&0.5);  // Error!
```

### Generic Numeric Type Support

The API works uniformly across different numeric types.

```rust
// Generic function that works with any compatible numeric type
fn compute_density<T, F>(distribution: T, x: F) -> F 
where
    T: LogDensityBuilder<F>,
    F: Copy + std::fmt::Debug,
{
    let result = distribution.log_density().at(&x);
    println!("Density at {:?}: {:?}", x, result);
    result
}

// Works with different precision levels
let normal_f64 = Normal::new(0.0, 1.0);
let normal_f32 = Normal::new(0.0_f32, 1.0_f32);

let result_f64: f64 = compute_density(normal_f64, 1.0);
let result_f32: f32 = compute_density(normal_f32, 1.0_f32);
```

## Common Usage Patterns

### Efficient Batch Evaluation

Optimize for evaluating the same distribution at multiple points.

```rust
// Efficient: reuse log-density object
let normal = Normal::new(0.0, 1.0);
let log_density = normal.log_density();

let results: Vec<f64> = data_points.iter()
    .map(|&x| log_density.at(&x))
    .collect();

// Less efficient: recreate log-density each time
let results_slow: Vec<f64> = data_points.iter()
    .map(|&x| normal.log_density().at(&x))  // Recreates object each time
    .collect();
```

### Caching for Expensive Computations

Manual caching strategy for repeated evaluations.

```rust
use std::collections::HashMap;

struct CachedDensity {
    distribution: Normal<f64>,
    cache: HashMap<i64, f64>,
    precision: f64,
}

impl CachedDensity {
    fn new(distribution: Normal<f64>, precision: f64) -> Self {
        Self {
            distribution,
            cache: HashMap::new(),
            precision,
        }
    }
    
    fn evaluate(&mut self, x: f64) -> f64 {
        let key = (x / self.precision).round() as i64;
        *self.cache.entry(key).or_insert_with(|| {
            let discretized_x = key as f64 * self.precision;
            self.distribution.log_density().at(&discretized_x)
        })
    }
}
```

### Maximum Likelihood Estimation

Pattern for parameter estimation using log-likelihood optimization.

```rust
// Log-likelihood function for normal distribution
fn normal_log_likelihood(params: &[f64], data: &[f64]) -> f64 {
    if params.len() != 2 || params[1] <= 0.0 {
        return f64::NEG_INFINITY;  // Invalid parameters
    }
    
    let normal = Normal::new(params[0], params[1]);  // mean, std_dev
    let log_density = normal.log_density();
    
    data.iter()
        .map(|&x| log_density.at(&x))
        .sum()
}

// Usage with optimization library
fn estimate_parameters(data: &[f64]) -> (f64, f64) {
    // Use with your preferred optimization library
    // let result = optimize(normal_log_likelihood, initial_guess, data);
    // (result.params[0], result.params[1])
    (0.0, 1.0)  // Placeholder
}
```

### Relative Density Computation

Computing how one distribution relates to another.

```rust
// Compute relative density between two normal distributions
fn relative_normal_density(
    target: Normal<f64>,
    reference: Normal<f64>,
    x: f64
) -> f64 {
    target.log_density().wrt(reference).at(&x)
}

// Batch relative density computation
fn batch_relative_density(
    target: Normal<f64>,
    reference: Normal<f64>,
    points: &[f64]
) -> Vec<f64> {
    let relative_ld = target.log_density().wrt(reference);
    points.iter()
        .map(|&x| relative_ld.at(&x))
        .collect()
}
```

This API reference provides comprehensive coverage of the library's capabilities while maintaining focus on practical usage patterns and technical accuracy. 