# Ergonomic Symbolic IR Improvements

This document summarizes the major improvements made to the symbolic IR system to provide a more ergonomic, Rust-like interface similar to working with `f64` values.

## Overview

The symbolic IR has been enhanced with:

1. **Natural Rust syntax** through operator overloading and conversion traits
2. **Improved readable output** with LaTeX and Python code generation
3. **Convenient macros** for expression building
4. **Builder functions** for common mathematical patterns
5. **Enhanced method chaining** for fluent expression construction

## Key Features

### 1. Natural Rust-like Syntax

**Before (Hand-written IR):**
```rust
let expr = Expr::Add(
    Box::new(Expr::Mul(
        Box::new(Expr::Const(2.0)), 
        Box::new(Expr::Var("x".to_string()))
    )),
    Box::new(Expr::Const(3.0))
);
```

**After (Ergonomic Interface):**
```rust
let x = var!("x");
let expr = 2.0 * x + 3.0;
```

### 2. Seamless Type Conversion

The IR now supports automatic conversion from common Rust types:

```rust
// From numeric types
let expr1 = Expr::from(3.14);
let expr2 = Expr::from(42);

// From strings
let expr3 = Expr::from("x");
let expr4 = Expr::from("variable_name".to_string());

// Mixed operations
let mixed = 2.0 * var!("x") + 3;  // Works seamlessly!
```

### 3. Method Chaining for Mathematical Functions

```rust
let x = var!("x");

// Fluent mathematical operations
let complex_expr = x.clone()
    .square()                    // x²
    .natural_log()              // ln(x²)
    .exponential()              // exp(ln(x²)) = x²
    .square_root();             // √(x²) = |x|
```

### 4. Enhanced Display Formats

#### Standard Display
```rust
let expr = (x - mu) / sigma;
println!("{}", expr);  // ((x - mu) / sigma)
```

#### LaTeX Output
```rust
println!("{}", display::latex(&expr));  // \frac{x - mu}{sigma}
```

#### Python Code Generation
```rust
println!("{}", display::python(&expr));
// Output:
// import math
// 
// def f(mu, sigma, x):
//     return ((x - mu) / sigma)
```

### 5. Builder Functions for Common Patterns

#### Normal Distribution Log-PDF
```rust
let normal_pdf = builders::normal_log_pdf(x, mu, sigma);
// Automatically generates: -0.5 * ((x - μ) / σ)² - ln(σ√(2π))
```

#### Polynomial Construction
```rust
let quadratic = builders::polynomial(x, &[1.0, -2.0, 1.0]);  // x² - 2x + 1
```

#### Gaussian Kernel
```rust
let kernel = builders::gaussian_kernel(x, 0.0, 1.0);  // exp(-0.5 * x²)
```

#### Logistic Function
```rust
let logistic = builders::logistic(x);  // 1 / (1 + exp(-x))
```

### 6. Macro-Based Expression Building

```rust
// Simple expressions
let x = var!("x");
let constant = const_expr!(3.14);

// Complex expressions with natural syntax
let expr = expr!(2.0 * x + sin(y) - ln(z));
```

### 7. Enhanced Bayesian Modeling

**Before:**
```rust
// Complex hand-built expressions
let x = Expr::Var("x".to_string());
let mu = Expr::Var("mu".to_string());
let sigma = Expr::Var("sigma".to_string());

let diff = Expr::Sub(Box::new(x), Box::new(mu));
let standardized = Expr::Div(Box::new(diff), Box::new(sigma.clone()));
// ... many more lines of boilerplate
```

**After:**
```rust
// Clean, readable Bayesian models
let likelihood = builders::normal_log_pdf(x, mu.clone(), sigma.clone());
let prior = builders::normal_log_pdf(mu, 0.0, 10.0);
let posterior = likelihood + prior;

println!("Posterior: {}", display::equation("log p(μ|x)", &posterior));
```

### 8. Hierarchical Models Made Simple

```rust
let hierarchical = expressions::hierarchical_normal(
    "x", "mu", "sigma", "tau", 0.0, 1.0
);
// Automatically combines:
// - Likelihood: x ~ Normal(μ, σ)
// - Prior on μ: μ ~ Normal(0, τ)  
// - Prior on σ: log(σ) ~ Normal(0, 1)
```

### 9. Mixture Models

```rust
let mixture = expressions::mixture_likelihood(
    "x", 
    &[0.3, 0.7],           // weights
    &[-2.0, 2.0],          // means
    &[1.0, 1.5]            // standard deviations
);
```

## Performance Benefits

1. **Compile-time optimization**: The ergonomic interface doesn't add runtime overhead
2. **Expression simplification**: Automatic algebraic simplification reduces complexity
3. **JIT compilation ready**: All expressions can still be compiled to optimized machine code
4. **Memory efficiency**: Smart use of `Clone` and move semantics

## Backward Compatibility

All existing code continues to work unchanged. The ergonomic interface is purely additive:

- Original `Expr` enum and methods remain unchanged
- Existing builder functions in `expr::builders` still work
- JIT compilation interface is unaffected
- All tests pass without modification

## Usage Examples

### Basic Arithmetic
```rust
let x = var!("x");
let y = var!("y");

let linear = 2.0 * x.clone() + 3.0 * y.clone();
let quadratic = x.square() + 2.0 * x + 1.0;
```

### Statistical Models
```rust
// Normal distribution
let normal = builders::normal_log_pdf("x", "mu", "sigma");

// Bayesian inference
let posterior = normal + builders::normal_prior("mu", 0.0, 10.0);

// Model evaluation
let mut params = HashMap::new();
params.insert("x".to_string(), 1.0);
params.insert("mu".to_string(), 0.0);
params.insert("sigma".to_string(), 1.0);

let log_prob = posterior.evaluate(&params)?;
```

### Complex Mathematical Functions
```rust
let x = var!("x");

// Trigonometric
let trig = x.sin() + (2.0 * x).cos();

// Exponential and logarithmic
let exp_log = (x + 1.0).natural_log().exponential();

// Power functions
let power = x.powf(2.5) + x.cube().square_root();
```

## Testing

Comprehensive tests ensure correctness:

```bash
cargo test --example ergonomic_ir_demo
cargo run --example ergonomic_ir_demo
```

All tests pass, demonstrating:
- Correct expression evaluation
- Proper display formatting
- Builder function accuracy
- Type conversion reliability

## Future Enhancements

Potential areas for further improvement:

1. **Advanced precedence-aware display formatting**
2. **More mathematical function builders** (Beta, Gamma distributions, etc.)
3. **Automatic differentiation support**
4. **Integration with plotting libraries**
5. **WASM compilation targets**

## Conclusion

The ergonomic improvements transform the symbolic IR from a low-level, verbose system into a high-level, intuitive interface that feels natural to Rust developers. Mathematical expressions can now be written almost exactly as they would appear in mathematical notation or standard programming languages, while maintaining all the power and performance of the underlying symbolic system. 