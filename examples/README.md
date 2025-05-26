# Exponential Family IR Optimization Examples

This directory contains examples demonstrating the symbolic intermediate representation (IR) and egglog optimization for exponential family distributions.

## Main Example: `exponential_family_ir_example.rs`

This comprehensive example shows how exponential family log-density expressions are represented in symbolic IR and optimized using egglog. It demonstrates:

### 1. Normal Distribution
- **Formula**: `log p(x|μ,σ²) = -½log(2πσ²) - (x-μ)²/(2σ²)`
- **Before optimization**: Complex nested expression with redundant operations
- **After optimization**: Simplified expression with algebraic identities applied

### 2. Poisson Distribution  
- **Formula**: `log p(k|λ) = k·log(λ) - λ - log(k!)`
- **Exponential family form**: `η·T(k) - A(η) + log h(k)`
- Shows how egglog simplifies multiplication by 1, addition of 0, etc.

### 3. Gamma Distribution
- **Formula**: `log p(x|α,β) = (α-1)·log(x) - β·x + α·log(β) - log(Γ(α))`
- Demonstrates `exp(ln(x)) → x` simplification and other identities

### 4. Complex Exponential Family Expression
- Shows advanced algebraic simplifications like:
  - `ln(exp(x)) → x`
  - `(a + b) * x - a * x - b * x → 0` (distributive law)
  - Constant folding and identity elimination

### 5. IID (Independent and Identically Distributed) Case
- **Formula**: `log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)`
- Shows optimization of sum expressions and redundant exponential/logarithm pairs

## Running the Example

```bash
# Make sure you have the jit feature enabled
cargo run --example exponential_family_ir_example --features jit
```

## Expected Output

The example will show for each distribution:

1. **BEFORE optimization**: The original symbolic expression tree with complexity count
2. **AFTER optimization**: The simplified expression after egglog processing  
3. **Reduction statistics**: Number of operations reduced and percentage improvement
4. **Mathematical verification**: Confirms that both expressions evaluate to the same value

Example output format:
```
1. NORMAL DISTRIBUTION
   Formula: log p(x|μ,σ²) = -½log(2πσ²) - (x-μ)²/(2σ²)
   BEFORE optimization:
   (((-0.50 * (ln((2.00 * 3.14)) + ln(sigma_sq))) + ...
   Complexity: 23 operations
   AFTER egglog optimization:
   ((-0.50 * (1.84 + ln(sigma_sq))) + ...
   Complexity: 18 operations
   Reduction: 5 operations (21.7%)
   ✓ Mathematical equivalence verified (error: 1.23e-15)
```

## Key Optimizations Demonstrated

### Algebraic Identities
- `x + 0 → x`
- `x * 1 → x` 
- `x * 0 → 0`
- `x - x → 0`
- `x / x → 1`

### Logarithmic/Exponential Rules
- `ln(exp(x)) → x`
- `exp(ln(x)) → x`
- `ln(a) + ln(b) → ln(a * b)`
- `ln(a) - ln(b) → ln(a / b)`

### Advanced Simplifications
- Distributive law: `(a * x) + (b * x) → (a + b) * x`
- Constant folding: `ln(2.0 * π) → ln(6.28) → 1.84`
- Power simplifications: `x^2 → x * x`

## Other Examples

### `benchmark_comparison.rs`
Benchmarks the performance of egglog optimization on various expression types:

```bash
cargo run --example benchmark_comparison --features jit
```

This shows timing data and complexity reduction statistics for different optimization scenarios.

## Understanding the IR

The symbolic IR uses an enum-based expression tree:

```rust
pub enum Expr {
    Const(f64),           // Constant values
    Var(String),          // Variables like "x", "mu", "sigma"
    Add(Box<Expr>, Box<Expr>),  // Binary operations
    Mul(Box<Expr>, Box<Expr>),
    Ln(Box<Expr>),        // Unary functions
    Exp(Box<Expr>),
    // ... more operations
}
```

The egglog optimizer applies rewrite rules to find mathematically equivalent but simpler expressions, reducing both computational complexity and improving numerical stability. 