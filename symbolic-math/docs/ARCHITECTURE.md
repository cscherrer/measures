# Architecture Guide

This document explains the design principles and architecture of the symbolic-math crate.

## Design Philosophy

The symbolic-math crate is built around three core principles:

1. **Performance**: Multiple evaluation strategies optimized for different use cases
2. **Flexibility**: General-purpose symbolic representation that works for any mathematical domain
3. **Ergonomics**: Natural syntax and convenient builders for common patterns

## Core Architecture

```
symbolic-math/
├── src/
│   ├── expr.rs           # Core expression representation
│   ├── jit.rs            # JIT compilation (optional)
│   ├── optimization.rs   # Advanced optimization (optional)
│   └── lib.rs           # Public API and macros
├── examples/            # Profiling and usage examples
└── docs/               # Documentation
```

## Module Overview

### `expr.rs` - Core Expression System

The heart of the crate is the `Expr` enum, which represents mathematical expressions as an Abstract Syntax Tree (AST):

```rust
pub enum Expr {
    Const(f64),                    // Constants: 3.14, 2.0
    Var(String),                   // Variables: "x", "y"
    Add(Box<Expr>, Box<Expr>),     // Addition: a + b
    Sub(Box<Expr>, Box<Expr>),     // Subtraction: a - b
    Mul(Box<Expr>, Box<Expr>),     // Multiplication: a * b
    Div(Box<Expr>, Box<Expr>),     // Division: a / b
    Pow(Box<Expr>, Box<Expr>),     // Exponentiation: a^b
    Ln(Box<Expr>),                 // Natural logarithm: ln(a)
    Exp(Box<Expr>),                // Exponential: exp(a)
    Sqrt(Box<Expr>),               // Square root: sqrt(a)
    Sin(Box<Expr>),                // Sine: sin(a)
    Cos(Box<Expr>),                // Cosine: cos(a)
    Neg(Box<Expr>),                // Negation: -a
}
```

**Key Features:**
- **Immutable**: Expressions are immutable by design for safety and performance
- **Recursive**: Complex expressions are built from simpler ones
- **Type-safe**: Compile-time guarantees about expression structure
- **Extensible**: Easy to add new operations

**Performance Characteristics:**
- Expression creation: 59-175 ns depending on complexity
- Memory usage: Proportional to expression tree depth
- Evaluation: 109-962 ns/call for interpreted execution

### `jit.rs` - JIT Compilation System

The JIT module provides native code generation using Cranelift:

```rust
pub struct GeneralJITCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
}

pub struct GeneralJITFunction {
    function_ptr: *const u8,
    signature: JITSignature,
    compilation_stats: CompilationStats,
}
```

**Architecture:**
1. **Expression → CLIF IR**: Convert symbolic expressions to Cranelift IR
2. **CLIF IR → Machine Code**: Cranelift generates optimized x86-64 assembly
3. **Function Wrapping**: Safe Rust interface to native functions

**Signature Types:**
- `SingleInput`: `f(x) -> f64`
- `DataAndParameter`: `f(x, θ) -> f64`
- `DataAndParameters(n)`: `f(x, θ₁, θ₂, ..., θₙ) -> f64`
- `MultipleDataAndParameters`: `f(x₁, x₂, ..., θ₁, θ₂, ...) -> f64`

**Performance Characteristics:**
- Compilation: 356-1578 μs per expression
- Execution: 3.9-11.4 ns/call (13-84x speedup)
- Code size: ~128 bytes typical
- Break-even: 4-8 evaluations

### `optimization.rs` - Advanced Optimization

The optimization module uses egglog for deep mathematical simplification:

```rust
pub trait EgglogOptimize {
    fn optimize_with_egglog(&self) -> Result<Expr, String>;
}

pub struct EgglogOptimizer {
    // Internal egglog state
}
```

**Optimization Process:**
1. **Expression → egglog**: Convert to egglog representation
2. **Rule Application**: Apply mathematical rewrite rules
3. **Extraction**: Extract the simplified expression
4. **egglog → Expression**: Convert back to `Expr`

**Mathematical Rules:**
- Algebraic identities: `x + 0 = x`, `x * 1 = x`
- Trigonometric identities: `sin²(x) + cos²(x) = 1`
- Logarithmic rules: `ln(exp(x)) = x`, `exp(ln(x)) = x`
- Polynomial simplification: Factor and combine terms

**Performance Characteristics:**
- Optimization time: 8-232 ms
- Effectiveness: Can find identities basic simplification misses
- Memory usage: Proportional to expression complexity

## Evaluation Strategies

The crate provides three evaluation strategies, each optimized for different scenarios:

### 1. Interpreted Evaluation

```rust
let result = expr.evaluate(&variables)?;
```

**When to use:**
- Development and debugging
- One-time evaluations
- Simple expressions
- When JIT overhead isn't justified

**Performance:** 109-962 ns/call

### 2. JIT Compilation

```rust
let compiler = GeneralJITCompiler::new()?;
let jit_fn = compiler.compile_expression(&expr, &vars, &[], &HashMap::new())?;
let result = jit_fn.call_single(x);
```

**When to use:**
- Repeated evaluations (>100 calls)
- Performance-critical code
- Complex expressions
- Batch processing

**Performance:** 3.9-11.4 ns/call (13-84x speedup)

### 3. Advanced Optimization + JIT

```rust
let optimized = expr.optimize_with_egglog()?;
let jit_fn = compiler.compile_expression(&optimized, &vars, &[], &HashMap::new())?;
```

**When to use:**
- Offline preprocessing
- Mathematical correctness required
- Complex symbolic expressions

**Performance:** Best of both worlds after optimization

## Builder System

The builder system provides convenient APIs for common mathematical patterns:

```rust
pub mod builders {
    pub fn normal_log_pdf(x: impl Into<Expr>, mu: impl Into<Expr>, sigma: impl Into<Expr>) -> Expr;
    pub fn polynomial(x: impl Into<Expr>, coefficients: &[f64]) -> Expr;
    pub fn gaussian_kernel(x: impl Into<Expr>, mu: impl Into<Expr>, sigma: impl Into<Expr>) -> Expr;
}
```

**Design Principles:**
- **Flexible inputs**: Accept both `Expr` and primitive types
- **Domain-specific**: Optimized for common mathematical patterns
- **Composable**: Builders can be combined and nested

## Macro System

The macro system provides natural mathematical syntax:

```rust
let expr = expr!(x^2 + 2*x + 1);
let trig = expr!(sin(2*x) + cos(x));
```

**Implementation:**
- **Recursive macro**: Handles nested expressions
- **Operator precedence**: Respects mathematical precedence rules
- **Type conversion**: Automatically converts literals to `Expr::Const`

## Error Handling

The crate uses structured error handling throughout:

```rust
#[derive(Debug)]
pub enum EvalError {
    UnknownVariable(String),
    DivisionByZero,
    InvalidOperation(String),
    NumericalError(String),
}

#[derive(Debug)]
pub enum JITError {
    CompilationError(String),
    UnsupportedExpression(String),
    MemoryError(String),
    ModuleError(String),
}
```

**Error Strategy:**
- **Explicit errors**: All operations return `Result` types
- **Descriptive messages**: Clear error descriptions for debugging
- **Recoverable**: Most errors allow graceful fallback

## Memory Management

The crate is designed for efficient memory usage:

### Expression Trees
- **Boxed recursion**: Prevents stack overflow for deep expressions
- **Shared ownership**: `Clone` is cheap for shared subexpressions
- **Immutable**: No mutation reduces memory safety concerns

### JIT Functions
- **Owned modules**: JIT functions own their compiled code
- **Automatic cleanup**: Memory freed when function is dropped
- **Compact code**: Generated code is typically ~128 bytes

### Optimization
- **Temporary allocation**: egglog uses temporary memory during optimization
- **Result caching**: Could be added for repeated optimizations

## Thread Safety

The crate is designed to be thread-safe:

- **Immutable expressions**: `Expr` is `Send + Sync`
- **JIT functions**: Compiled functions are thread-safe
- **Compiler isolation**: Each thread should have its own `GeneralJITCompiler`

## Extension Points

The architecture supports several extension points:

### New Expression Types
```rust
// Add to Expr enum
Tan(Box<Expr>),
Atan(Box<Expr>),

// Implement in evaluation
Expr::Tan(inner) => inner.evaluate(vars)?.tan(),

// Implement in JIT
Expr::Tan(inner) => generate_tan_call(builder, inner_val),
```

### New Optimization Rules
```rust
// Add to egglog rules
"(tan (atan ?x))" => "?x",
"(atan (tan ?x))" => "?x",
```

### New Builder Patterns
```rust
pub fn exponential_distribution(x: impl Into<Expr>, lambda: impl Into<Expr>) -> Expr {
    let x = x.into();
    let lambda = lambda.into();
    // -lambda * x + ln(lambda)
    Expr::add(
        Expr::mul(Expr::neg(lambda.clone()), x),
        Expr::ln(lambda)
    )
}
```

## Performance Considerations

### Expression Construction
- **Avoid deep nesting**: Can cause stack overflow
- **Use builders**: More efficient than manual construction
- **Simplify early**: Reduces subsequent operation costs

### JIT Compilation
- **Compile once**: Reuse compiled functions
- **Simplify first**: Reduces compilation time
- **Batch operations**: Amortize compilation cost

### Memory Usage
- **Share subexpressions**: Use `Arc<Expr>` for shared parts
- **Limit optimization scope**: egglog can use significant memory
- **Profile memory**: Use tools to identify memory hotspots

## Future Architecture Improvements

1. **Expression interning**: Share identical subexpressions
2. **Vectorized operations**: SIMD support for batch evaluation
3. **Parallel compilation**: Multi-threaded JIT compilation
4. **Persistent caching**: Cache compiled functions across runs
5. **GPU compilation**: Extend to GPU targets via SPIR-V 