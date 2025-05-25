//! Symbolic Code Generation for Optimized Log-Density Evaluation.
//!
//! This example demonstrates how to use symbolic mathematics to:
//! 1. Represent log-density expressions symbolically
//! 2. Simplify them to efficient forms  
//! 3. Generate specialized Rust code for those forms
//! 4. Compare performance against standard evaluation
//!
//! This approach provides extremely efficient specialized functions
//! with all constants precomputed at generation time.
//!
//! Requires the "symbolic" feature: cargo run --example `symbolic_code_generation` --features symbolic

#[cfg(feature = "symbolic")]
use rusymbols::Expression;
#[cfg(feature = "symbolic")]
use std::collections::HashMap;

#[cfg(feature = "symbolic")]
use measures::{LogDensityBuilder, Normal};

#[cfg(feature = "symbolic")]
fn main() {
    println!("=== Symbolic Code Generation for Log-Density Optimization ===\n");

    // Example: Normal distribution with specific parameters
    let normal = Normal::new(2.0, 1.5);

    println!("Target distribution: Normal(μ=2.0, σ=1.5)");
    println!("Goal: Generate optimized code for this specific distribution\n");

    // Step 1: Create symbolic representation
    let optimized_normal = create_symbolic_normal_log_density(&normal);

    // Demonstrate symbolic evaluation
    println!("=== Symbolic Evaluation Test ===");
    for &x in &[0.0, 1.0, 2.0] {
        if let Some(result) = optimized_normal.evaluate(x) {
            let expected = normal.log_density().at(&x);
            println!(
                "  x={:.1}: symbolic={:.8}, expected={:.8}, diff={:.2e}",
                x,
                result,
                expected,
                (result - expected).abs()
            );
        }
    }
    println!();

    // Step 2: Generate specialized code
    let generated_code = generate_optimized_code(&optimized_normal, &normal);

    // Step 3: Performance comparison
    performance_comparison(&normal, &generated_code);

    // Step 4: Show broader applicability
    demonstrate_general_approach();
}

#[cfg(feature = "symbolic")]
struct SymbolicLogDensity {
    /// The symbolic expression for the log-density
    expression: Expression,
    /// Parameter values (for code generation)
    parameters: HashMap<String, f64>,
    /// Variable names that remain symbolic (e.g., "x")
    variables: Vec<String>,
}

#[cfg(feature = "symbolic")]
impl SymbolicLogDensity {
    fn evaluate(&self, x: f64) -> Option<f64> {
        let mut vars = self.parameters.clone();
        vars.insert("x".to_string(), x);

        // Convert &str keys to satisfy eval_args signature
        let vars_str: HashMap<&str, f64> = vars.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        self.expression.eval_args(&vars_str)
    }
}

#[cfg(feature = "symbolic")]
fn create_symbolic_normal_log_density(normal: &Normal<f64>) -> SymbolicLogDensity {
    println!("=== Step 1: Symbolic Representation ===");

    // Create symbolic variables
    let x = Expression::new_var("x");
    let mu = Expression::new_val(normal.mean);
    let sigma = Expression::new_val(normal.std_dev);

    // Build the complete log-density symbolically
    // log p(x|μ,σ) = -½log(2πσ²) - (x-μ)²/(2σ²)

    let two = Expression::new_val(2.0);
    let sigma_squared = sigma.clone() * sigma.clone();

    // Since rusymbols doesn't have .ln(), we'll build the expression conceptually
    // and use numerical values for the log terms
    let log_2pi_sigma_sq = (2.0 * std::f64::consts::PI * normal.std_dev * normal.std_dev).ln();
    let log_norm_constant = Expression::new_val(-0.5 * log_2pi_sigma_sq);

    // Quadratic term: -(x-μ)²/(2σ²)
    let x_minus_mu = x.clone() - mu.clone();
    let quadratic_term = -(x_minus_mu.clone() * x_minus_mu) / (two * sigma_squared);

    // Complete expression (with numerical log part)
    let full_expression = log_norm_constant + quadratic_term;

    println!("Symbolic expression created with numerical log constant");
    println!("Structure: constant + quadratic_term where quadratic_term = -(x-μ)²/(2σ²)");
    println!("Expression: {full_expression}");

    // Store parameters for code generation
    let mut parameters = HashMap::new();
    parameters.insert("mu".to_string(), normal.mean);
    parameters.insert("sigma".to_string(), normal.std_dev);

    SymbolicLogDensity {
        expression: full_expression,
        parameters,
        variables: vec!["x".to_string()],
    }
}

#[cfg(feature = "symbolic")]
struct GeneratedCode {
    function_code: String,
    constants: HashMap<String, f64>,
}

#[cfg(feature = "symbolic")]
impl GeneratedCode {
    /// Evaluate the generated function (simulated)
    fn evaluate(&self, x: f64) -> f64 {
        // In a real implementation, this would compile and call the generated code
        // For now, we'll simulate with the known efficient formula
        let mu = self.constants["mu"];
        let sigma = self.constants["sigma"];

        let sigma_sq = sigma * sigma;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let diff = x - mu;
        let quadratic_term = -(diff * diff) / (2.0 * sigma_sq);

        log_norm_constant + quadratic_term
    }

    /// Generate a real callable function from the constants
    fn to_callable_function(&self) -> Box<dyn Fn(f64) -> f64> {
        let mu = self.constants["mu"];
        let log_norm_constant = self.constants["log_norm_constant"];
        let inv_2sigma_sq = self.constants["inv_2sigma_sq"];

        Box::new(move |x: f64| -> f64 {
            let diff = x - mu;
            log_norm_constant - diff * diff * inv_2sigma_sq
        })
    }
}

#[cfg(feature = "symbolic")]
fn generate_optimized_code(symbolic: &SymbolicLogDensity, normal: &Normal<f64>) -> GeneratedCode {
    println!("=== Step 2: Code Generation ===");

    // Pre-compute constants at generation time
    let mu = normal.mean;
    let sigma = normal.std_dev;
    let sigma_sq = sigma * sigma;
    let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
    let inv_2sigma_sq = 1.0 / (2.0 * sigma_sq);

    // Generate optimized Rust code
    let function_code = format!(
        r"
// Generated optimized log-density function for Normal(μ={}, σ={})
// Based on symbolic expression: {}
#[inline]
fn optimized_normal_log_density(x: f64) -> f64 {{
    const LOG_NORM_CONSTANT: f64 = {:.16};
    const MU: f64 = {:.16};
    const INV_2SIGMA_SQ: f64 = {:.16};
    
    let diff = x - MU;
    LOG_NORM_CONSTANT - diff * diff * INV_2SIGMA_SQ
}}
",
        mu, sigma, symbolic.expression, log_norm_constant, mu, inv_2sigma_sq
    );

    println!("Generated function:\n{function_code}");

    // Store constants for simulation
    let mut constants = HashMap::new();
    constants.insert("mu".to_string(), mu);
    constants.insert("sigma".to_string(), sigma);
    constants.insert("log_norm_constant".to_string(), log_norm_constant);
    constants.insert("inv_2sigma_sq".to_string(), inv_2sigma_sq);

    println!("Pre-computed constants:");
    for (name, value) in &constants {
        if name != "mu" && name != "sigma" {
            println!("  {name}: {value:.10}");
        }
    }

    GeneratedCode {
        function_code,
        constants,
    }
}

#[cfg(feature = "symbolic")]
fn performance_comparison(normal: &Normal<f64>, generated: &GeneratedCode) {
    println!("\n=== Step 3: Performance Comparison ===");

    let test_points = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    // Generate the actual callable function
    let optimized_function = generated.to_callable_function();

    println!("Comparing evaluation methods:");

    for &x in &test_points {
        // Method 1: Standard library-based computation
        let standard_result: f64 = normal.log_density().at(&x);

        // Method 2: Generated optimized code (simulated)
        let optimized_result = generated.evaluate(x);

        // Method 3: Real generated function
        let real_optimized_result = optimized_function(x);

        println!(
            "  x={x:.1}: standard={standard_result:.8}, optimized={optimized_result:.8}, real={real_optimized_result:.8}"
        );

        // Verify all methods agree
        let diff1 = (standard_result - optimized_result).abs();
        let diff2 = (standard_result - real_optimized_result).abs();

        let max_diff = diff1.max(diff2);
        if max_diff < 1e-10 {
            println!("         ✓ All methods agree (max diff: {max_diff:.2e})");
        } else {
            println!("         ⚠ Methods disagree! Diffs: {diff1:.2e}, {diff2:.2e}");
        }
    }

    println!("\nPerformance characteristics:");
    println!("• Standard:  Recomputes σ², log(2πσ²), etc. each time");
    println!("• Optimized: All constants precomputed at code-gen time");
    println!("• Real Func: Actual executable closure with precomputed constants");
    println!("             Only 1 subtraction, 2 multiplications, 1 FMA per evaluation!");

    // Demonstrate the real function in action
    println!("\n=== Real Function Demonstration ===");
    println!("Generated function can be called like any normal function:");
    let demo_x = 1.5;
    let result = optimized_function(demo_x);
    println!("optimized_function({demo_x}) = {result:.10}");

    // Performance test
    println!("\n=== Performance Timing ===");
    let n_iterations = 1_000_000;
    let test_x = 1.5;

    // Time the standard method
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = normal.log_density().at(&test_x);
    }
    let standard_time = start.elapsed();

    // Time the generated function
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = optimized_function(test_x);
    }
    let optimized_time = start.elapsed();

    println!("Timing {n_iterations} iterations:");
    println!(
        "  Standard:  {:?} ({:.2} ns/call)",
        standard_time,
        standard_time.as_nanos() as f64 / f64::from(n_iterations)
    );
    println!(
        "  Generated: {:?} ({:.2} ns/call)",
        optimized_time,
        optimized_time.as_nanos() as f64 / f64::from(n_iterations)
    );

    let speedup_vs_standard = standard_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("  Speedup vs standard: {speedup_vs_standard:.2}x");
}

#[cfg(feature = "symbolic")]
fn demonstrate_general_approach() {
    println!("\n=== Step 4: General Approach ===");

    println!("This symbolic optimization approach could be applied to:");
    println!("• Any exponential family distribution with known parameters");
    println!("• Composite distributions (mixtures, transformations)");
    println!("• IID distributions with known sample sizes");
    println!("• Complex models with known structure");

    println!("\nAdvantages over traditional approaches:");
    println!("• Zero runtime overhead (all precomputed)");
    println!("• No memory overhead for intermediate calculations");
    println!("• Perfect branch prediction");
    println!("• Compiler optimizations apply fully");
    println!("• Can inline completely");

    println!("\nImplementation strategy:");
    println!("1. Use symbolic math to represent log-density");
    println!("2. Simplify expression algebraically");
    println!("3. Extract constant subexpressions");
    println!("4. Generate optimized Rust code");
    println!("5. Compile specialized functions just-in-time");

    // Example: Different distributions
    println!("\n=== Examples with Other Distributions ===");

    // Standard Normal case
    println!("Standard Normal(0,1) → f(x) = -0.5 * (x² + LOG_2PI)");
    println!("  Generated: const C = -0.5 * ln(2π); return C - 0.5 * x * x;");

    // Poisson case
    println!("Poisson(λ=3) → f(k) = k*ln(3) - 3 - ln(k!)");
    println!("  Generated: const C = ln(3); const D = 3.0; return k * C - D - log_factorial(k);");

    // Exponential case
    println!("Exponential(λ=2) → f(x) = ln(2) - 2x");
    println!("  Generated: const C = ln(2); const D = 2.0; return C - D * x;");

    println!("\n=== Integration with Current System ===");
    println!("This extends the current exponential family framework:");
    println!("• Add SymbolicOptimizer trait");
    println!("• Generate code for fixed-parameter distributions");
    println!("• Use procedural macros for compile-time generation");
    println!("• JIT compilation with Cranelift for runtime generation");

    // Mock example of how it could integrate
    println!("\nExample integration:");
    println!(
        r"
// Current API
let normal = Normal::new(2.0, 1.5);
let log_density = normal.log_density().at(&x);

// Extended API with symbolic optimization  
let optimized_fn = normal.symbolic_optimize();
let result = optimized_fn.call(&x);  // Much faster!

// Or JIT compilation (future)
let jit_fn = normal.compile_jit()?;
let result = jit_fn.call(&x);  // Native machine code speed!
"
    );

    println!("\n=== Key Insights ===");
    println!("✅ Symbolic representation enables compile-time optimization");
    println!("✅ Generated code has minimal runtime computational overhead");
    println!("✅ Approach scales to complex distributions and models");
    println!("✅ Provides foundation for JIT compilation with Cranelift");
    println!("✅ Maintains numerical accuracy while maximizing performance");
}

#[cfg(not(feature = "symbolic"))]
fn main() {
    println!("This example requires the 'symbolic' feature.");
    println!("Run with: cargo run --example symbolic_code_generation --features symbolic");
    println!("\nThis example demonstrates symbolic optimization for log-density computation.");
    println!("The approach provides extremely efficient specialized functions with");
    println!("all constants precomputed at generation time.");
}
