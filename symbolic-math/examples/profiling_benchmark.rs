//! Comprehensive Profiling for Symbolic Math Operations
//!
//! This example profiles three key aspects of the symbolic-math crate:
//! 1. Expression simplification performance
//! 2. JIT compilation (codegen) performance  
//! 3. Execution performance (interpreted vs JIT)
//!
//! Run with: cargo run --example profiling_benchmark --features="jit,optimization"

use std::collections::HashMap;
use std::time::{Duration, Instant};
use symbolic_math::{Expr, builders};

#[cfg(feature = "jit")]
use symbolic_math::{CustomSymbolicLogDensity, GeneralJITCompiler};

#[cfg(feature = "optimization")]
use symbolic_math::EgglogOptimize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Symbolic Math Profiling Benchmark");
    println!("=====================================\n");

    // Test expressions of varying complexity
    let test_cases = create_test_expressions();

    for (name, expr) in &test_cases {
        println!("📊 Profiling: {}", name);
        println!("Expression: {}", expr);
        println!("Complexity: {} operations", expr.complexity());
        println!("Variables: {:?}", expr.variables());
        println!();

        profile_expression(name, expr)?;
        println!("{}", "─".repeat(80));
    }

    // Summary comparison
    println!("\n📈 Performance Summary");
    run_comparative_benchmark()?;

    Ok(())
}

fn create_test_expressions() -> Vec<(&'static str, Expr)> {
    vec![
        // Simple expressions
        (
            "Linear",
            Expr::add(
                Expr::mul(Expr::constant(2.0), Expr::variable("x")),
                Expr::constant(3.0),
            ),
        ),
        (
            "Quadratic",
            Expr::add(
                Expr::add(
                    Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
                    Expr::mul(Expr::constant(2.0), Expr::variable("x")),
                ),
                Expr::constant(1.0),
            ),
        ),
        // Normal distribution log-PDF
        ("Normal Log-PDF", builders::normal_log_pdf("x", 0.0, 1.0)),
        // Complex polynomial
        (
            "Polynomial Degree 5",
            builders::polynomial("x", &[1.0, -2.0, 3.0, -1.0, 0.5, 0.1]),
        ),
        // Trigonometric expression
        (
            "Trigonometric",
            Expr::add(
                Expr::sin(Expr::mul(Expr::constant(2.0), Expr::variable("x"))),
                Expr::cos(Expr::variable("x")),
            ),
        ),
        // Complex nested expression
        (
            "Nested Complex",
            Expr::exp(Expr::neg(Expr::div(
                Expr::pow(
                    Expr::sub(Expr::variable("x"), Expr::constant(2.0)),
                    Expr::constant(2.0),
                ),
                Expr::mul(
                    Expr::constant(2.0),
                    Expr::pow(Expr::constant(1.5), Expr::constant(2.0)),
                ),
            ))),
        ),
        // Expression with redundancy (good for simplification testing)
        (
            "Redundant Expression",
            Expr::add(
                Expr::add(
                    Expr::mul(Expr::variable("x"), Expr::constant(0.0)),
                    Expr::mul(Expr::variable("x"), Expr::constant(1.0)),
                ),
                Expr::add(
                    Expr::constant(0.0),
                    Expr::pow(Expr::variable("x"), Expr::constant(1.0)),
                ),
            ),
        ),
    ]
}

fn profile_expression(name: &str, expr: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Profile simplification
    profile_simplification(expr)?;

    // 2. Profile JIT compilation
    #[cfg(feature = "jit")]
    profile_jit_compilation(expr)?;

    // 3. Profile execution performance
    profile_execution_performance(expr)?;

    println!();
    Ok(())
}

fn profile_simplification(expr: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    println!("  🧮 Simplification Profiling:");

    // Basic simplification
    let start = Instant::now();
    let simplified_basic = expr.clone().simplify();
    let basic_time = start.elapsed();

    println!("    Basic simplify: {:?}", basic_time);
    println!(
        "    Complexity reduction: {} → {} operations",
        expr.complexity(),
        simplified_basic.complexity()
    );

    // Advanced optimization with egglog (if available)
    #[cfg(feature = "optimization")]
    {
        let start = Instant::now();
        let simplified_advanced = expr.optimize_with_egglog();
        let advanced_time = start.elapsed();

        match simplified_advanced {
            Ok(optimized) => {
                println!("    Egglog optimize: {:?}", advanced_time);
                println!(
                    "    Advanced complexity reduction: {} → {} operations",
                    expr.complexity(),
                    optimized.complexity()
                );

                // Compare results
                let test_vars = create_test_variables(&expr.variables());
                let original_result = expr.evaluate(&test_vars);
                let basic_result = simplified_basic.evaluate(&test_vars);
                let advanced_result = optimized.evaluate(&test_vars);

                if let (Ok(orig), Ok(basic), Ok(adv)) =
                    (original_result, basic_result, advanced_result)
                {
                    println!("    Correctness check:");
                    println!("      Original: {:.6}", orig);
                    println!(
                        "      Basic:    {:.6} (diff: {:.2e})",
                        basic,
                        (orig - basic).abs()
                    );
                    println!(
                        "      Advanced: {:.6} (diff: {:.2e})",
                        adv,
                        (orig - adv).abs()
                    );
                }
            }
            Err(e) => {
                println!("    Egglog optimize: Failed ({:?}) - {}", advanced_time, e);
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("    Egglog optimize: Not available (feature disabled)");
    }

    Ok(())
}

#[cfg(feature = "jit")]
fn profile_jit_compilation(expr: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    println!("  ⚡ JIT Compilation Profiling:");

    let variables = expr.variables();
    if variables.is_empty() {
        println!("    JIT compilation: Skipped (no variables)");
        return Ok(());
    }

    // Single variable case
    if variables.len() == 1 {
        let start = Instant::now();
        let compiler = GeneralJITCompiler::new()?;
        let compilation_time = start.elapsed();

        let start = Instant::now();
        let jit_function = compiler.compile_expression(expr, &variables, &[], &HashMap::new())?;
        let total_time = start.elapsed();

        println!("    Compiler creation: {:?}", compilation_time);
        println!("    Expression compilation: {:?}", total_time);
        println!(
            "    Code size: {} bytes",
            jit_function.compilation_stats.code_size_bytes
        );
        println!(
            "    CLIF instructions: {}",
            jit_function.compilation_stats.clif_instructions
        );
        println!(
            "    Estimated speedup: {:.1}x",
            jit_function.compilation_stats.estimated_speedup
        );

        return Ok(());
    }

    // Multi-variable case
    let start = Instant::now();
    let compiler = GeneralJITCompiler::new()?;
    let compilation_time = start.elapsed();

    let start = Instant::now();
    let jit_function = compiler.compile_expression(
        expr,
        &variables[..1], // First variable as data
        &variables[1..], // Rest as parameters
        &HashMap::new(),
    )?;
    let total_time = start.elapsed();

    println!("    Compiler creation: {:?}", compilation_time);
    println!("    Expression compilation: {:?}", total_time);
    println!(
        "    Code size: {} bytes",
        jit_function.compilation_stats.code_size_bytes
    );
    println!(
        "    CLIF instructions: {}",
        jit_function.compilation_stats.clif_instructions
    );

    Ok(())
}

#[cfg(not(feature = "jit"))]
fn profile_jit_compilation(_expr: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    println!("  ⚡ JIT Compilation Profiling:");
    println!("    JIT compilation: Not available (feature disabled)");
    Ok(())
}

fn profile_execution_performance(expr: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    println!("  🏃 Execution Performance Profiling:");

    let variables = expr.variables();
    if variables.is_empty() {
        println!("    Execution profiling: Skipped (no variables)");
        return Ok(());
    }

    let test_values = vec![0.0, 0.5, 1.0, 1.5, 2.0, -1.0, -0.5, 10.0, -10.0, 0.1];
    let iterations = 10000;

    // Interpreted execution
    let start = Instant::now();
    let mut interpreted_results = Vec::new();
    for _ in 0..iterations {
        for &val in &test_values {
            let mut vars = HashMap::new();
            vars.insert(variables[0].clone(), val);
            if let Ok(result) = expr.evaluate(&vars) {
                interpreted_results.push(result);
            }
        }
    }
    let interpreted_time = start.elapsed();

    println!(
        "    Interpreted execution ({} calls): {:?}",
        iterations * test_values.len(),
        interpreted_time
    );
    println!(
        "    Avg per call: {:.2} ns",
        interpreted_time.as_nanos() as f64 / (iterations * test_values.len()) as f64
    );

    // JIT execution (if available)
    #[cfg(feature = "jit")]
    {
        if variables.len() == 1 {
            let compiler = GeneralJITCompiler::new()?;
            if let Ok(jit_function) =
                compiler.compile_expression(expr, &variables, &[], &HashMap::new())
            {
                let start = Instant::now();
                let mut jit_results = Vec::new();
                for _ in 0..iterations {
                    for &val in &test_values {
                        let result = jit_function.call_single(val);
                        jit_results.push(result);
                    }
                }
                let jit_time = start.elapsed();

                println!(
                    "    JIT execution ({} calls): {:?}",
                    iterations * test_values.len(),
                    jit_time
                );
                println!(
                    "    Avg per call: {:.2} ns",
                    jit_time.as_nanos() as f64 / (iterations * test_values.len()) as f64
                );

                if interpreted_time.as_nanos() > 0 {
                    let speedup = interpreted_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
                    println!("    Actual speedup: {:.2}x", speedup);
                }

                // Verify correctness
                let max_diff = interpreted_results
                    .iter()
                    .zip(jit_results.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                println!("    Max difference: {:.2e}", max_diff);
            }
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        println!("    JIT execution: Not available (feature disabled)");
    }

    Ok(())
}

fn create_test_variables(var_names: &[String]) -> HashMap<String, f64> {
    let mut vars = HashMap::new();
    for (i, name) in var_names.iter().enumerate() {
        vars.insert(name.clone(), 1.0 + i as f64 * 0.5);
    }
    vars
}

fn run_comparative_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running comparative benchmark across expression types...\n");

    let expressions = vec![
        (
            "Simple",
            Expr::add(Expr::variable("x"), Expr::constant(1.0)),
        ),
        (
            "Quadratic",
            Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        ),
        ("Exponential", Expr::exp(Expr::variable("x"))),
        ("Logarithmic", Expr::ln(Expr::variable("x"))),
        ("Trigonometric", Expr::sin(Expr::variable("x"))),
        (
            "Complex",
            Expr::exp(Expr::neg(Expr::pow(
                Expr::variable("x"),
                Expr::constant(2.0),
            ))),
        ),
    ];

    println!(
        "| Expression Type | Complexity | Simplify (μs) | JIT Compile (μs) | Interpreted (ns/call) | JIT (ns/call) | Speedup |"
    );
    println!(
        "|----------------|------------|---------------|------------------|-----------------------|---------------|---------|"
    );

    for (name, expr) in expressions {
        let complexity = expr.complexity();

        // Measure simplification
        let start = Instant::now();
        let _simplified = expr.clone().simplify();
        let simplify_time = start.elapsed().as_micros();

        // Measure interpreted execution
        let start = Instant::now();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        for _ in 0..1000 {
            let _ = expr.evaluate(&vars);
        }
        let interpreted_ns = start.elapsed().as_nanos() / 1000;

        // Measure JIT compilation and execution
        #[cfg(feature = "jit")]
        let (jit_compile_time, jit_ns, speedup) = {
            if let Ok(compiler) = GeneralJITCompiler::new() {
                let start = Instant::now();
                if let Ok(jit_fn) =
                    compiler.compile_expression(&expr, &["x".to_string()], &[], &HashMap::new())
                {
                    let compile_time = start.elapsed().as_micros();

                    let start = Instant::now();
                    for _ in 0..1000 {
                        let _ = jit_fn.call_single(1.0);
                    }
                    let jit_time = start.elapsed().as_nanos() / 1000;
                    let speedup = interpreted_ns as f64 / jit_time as f64;

                    (compile_time, jit_time, speedup)
                } else {
                    (0, 0, 0.0)
                }
            } else {
                (0, 0, 0.0)
            }
        };

        #[cfg(not(feature = "jit"))]
        let (jit_compile_time, jit_ns, speedup) = (0, 0, 0.0);

        println!(
            "| {:14} | {:10} | {:13} | {:16} | {:21} | {:13} | {:7.1}x |",
            name, complexity, simplify_time, jit_compile_time, interpreted_ns, jit_ns, speedup
        );
    }

    Ok(())
}
