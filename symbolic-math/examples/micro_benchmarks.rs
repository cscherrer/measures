//! Micro-benchmarks for Symbolic Math Operations
//!
//! This example provides focused micro-benchmarks for:
//! - Expression creation and manipulation
//! - Simplification algorithms
//! - JIT compilation overhead
//! - Execution performance comparison
//!
//! Run with: cargo run --example `micro_benchmarks` --features="jit,optimization"

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::{Expr, builders};

#[cfg(feature = "jit")]
use symbolic_math::GeneralJITCompiler;

#[cfg(feature = "optimization")]
use symbolic_math::EgglogOptimize;

fn main() {
    println!("🔬 Symbolic Math Micro-Benchmarks");
    println!("==================================\n");

    benchmark_expression_creation();
    benchmark_simplification();
    benchmark_evaluation();

    #[cfg(feature = "jit")]
    benchmark_jit_compilation();

    #[cfg(feature = "optimization")]
    benchmark_advanced_optimization();
}

fn benchmark_expression_creation() {
    println!("📊 Expression Creation Benchmarks");
    println!("----------------------------------");

    let iterations = 100_000;

    // Simple expression creation
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = Expr::add(Expr::variable("x"), Expr::constant(f64::from(i)));
    }
    let simple_time = start.elapsed();

    // Complex expression creation
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = Expr::exp(Expr::neg(Expr::div(
            Expr::pow(
                Expr::sub(Expr::variable("x"), Expr::constant(f64::from(i))),
                Expr::constant(2.0),
            ),
            Expr::constant(2.0),
        )));
    }
    let complex_time = start.elapsed();

    // Using builders
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = builders::normal_log_pdf("x", f64::from(i), 1.0);
    }
    let builder_time = start.elapsed();

    println!("Simple expressions ({iterations} iterations): {simple_time:?}");
    println!(
        "  Per expression: {:.2} ns",
        simple_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!("Complex expressions ({iterations} iterations): {complex_time:?}");
    println!(
        "  Per expression: {:.2} ns",
        complex_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!("Builder expressions ({iterations} iterations): {builder_time:?}");
    println!(
        "  Per expression: {:.2} ns",
        builder_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!();
}

fn benchmark_simplification() {
    println!("🧮 Simplification Benchmarks");
    println!("-----------------------------");

    let test_expressions = vec![
        ("x + 0", Expr::add(Expr::variable("x"), Expr::constant(0.0))),
        ("x * 1", Expr::mul(Expr::variable("x"), Expr::constant(1.0))),
        ("x * 0", Expr::mul(Expr::variable("x"), Expr::constant(0.0))),
        ("x^1", Expr::pow(Expr::variable("x"), Expr::constant(1.0))),
        ("x^0", Expr::pow(Expr::variable("x"), Expr::constant(0.0))),
        ("ln(exp(x))", Expr::ln(Expr::exp(Expr::variable("x")))),
        ("exp(ln(x))", Expr::exp(Expr::ln(Expr::variable("x")))),
        ("x + x", Expr::add(Expr::variable("x"), Expr::variable("x"))),
        ("2 + 3", Expr::add(Expr::constant(2.0), Expr::constant(3.0))),
        ("Complex", create_complex_redundant_expression()),
    ];

    for (name, expr) in test_expressions {
        let original_complexity = expr.complexity();

        let start = Instant::now();
        let simplified = expr.simplify();
        let simplify_time = start.elapsed();

        let new_complexity = simplified.complexity();
        let reduction = if original_complexity > 0 {
            (original_complexity - new_complexity) as f64 / original_complexity as f64 * 100.0
        } else {
            0.0
        };

        println!(
            "{:15} | {:3} → {:3} ops ({:5.1}% reduction) | {:8.2} μs",
            name,
            original_complexity,
            new_complexity,
            reduction,
            simplify_time.as_micros()
        );
    }

    println!();
}

fn benchmark_evaluation() {
    println!("🏃 Evaluation Benchmarks");
    println!("------------------------");

    let expressions = vec![
        (
            "Linear",
            Expr::add(
                Expr::mul(Expr::constant(2.0), Expr::variable("x")),
                Expr::constant(3.0),
            ),
        ),
        (
            "Quadratic",
            Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        ),
        ("Exponential", Expr::exp(Expr::variable("x"))),
        ("Logarithmic", Expr::ln(Expr::variable("x"))),
        ("Trigonometric", Expr::sin(Expr::variable("x"))),
        ("Normal PDF", builders::normal_log_pdf("x", 0.0, 1.0)),
    ];

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    let iterations = 10_000;

    for (name, expr) in expressions {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = expr.evaluate(&vars);
        }
        let eval_time = start.elapsed();

        println!(
            "{:15} | {:8.2} ns/call | {:8.2} μs total",
            name,
            eval_time.as_nanos() as f64 / f64::from(iterations),
            eval_time.as_micros()
        );
    }

    println!();
}

#[cfg(feature = "jit")]
fn benchmark_jit_compilation() {
    println!("⚡ JIT Compilation Benchmarks");
    println!("-----------------------------");

    let expressions = vec![
        (
            "Linear",
            Expr::add(
                Expr::mul(Expr::constant(2.0), Expr::variable("x")),
                Expr::constant(3.0),
            ),
        ),
        (
            "Quadratic",
            Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        ),
        ("Exponential", Expr::exp(Expr::variable("x"))),
        ("Logarithmic", Expr::ln(Expr::variable("x"))),
        ("Trigonometric", Expr::sin(Expr::variable("x"))),
    ];

    for (name, expr) in expressions {
        // Measure compilation time
        let start = Instant::now();
        let compiler_result = GeneralJITCompiler::new();

        match compiler_result {
            Ok(compiler) => {
                let jit_result = compiler.compile_expression(
                    &expr,
                    &["x".to_string()], // Single data variable
                    &[],                // No parameters
                    &HashMap::new(),
                );
                let compile_time = start.elapsed();

                match jit_result {
                    Ok(jit_fn) => {
                        println!("JIT Function signature: {:?}", jit_fn.signature);

                        // Measure execution time based on signature
                        let iterations = 10_000;
                        let start = Instant::now();

                        // Use the appropriate call method based on signature
                        match jit_fn.signature {
                            symbolic_math::JITSignature::SingleInput => {
                                for _ in 0..iterations {
                                    let _ = jit_fn.call_single(1.0);
                                }
                            }
                            symbolic_math::JITSignature::DataAndParameters(0) => {
                                for _ in 0..iterations {
                                    let _ = jit_fn.call_data_params(1.0, &[]);
                                }
                            }
                            _ => {
                                println!("{name:15} | Unsupported signature for benchmarking");
                                continue;
                            }
                        }

                        let exec_time = start.elapsed();

                        // Compare with interpreted
                        let mut vars = HashMap::new();
                        vars.insert("x".to_string(), 1.0);
                        let start = Instant::now();
                        for _ in 0..iterations {
                            let _ = expr.evaluate(&vars);
                        }
                        let interpreted_time = start.elapsed();

                        let speedup = if exec_time.as_nanos() > 0 {
                            interpreted_time.as_nanos() as f64 / exec_time.as_nanos() as f64
                        } else {
                            0.0
                        };

                        println!(
                            "{:15} | Compile: {:8.2} μs | JIT: {:6.2} ns/call | Interpreted: {:6.2} ns/call | Speedup: {:5.2}x",
                            name,
                            compile_time.as_micros(),
                            exec_time.as_nanos() as f64 / f64::from(iterations),
                            interpreted_time.as_nanos() as f64 / f64::from(iterations),
                            speedup
                        );
                    }
                    Err(e) => {
                        println!("{name:15} | Compilation failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("{name:15} | Compiler creation failed: {e}");
            }
        }
    }

    println!();
}

#[cfg(feature = "optimization")]
fn benchmark_advanced_optimization() {
    println!("🔧 Advanced Optimization Benchmarks");
    println!("------------------------------------");

    let expressions = vec![
        ("Redundant", create_complex_redundant_expression()),
        (
            "Polynomial",
            builders::polynomial("x", &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        (
            "Trigonometric",
            Expr::add(
                Expr::pow(Expr::sin(Expr::variable("x")), Expr::constant(2.0)),
                Expr::pow(Expr::cos(Expr::variable("x")), Expr::constant(2.0)),
            ),
        ),
    ];

    for (name, expr) in expressions {
        let original_complexity = expr.complexity();

        // Basic simplification
        let start = Instant::now();
        let basic_simplified = expr.clone().simplify();
        let basic_time = start.elapsed();
        let basic_complexity = basic_simplified.complexity();

        // Advanced optimization
        let start = Instant::now();
        let advanced_result = expr.optimize_with_egglog();
        let advanced_time = start.elapsed();

        match advanced_result {
            Ok(advanced_simplified) => {
                let advanced_complexity = advanced_simplified.complexity();

                println!(
                    "{:15} | Original: {:3} ops | Basic: {:3} ops ({:6.2} μs) | Advanced: {:3} ops ({:8.2} μs)",
                    name,
                    original_complexity,
                    basic_complexity,
                    basic_time.as_micros(),
                    advanced_complexity,
                    advanced_time.as_micros()
                );
            }
            Err(e) => {
                println!(
                    "{:15} | Original: {:3} ops | Basic: {:3} ops ({:6.2} μs) | Advanced: Failed ({})",
                    name,
                    original_complexity,
                    basic_complexity,
                    basic_time.as_micros(),
                    e
                );
            }
        }
    }

    println!();
}

fn create_complex_redundant_expression() -> Expr {
    // Create an expression with lots of redundancy that should simplify well
    Expr::add(
        Expr::add(
            Expr::mul(Expr::variable("x"), Expr::constant(0.0)), // Should become 0
            Expr::mul(Expr::variable("x"), Expr::constant(1.0)), // Should become x
        ),
        Expr::add(
            Expr::add(Expr::constant(0.0), Expr::variable("x")), // Should become x
            Expr::pow(Expr::variable("x"), Expr::constant(1.0)), // Should become x
        ),
    )
    // Overall should simplify to 3x
}
