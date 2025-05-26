//! Simple Profiling Example for Symbolic Math
//!
//! This example demonstrates basic profiling of symbolic math operations
//! without requiring optional features.

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::{Expr, builders};

fn main() {
    println!("🔬 Simple Symbolic Math Profiling");
    println!("==================================\n");

    test_expression_creation();
    test_simplification();
    test_evaluation();
    test_builders();
}

fn test_expression_creation() {
    println!("📊 Expression Creation Performance");
    println!("----------------------------------");

    let iterations = 50_000;

    // Test simple expression creation
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = Expr::add(Expr::variable("x"), Expr::constant(f64::from(i)));
    }
    let simple_time = start.elapsed();

    // Test complex nested expression creation
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = Expr::exp(Expr::neg(Expr::pow(
            Expr::sub(Expr::variable("x"), Expr::constant(f64::from(i))),
            Expr::constant(2.0),
        )));
    }
    let complex_time = start.elapsed();

    println!(
        "Simple expressions: {:?} ({:.2} ns/expr)",
        simple_time,
        simple_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "Complex expressions: {:?} ({:.2} ns/expr)",
        complex_time,
        complex_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!();
}

fn test_simplification() {
    println!("🧮 Simplification Performance");
    println!("-----------------------------");

    let test_cases = vec![
        ("x + 0", Expr::add(Expr::variable("x"), Expr::constant(0.0))),
        ("x * 1", Expr::mul(Expr::variable("x"), Expr::constant(1.0))),
        ("x * 0", Expr::mul(Expr::variable("x"), Expr::constant(0.0))),
        ("x^1", Expr::pow(Expr::variable("x"), Expr::constant(1.0))),
        ("x^0", Expr::pow(Expr::variable("x"), Expr::constant(0.0))),
        ("ln(exp(x))", Expr::ln(Expr::exp(Expr::variable("x")))),
        ("exp(ln(x))", Expr::exp(Expr::ln(Expr::variable("x")))),
        ("2 + 3", Expr::add(Expr::constant(2.0), Expr::constant(3.0))),
        ("Complex", create_redundant_expression()),
    ];

    for (name, expr) in test_cases {
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
            "{:12} | {:2} → {:2} ops ({:5.1}% reduction) | {:6.2} μs | Result: {}",
            name,
            original_complexity,
            new_complexity,
            reduction,
            simplify_time.as_micros(),
            simplified
        );
    }
    println!();
}

fn test_evaluation() {
    println!("🏃 Evaluation Performance");
    println!("-------------------------");

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

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    let iterations = 10_000;

    for (name, expr) in expressions {
        // Test evaluation performance
        let start = Instant::now();
        let mut results = Vec::new();
        for _ in 0..iterations {
            if let Ok(result) = expr.evaluate(&vars) {
                results.push(result);
            }
        }
        let eval_time = start.elapsed();

        // Test with different values
        let test_values = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let start = Instant::now();
        for &val in &test_values {
            vars.insert("x".to_string(), val);
            for _ in 0..1000 {
                let _ = expr.evaluate(&vars);
            }
        }
        let multi_eval_time = start.elapsed();

        println!(
            "{:13} | {:6.2} ns/call | Sample result: {:.6} | Multi-value: {:6.2} μs",
            name,
            eval_time.as_nanos() as f64 / f64::from(iterations),
            results.first().unwrap_or(&0.0),
            multi_eval_time.as_micros()
        );
    }
    println!();
}

fn test_builders() {
    println!("🏗️  Builder Performance");
    println!("----------------------");

    let iterations = 10_000;

    // Test normal log-PDF builder
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = builders::normal_log_pdf("x", f64::from(i) % 10.0, 1.0);
    }
    let normal_time = start.elapsed();

    // Test polynomial builder
    let start = Instant::now();
    for _ in 0..iterations {
        let _expr = builders::polynomial("x", &[1.0, 2.0, 3.0, 4.0]);
    }
    let poly_time = start.elapsed();

    // Test gaussian kernel builder
    let start = Instant::now();
    for i in 0..iterations {
        let _expr = builders::gaussian_kernel("x", f64::from(i) % 5.0, 1.0);
    }
    let gaussian_time = start.elapsed();

    println!(
        "Normal log-PDF: {:?} ({:.2} ns/expr)",
        normal_time,
        normal_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "Polynomial:     {:?} ({:.2} ns/expr)",
        poly_time,
        poly_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "Gaussian kernel: {:?} ({:.2} ns/expr)",
        gaussian_time,
        gaussian_time.as_nanos() as f64 / f64::from(iterations)
    );

    // Test evaluation of built expressions
    let normal_expr = builders::normal_log_pdf("x", 0.0, 1.0);
    let poly_expr = builders::polynomial("x", &[1.0, -2.0, 1.0]);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = normal_expr.evaluate(&vars);
    }
    let normal_eval_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = poly_expr.evaluate(&vars);
    }
    let poly_eval_time = start.elapsed();

    println!("\nEvaluation Performance:");
    println!(
        "Normal log-PDF eval: {:6.2} ns/call",
        normal_eval_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "Polynomial eval:     {:6.2} ns/call",
        poly_eval_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!();
}

fn create_redundant_expression() -> Expr {
    // Create an expression with redundancy that should simplify
    Expr::add(
        Expr::add(
            Expr::mul(Expr::variable("x"), Expr::constant(0.0)), // Should become 0
            Expr::mul(Expr::variable("x"), Expr::constant(1.0)), // Should become x
        ),
        Expr::add(
            Expr::constant(0.0),                                 // Should be removed
            Expr::pow(Expr::variable("x"), Expr::constant(1.0)), // Should become x
        ),
    )
    // Overall should simplify to 2x
}
