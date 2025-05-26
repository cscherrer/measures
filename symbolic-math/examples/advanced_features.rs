//! Advanced Features Example
//!
//! This example demonstrates the advanced features of the symbolic-math crate:
//! - Expression caching for improved performance
//! - Vectorized evaluation for batch processing
//! - Performance monitoring and optimization
//!
//! Run with: cargo run --example advanced_features

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::{Expr, clear_caches, get_cache_stats};

fn main() {
    println!("🚀 Advanced Features Demonstration");
    println!("==================================\n");

    test_expression_caching();
    test_vectorized_evaluation();
    test_batch_processing_performance();
    test_grid_evaluation();
}

fn test_expression_caching() {
    println!("💾 Expression Caching");
    println!("--------------------");

    // Clear caches to start fresh
    clear_caches();

    // Create a complex expression that benefits from caching
    let expr = create_complex_expression();
    println!("Expression: {}", expr);

    // First simplification (cache miss)
    let start = Instant::now();
    let simplified1 = expr.clone().simplify_cached();
    let first_time = start.elapsed();

    // Second simplification (cache hit)
    let start = Instant::now();
    let simplified2 = expr.clone().simplify_cached();
    let second_time = start.elapsed();

    println!(
        "First simplification:  {:6.2} μs (cache miss)",
        first_time.as_micros()
    );
    println!(
        "Second simplification: {:6.2} μs (cache hit)",
        second_time.as_micros()
    );

    let speedup = first_time.as_nanos() as f64 / second_time.as_nanos() as f64;
    println!("Cache speedup: {:.2}x", speedup);

    // Test evaluation caching
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    vars.insert("y".to_string(), 3.0);

    // First evaluation (cache miss)
    let start = Instant::now();
    let result1 = simplified1.evaluate_cached(&vars).unwrap();
    let first_eval_time = start.elapsed();

    // Second evaluation (cache hit)
    let start = Instant::now();
    let result2 = simplified1.evaluate_cached(&vars).unwrap();
    let second_eval_time = start.elapsed();

    println!(
        "First evaluation:  {:6.2} μs (cache miss)",
        first_eval_time.as_micros()
    );
    println!(
        "Second evaluation: {:6.2} μs (cache hit)",
        second_eval_time.as_micros()
    );

    assert_eq!(result1, result2);

    // Show cache statistics
    let stats = get_cache_stats();
    println!("Cache Statistics:");
    println!(
        "  Simplification hit rate: {:.1}%",
        stats.simplification_hit_rate() * 100.0
    );
    println!(
        "  Evaluation hit rate:     {:.1}%",
        stats.evaluation_hit_rate() * 100.0
    );
    println!();
}

fn test_vectorized_evaluation() {
    println!("📊 Vectorized Evaluation");
    println!("------------------------");

    let expr = Expr::add(
        Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        Expr::mul(Expr::constant(2.0), Expr::variable("x")),
    );

    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Single evaluations
    let start = Instant::now();
    let mut single_results = Vec::new();
    let mut vars = HashMap::new();
    for &x in &values {
        vars.insert("x".to_string(), x);
        single_results.push(expr.evaluate(&vars).unwrap());
    }
    let single_time = start.elapsed();

    // Batch evaluation
    let start = Instant::now();
    let batch_results = expr.evaluate_batch("x", &values).unwrap();
    let batch_time = start.elapsed();

    println!("Single evaluations: {:6.2} μs", single_time.as_micros());
    println!("Batch evaluation:   {:6.2} μs", batch_time.as_micros());

    let speedup = single_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
    println!("Batch speedup: {:.2}x", speedup);

    // Verify results are the same
    assert_eq!(single_results, batch_results);
    println!("Results: {:?}", batch_results);
    println!();
}

fn test_batch_processing_performance() {
    println!("⚡ Batch Processing Performance");
    println!("------------------------------");

    let expr = create_polynomial_expression();
    let num_points = 1000;
    let values: Vec<f64> = (0..num_points).map(|i| i as f64 * 0.01).collect();

    // Individual evaluations
    let start = Instant::now();
    let mut individual_results = Vec::new();
    let mut vars = HashMap::new();
    for &x in &values {
        vars.insert("x".to_string(), x);
        individual_results.push(expr.evaluate(&vars).unwrap());
    }
    let individual_time = start.elapsed();

    // Batch evaluation
    let start = Instant::now();
    let batch_results = expr.evaluate_batch("x", &values).unwrap();
    let batch_time = start.elapsed();

    println!(
        "Individual evaluations ({} points): {:6.2} ms",
        num_points,
        individual_time.as_millis()
    );
    println!(
        "Batch evaluation ({} points):       {:6.2} ms",
        num_points,
        batch_time.as_millis()
    );

    let speedup = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
    println!("Performance improvement: {:.2}x", speedup);

    // Verify results are the same
    assert_eq!(individual_results, batch_results);
    println!("✓ Results verified identical");
    println!();
}

fn test_grid_evaluation() {
    println!("🗺️  Grid Evaluation");
    println!("------------------");

    // Create a 2D function: x^2 + y^2
    let expr = Expr::add(
        Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        Expr::pow(Expr::variable("y"), Expr::constant(2.0)),
    );

    let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y_values = vec![-1.0, 0.0, 1.0];

    let start = Instant::now();
    let grid_results = expr.evaluate_grid("x", &x_values, "y", &y_values).unwrap();
    let grid_time = start.elapsed();

    println!(
        "Grid evaluation ({}x{} points): {:6.2} μs",
        x_values.len(),
        y_values.len(),
        grid_time.as_micros()
    );
    println!("Grid results (x^2 + y^2):");

    // Print header with x values
    print!("     y\\x  ");
    for &x in &x_values {
        print!("{:6.1} ", x);
    }
    println!();

    // Print each row with y value and results
    for (i, row) in grid_results.iter().enumerate() {
        print!("y={:6.1}  ", y_values[i]);
        for &value in row {
            print!("{:6.1} ", value);
        }
        println!();
    }
    println!();
}

// Helper functions to create test expressions

fn create_complex_expression() -> Expr {
    // (x^2 + y^2) * ln(x + 1) + exp(-x/2)
    let x = Expr::variable("x");
    let y = Expr::variable("y");

    let term1 = Expr::mul(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::pow(y, Expr::constant(2.0)),
        ),
        Expr::ln(Expr::add(x.clone(), Expr::constant(1.0))),
    );

    let term2 = Expr::exp(Expr::neg(Expr::div(x, Expr::constant(2.0))));

    Expr::add(term1, term2)
}

fn create_polynomial_expression() -> Expr {
    // 3x^4 - 2x^3 + x^2 - 5x + 7
    let x = Expr::variable("x");

    Expr::add(
        Expr::add(
            Expr::add(
                Expr::add(
                    Expr::mul(
                        Expr::constant(3.0),
                        Expr::pow(x.clone(), Expr::constant(4.0)),
                    ),
                    Expr::neg(Expr::mul(
                        Expr::constant(2.0),
                        Expr::pow(x.clone(), Expr::constant(3.0)),
                    )),
                ),
                Expr::pow(x.clone(), Expr::constant(2.0)),
            ),
            Expr::neg(Expr::mul(Expr::constant(5.0), x)),
        ),
        Expr::constant(7.0),
    )
}
