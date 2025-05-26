//! Comprehensive Benchmark Example
//!
//! This example demonstrates all performance features of the symbolic-math crate:
//! - Expression caching and memoization
//! - Vectorized evaluation for batch processing
//! - JIT compilation for maximum performance
//! - Advanced optimization with egglog
//! - Performance comparison across different approaches
//!
//! Run with: cargo run --example comprehensive_benchmark --features "jit optimization"

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::{Expr, clear_caches, get_cache_stats};

#[cfg(feature = "jit")]
use symbolic_math::{GeneralJITCompiler, JITSignature};

#[cfg(feature = "optimization")]
use symbolic_math::EgglogOptimize;

fn main() {
    println!("🚀 Comprehensive Performance Benchmark");
    println!("======================================\\n");

    // Test different expression types
    test_polynomial_performance();
    test_trigonometric_performance();
    test_complex_expression_performance();

    #[cfg(feature = "jit")]
    test_jit_performance();

    #[cfg(feature = "optimization")]
    test_optimization_performance();

    test_scaling_performance();
}

fn test_polynomial_performance() {
    println!("📈 Polynomial Expression Performance");
    println!("------------------------------------");

    // Create polynomial: 3x^4 - 2x^3 + x^2 - 5x + 7
    let expr = create_polynomial();
    println!("Expression: {}", expr);

    let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();

    // Test different evaluation methods
    benchmark_evaluation_methods(&expr, "x", &values);
    println!();
}

fn test_trigonometric_performance() {
    println!("🌊 Trigonometric Expression Performance");
    println!("---------------------------------------");

    // Create trig expression: sin(x)^2 + cos(x)^2 (should simplify to 1)
    let expr = Expr::add(
        Expr::pow(Expr::sin(Expr::variable("x")), Expr::constant(2.0)),
        Expr::pow(Expr::cos(Expr::variable("x")), Expr::constant(2.0)),
    );
    println!("Expression: {}", expr);

    let values: Vec<f64> = (0..500).map(|i| i as f64 * 0.01).collect();

    // Test simplification impact
    let simplified = expr.clone().simplify();
    println!("Simplified: {}", simplified);

    benchmark_evaluation_methods(&expr, "x", &values);
    benchmark_evaluation_methods(&simplified, "x", &values);
    println!();
}

fn test_complex_expression_performance() {
    println!("🧮 Complex Expression Performance");
    println!("---------------------------------");

    // Create complex expression: (x^2 + y^2) * ln(x + 1) + exp(-x/2)
    let expr = create_complex_expression();
    println!("Expression: {}", expr);

    let x_values: Vec<f64> = (1..100).map(|i| i as f64 * 0.1).collect();
    let y_values: Vec<f64> = (1..100).map(|i| i as f64 * 0.1).collect();

    // Test grid evaluation performance
    let start = Instant::now();
    let grid_results = expr.evaluate_grid("x", &x_values, "y", &y_values).unwrap();
    let grid_time = start.elapsed();

    println!(
        "Grid evaluation ({}x{} = {} points): {:.2} ms",
        x_values.len(),
        y_values.len(),
        x_values.len() * y_values.len(),
        grid_time.as_millis()
    );

    // Verify grid results
    let total_points = grid_results.iter().map(|row| row.len()).sum::<usize>();
    println!("✓ Grid evaluation completed: {} results", total_points);
    println!();
}

#[cfg(feature = "jit")]
fn test_jit_performance() {
    println!("⚡ JIT Compilation Performance");
    println!("------------------------------");

    let expr = create_polynomial();
    let values: Vec<f64> = (0..10000).map(|i| i as f64 * 0.001).collect();

    // Interpreted evaluation
    let start = Instant::now();
    let interpreted_results = expr.evaluate_batch("x", &values).unwrap();
    let interpreted_time = start.elapsed();

    // JIT compilation
    let start = Instant::now();
    let compiler = GeneralJITCompiler::new().unwrap();
    let jit_func = compiler
        .compile_expression(
            &expr,
            &["x".to_string()], // data variables
            &[],                // parameter variables
            &HashMap::new(),    // constants
        )
        .unwrap();
    let compilation_time = start.elapsed();

    // JIT evaluation
    let start = Instant::now();
    let jit_results: Vec<f64> = values.iter().map(|&x| jit_func.call_single(x)).collect();
    let jit_time = start.elapsed();

    println!(
        "Interpreted evaluation ({} points): {:.2} ms",
        values.len(),
        interpreted_time.as_millis()
    );
    println!(
        "JIT compilation time:                {:.2} ms",
        compilation_time.as_millis()
    );
    println!(
        "JIT evaluation ({} points):         {:.2} ms",
        values.len(),
        jit_time.as_millis()
    );

    let speedup = interpreted_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    println!("JIT speedup: {:.1}x", speedup);

    // Verify results are close (allowing for floating point differences)
    let max_diff = interpreted_results
        .iter()
        .zip(jit_results.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("✓ Maximum difference: {:.2e}", max_diff);
    println!();
}

#[cfg(feature = "optimization")]
fn test_optimization_performance() {
    println!("🔧 Advanced Optimization Performance");
    println!("------------------------------------");

    // Create expression with redundancy: x + x + x - x
    let expr = Expr::sub(
        Expr::add(
            Expr::add(Expr::variable("x"), Expr::variable("x")),
            Expr::variable("x"),
        ),
        Expr::variable("x"),
    );

    println!("Original: {}", expr);
    println!("Complexity: {} operations", expr.complexity());

    // Basic simplification
    let start = Instant::now();
    let basic_simplified = expr.clone().simplify();
    let basic_time = start.elapsed();

    println!("Basic simplified: {}", basic_simplified);
    println!(
        "Basic complexity: {} operations",
        basic_simplified.complexity()
    );
    println!(
        "Basic simplification time: {:.2} μs",
        basic_time.as_micros()
    );

    // Advanced optimization
    let start = Instant::now();
    let optimized = expr
        .clone()
        .optimize_with_egglog()
        .unwrap_or_else(|_| expr.clone());
    let opt_time = start.elapsed();

    println!("Egglog optimized: {}", optimized);
    println!(
        "Optimized complexity: {} operations",
        optimized.complexity()
    );
    println!("Egglog optimization time: {:.2} ms", opt_time.as_millis());

    let complexity_reduction =
        (expr.complexity() - optimized.complexity()) as f64 / expr.complexity() as f64 * 100.0;
    println!("Complexity reduction: {:.1}%", complexity_reduction);
    println!();
}

fn test_scaling_performance() {
    println!("📊 Scaling Performance Analysis");
    println!("-------------------------------");

    let expr = create_polynomial();
    let sizes = vec![100, 500, 1000, 5000, 10000];

    println!("Batch size | Interpreted | Cached | Speedup");
    println!("-----------|-------------|--------|--------");

    for &size in &sizes {
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();

        // Clear cache for fair comparison
        clear_caches();

        // Interpreted evaluation
        let start = Instant::now();
        let _results1 = expr.evaluate_batch("x", &values).unwrap();
        let interpreted_time = start.elapsed();

        // Cached evaluation (second run)
        let start = Instant::now();
        let _results2 = expr.evaluate_batch("x", &values).unwrap();
        let cached_time = start.elapsed();

        let speedup = interpreted_time.as_nanos() as f64 / cached_time.as_nanos() as f64;

        println!(
            "{:10} | {:9.2} ms | {:6.2} ms | {:6.1}x",
            size,
            interpreted_time.as_millis(),
            cached_time.as_millis(),
            speedup
        );
    }

    let stats = get_cache_stats();
    println!("\\nFinal cache statistics:");
    println!(
        "  Evaluation hit rate: {:.1}%",
        stats.evaluation_hit_rate() * 100.0
    );
    println!();
}

fn benchmark_evaluation_methods(expr: &Expr, var_name: &str, values: &[f64]) {
    // Individual evaluations
    let start = Instant::now();
    let mut individual_results = Vec::new();
    let mut vars = HashMap::new();
    for &x in values {
        vars.insert(var_name.to_string(), x);
        individual_results.push(expr.evaluate(&vars).unwrap());
    }
    let individual_time = start.elapsed();

    // Batch evaluation
    let start = Instant::now();
    let batch_results = expr.evaluate_batch(var_name, values).unwrap();
    let batch_time = start.elapsed();

    // Cached evaluation
    clear_caches();
    let start = Instant::now();
    for &x in values {
        let mut vars = HashMap::new();
        vars.insert(var_name.to_string(), x);
        let _ = expr.evaluate_cached(&vars).unwrap();
    }
    let cached_time = start.elapsed();

    println!(
        "Individual evaluations ({} points): {:.2} ms",
        values.len(),
        individual_time.as_millis()
    );
    println!(
        "Batch evaluation ({} points):       {:.2} ms",
        values.len(),
        batch_time.as_millis()
    );
    println!(
        "Cached evaluation ({} points):      {:.2} ms",
        values.len(),
        cached_time.as_millis()
    );

    let batch_speedup = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
    let cache_speedup = individual_time.as_nanos() as f64 / cached_time.as_nanos() as f64;

    println!("Batch speedup: {:.1}x", batch_speedup);
    println!("Cache speedup: {:.1}x", cache_speedup);

    // Verify results
    assert_eq!(individual_results, batch_results);
}

// Helper functions

fn create_polynomial() -> Expr {
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
