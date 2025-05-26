//! Overhead Reduction Benchmarks
//!
//! This benchmark compares the original evaluation methods with the new optimized methods
//! to measure the overhead reduction achieved.
//!
//! Run with: cargo bench --bench `overhead_reduction_benchmarks` --features "jit optimization"

use divan::Bencher;
use std::collections::HashMap;
use symbolic_math::Expr;

fn main() {
    divan::main();
}

// Test expressions for benchmarking
fn create_linear_expr() -> Expr {
    // 2x + 3
    Expr::add(
        Expr::mul(Expr::constant(2.0), Expr::variable("x")),
        Expr::constant(3.0),
    )
}

fn create_quadratic_expr() -> Expr {
    // x^2 + 2x + 1
    let x = Expr::variable("x");
    Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    )
}

fn create_complex_polynomial() -> Expr {
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

fn create_transcendental_expr() -> Expr {
    // sin(x) + cos(x) + exp(x)
    let x = Expr::variable("x");
    Expr::add(
        Expr::add(Expr::sin(x.clone()), Expr::cos(x.clone())),
        Expr::exp(x),
    )
}

// Baseline: Original evaluation method
#[divan::bench]
fn baseline_linear_original(bencher: Bencher) {
    let expr = create_linear_expr();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn baseline_quadratic_original(bencher: Bencher) {
    let expr = create_quadratic_expr();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn baseline_complex_polynomial_original(bencher: Bencher) {
    let expr = create_complex_polynomial();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Optimized: Single variable evaluation (no HashMap)
#[divan::bench]
fn optimized_linear_single_var(bencher: Bencher) {
    let expr = create_linear_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_single_var("x", 2.5).unwrap());
    });
}

#[divan::bench]
fn optimized_quadratic_single_var(bencher: Bencher) {
    let expr = create_quadratic_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_single_var("x", 2.5).unwrap());
    });
}

#[divan::bench]
fn optimized_complex_polynomial_single_var(bencher: Bencher) {
    let expr = create_complex_polynomial();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_single_var("x", 2.5).unwrap());
    });
}

// Ultra-optimized: Smart evaluation (pattern-specific)
#[divan::bench]
fn ultra_optimized_linear_smart(bencher: Bencher) {
    let expr = create_linear_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_smart("x", 2.5).unwrap());
    });
}

#[divan::bench]
fn ultra_optimized_quadratic_smart(bencher: Bencher) {
    let expr = create_quadratic_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_smart("x", 2.5).unwrap());
    });
}

#[divan::bench]
fn ultra_optimized_complex_polynomial_smart(bencher: Bencher) {
    let expr = create_complex_polynomial();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_smart("x", 2.5).unwrap());
    });
}

// Specialized: Linear coefficient extraction and evaluation
#[divan::bench]
fn specialized_linear_coefficients(bencher: Bencher) {
    let expr = create_linear_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_linear("x", 2.5).unwrap());
    });
}

// Specialized: Polynomial evaluation
#[divan::bench]
fn specialized_polynomial_evaluation(bencher: Bencher) {
    let expr = create_quadratic_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_polynomial("x", 2.5).unwrap());
    });
}

// Batch evaluation comparisons
#[divan::bench(args = [10, 100, 1000])]
fn batch_original(bencher: Bencher, size: usize) {
    let expr = create_quadratic_expr();
    let values: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();

    bencher.counter(size).bench(|| {
        divan::black_box(expr.evaluate_batch("x", &values).unwrap());
    });
}

#[divan::bench(args = [10, 100, 1000])]
fn batch_optimized(bencher: Bencher, size: usize) {
    let expr = create_quadratic_expr();
    let values: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();

    bencher.counter(size).bench(|| {
        divan::black_box(expr.evaluate_batch_optimized("x", &values).unwrap());
    });
}

// Constant folding optimization tests
#[divan::bench]
fn constant_folding_add_zero(bencher: Bencher) {
    let expr = Expr::add(Expr::variable("x"), Expr::constant(0.0));
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn constant_folding_mul_one(bencher: Bencher) {
    let expr = Expr::mul(Expr::variable("x"), Expr::constant(1.0));
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn constant_folding_mul_zero(bencher: Bencher) {
    let expr = Expr::mul(Expr::variable("x"), Expr::constant(0.0));
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Raw computation comparison
#[divan::bench]
fn raw_linear_computation(bencher: Bencher) {
    let x = 2.5f64;

    bencher.bench(|| {
        let result = 2.0 * x + 3.0;
        divan::black_box(result);
    });
}

#[divan::bench]
fn raw_quadratic_computation(bencher: Bencher) {
    let x = 2.5f64;

    bencher.bench(|| {
        let result = x * x + 2.0 * x + 1.0;
        divan::black_box(result);
    });
}

#[divan::bench]
fn raw_complex_polynomial_computation(bencher: Bencher) {
    let x = 2.5f64;

    bencher.bench(|| {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        let result = 3.0 * x4 - 2.0 * x3 + x2 - 5.0 * x + 7.0;
        divan::black_box(result);
    });
}

// Transcendental functions (should show less improvement)
#[divan::bench]
fn transcendental_original(bencher: Bencher) {
    let expr = create_transcendental_expr();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn transcendental_single_var(bencher: Bencher) {
    let expr = create_transcendental_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_single_var("x", 1.0).unwrap());
    });
}

#[divan::bench]
fn transcendental_smart(bencher: Bencher) {
    let expr = create_transcendental_expr();

    bencher.bench(|| {
        divan::black_box(expr.evaluate_smart("x", 1.0).unwrap());
    });
}

#[divan::bench]
fn raw_transcendental_computation(bencher: Bencher) {
    let x = 1.0f64;

    bencher.bench(|| {
        let result = x.sin() + x.cos() + x.exp();
        divan::black_box(result);
    });
}
