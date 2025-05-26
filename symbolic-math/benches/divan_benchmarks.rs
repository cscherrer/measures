//! Divan Benchmarks for Symbolic Math
//!
//! Run with: cargo bench --bench `divan_benchmarks`
//! Or with features: cargo bench --bench `divan_benchmarks` --features "jit optimization"

use divan::Bencher;
use std::collections::HashMap;
use symbolic_math::{Expr, clear_caches};

#[cfg(feature = "jit")]
use symbolic_math::GeneralJITCompiler;

#[cfg(feature = "optimization")]
use symbolic_math::EgglogOptimize;

fn main() {
    divan::main();
}

// Expression creation benchmarks
#[divan::bench]
fn create_simple_expression() -> Expr {
    Expr::add(
        Expr::mul(Expr::constant(2.0), Expr::variable("x")),
        Expr::constant(1.0),
    )
}

#[divan::bench]
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

#[divan::bench]
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

// Simplification benchmarks
#[divan::bench]
fn simplify_polynomial(bencher: Bencher) {
    let expr = create_polynomial();
    bencher.bench(|| expr.clone().simplify());
}

#[divan::bench]
fn simplify_trigonometric(bencher: Bencher) {
    let expr = Expr::add(
        Expr::pow(Expr::sin(Expr::variable("x")), Expr::constant(2.0)),
        Expr::pow(Expr::cos(Expr::variable("x")), Expr::constant(2.0)),
    );
    bencher.bench(|| expr.clone().simplify());
}

#[divan::bench]
fn simplify_algebraic_identity(bencher: Bencher) {
    // x + x + x - x should simplify to 2x
    let expr = Expr::sub(
        Expr::add(
            Expr::add(Expr::variable("x"), Expr::variable("x")),
            Expr::variable("x"),
        ),
        Expr::variable("x"),
    );
    bencher.bench(|| expr.clone().simplify());
}

// Evaluation benchmarks
#[divan::bench(args = [1, 10, 100, 1000])]
fn evaluate_polynomial_single(bencher: Bencher, size: usize) {
    let expr = create_polynomial();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    bencher.counter(size).bench(|| {
        for _ in 0..size {
            divan::black_box(expr.evaluate(&vars).unwrap());
        }
    });
}

#[divan::bench(args = [1, 10, 100, 1000])]
fn evaluate_polynomial_batch(bencher: Bencher, size: usize) {
    let expr = create_polynomial();
    let values: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();

    bencher.counter(size).bench(|| {
        divan::black_box(expr.evaluate_batch("x", &values).unwrap());
    });
}

#[divan::bench(args = [1, 10, 100, 1000])]
fn evaluate_complex_expression(bencher: Bencher, size: usize) {
    let expr = create_complex_expression();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    vars.insert("y".to_string(), 3.0);

    bencher.counter(size).bench(|| {
        for _ in 0..size {
            divan::black_box(expr.evaluate(&vars).unwrap());
        }
    });
}

// Caching benchmarks
#[divan::bench]
fn evaluate_with_cache_cold(bencher: Bencher) {
    let expr = create_polynomial();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    bencher.bench(|| {
        clear_caches();
        divan::black_box(expr.evaluate_cached(&vars).unwrap());
    });
}

#[divan::bench]
fn evaluate_with_cache_warm(bencher: Bencher) {
    let expr = create_polynomial();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.5);

    // Warm up the cache
    let _ = expr.evaluate_cached(&vars);

    bencher.bench(|| {
        divan::black_box(expr.evaluate_cached(&vars).unwrap());
    });
}

#[divan::bench]
fn simplify_with_cache_cold(bencher: Bencher) {
    let expr = create_complex_expression();

    bencher.bench(|| {
        clear_caches();
        divan::black_box(expr.clone().simplify_cached());
    });
}

#[divan::bench]
fn simplify_with_cache_warm(bencher: Bencher) {
    let expr = create_complex_expression();

    // Warm up the cache
    let _ = expr.clone().simplify_cached();

    bencher.bench(|| {
        divan::black_box(expr.clone().simplify_cached());
    });
}

// Grid evaluation benchmarks
#[divan::bench(args = [5, 10, 20, 50])]
fn evaluate_grid(bencher: Bencher, grid_size: usize) {
    let expr = Expr::add(
        Expr::pow(Expr::variable("x"), Expr::constant(2.0)),
        Expr::pow(Expr::variable("y"), Expr::constant(2.0)),
    );

    let x_values: Vec<f64> = (0..grid_size).map(|i| i as f64).collect();
    let y_values: Vec<f64> = (0..grid_size).map(|i| i as f64).collect();

    bencher.counter(grid_size * grid_size).bench(|| {
        divan::black_box(expr.evaluate_grid("x", &x_values, "y", &y_values).unwrap());
    });
}

// JIT compilation benchmarks
#[cfg(feature = "jit")]
#[divan::bench]
fn jit_compile_polynomial(bencher: Bencher) {
    let expr = create_polynomial();

    bencher.bench(|| {
        let compiler = GeneralJITCompiler::new().unwrap();
        divan::black_box(
            compiler
                .compile_expression(&expr, &["x".to_string()], &[], &HashMap::new())
                .unwrap(),
        );
    });
}

// Advanced optimization benchmarks
#[cfg(feature = "optimization")]
#[divan::bench]
fn egglog_optimize_simple(bencher: Bencher) {
    let expr = Expr::sub(
        Expr::add(
            Expr::add(Expr::variable("x"), Expr::variable("x")),
            Expr::variable("x"),
        ),
        Expr::variable("x"),
    );

    bencher.bench(|| {
        divan::black_box(
            expr.clone()
                .optimize_with_egglog()
                .unwrap_or_else(|_| expr.clone()),
        );
    });
}

#[cfg(feature = "optimization")]
#[divan::bench]
fn egglog_optimize_trigonometric(bencher: Bencher) {
    let expr = Expr::add(
        Expr::pow(Expr::sin(Expr::variable("x")), Expr::constant(2.0)),
        Expr::pow(Expr::cos(Expr::variable("x")), Expr::constant(2.0)),
    );

    bencher.bench(|| {
        divan::black_box(
            expr.clone()
                .optimize_with_egglog()
                .unwrap_or_else(|_| expr.clone()),
        );
    });
}

// Complexity analysis benchmarks
#[divan::bench(args = [2, 4, 6, 8, 10])]
fn complexity_scaling(bencher: Bencher, depth: usize) {
    // Create nested expression: ((((x + 1) + 1) + 1) + 1)...
    let mut expr = Expr::variable("x");
    for _ in 0..depth {
        expr = Expr::add(expr, Expr::constant(1.0));
    }

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    bencher.counter(depth).bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Memory usage benchmarks
#[divan::bench(args = [10, 100, 1000])]
fn memory_usage_large_expressions(bencher: Bencher, terms: usize) {
    bencher.counter(terms).bench(|| {
        let mut expr = Expr::constant(0.0);
        for i in 0..terms {
            let term = Expr::mul(
                Expr::constant(i as f64),
                Expr::pow(Expr::variable("x"), Expr::constant(i as f64)),
            );
            expr = Expr::add(expr, term);
        }
        divan::black_box(expr);
    });
}
