//! Execution Overhead Analysis
//!
//! This benchmark analyzes the breakdown between call overhead and actual computation
//! to understand where time is spent during expression evaluation.
//!
//! Run with: cargo bench --bench `execution_overhead_analysis` --features "jit optimization"

use divan::Bencher;
use std::collections::HashMap;
use symbolic_math::Expr;

#[cfg(feature = "jit")]
use symbolic_math::GeneralJITCompiler;

fn main() {
    divan::main();
}

// Baseline benchmarks - minimal computation to measure call overhead
#[divan::bench]
fn baseline_constant_evaluation(bencher: Bencher) {
    let expr = Expr::constant(42.0);
    let vars = HashMap::new();

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn baseline_variable_lookup(bencher: Bencher) {
    let expr = Expr::variable("x");
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 42.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn baseline_single_addition(bencher: Bencher) {
    let expr = Expr::add(Expr::constant(1.0), Expr::constant(2.0));
    let vars = HashMap::new();

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Computational complexity benchmarks
#[divan::bench]
fn computation_linear_operations(bencher: Bencher) {
    // x + 1 + 2 + 3 + 4 + 5 (6 operations)
    let mut expr = Expr::variable("x");
    for i in 1..=5 {
        expr = Expr::add(expr, Expr::constant(f64::from(i)));
    }

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn computation_multiplication_chain(bencher: Bencher) {
    // x * 2 * 3 * 4 * 5 * 6 (6 operations)
    let mut expr = Expr::variable("x");
    for i in 2..=6 {
        expr = Expr::mul(expr, Expr::constant(f64::from(i)));
    }

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.5);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn computation_polynomial_degree_2(bencher: Bencher) {
    // x^2 + 2x + 1 (3 operations: pow, mul, add, add)
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn computation_polynomial_degree_4(bencher: Bencher) {
    // 3x^4 - 2x^3 + x^2 - 5x + 7
    let x = Expr::variable("x");
    let expr = Expr::add(
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
    );

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench]
fn computation_transcendental_functions(bencher: Bencher) {
    // sin(x) + cos(x) + exp(x) + ln(x)
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::add(Expr::sin(x.clone()), Expr::cos(x.clone())),
            Expr::exp(x.clone()),
        ),
        Expr::ln(x),
    );

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    bencher.bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Nested expression benchmarks to measure call stack overhead
#[divan::bench(args = [1, 2, 4, 8, 16])]
fn overhead_nested_additions(bencher: Bencher, depth: usize) {
    // Create deeply nested additions: (((x + 1) + 1) + 1)...
    let mut expr = Expr::variable("x");
    for _ in 0..depth {
        expr = Expr::add(expr, Expr::constant(1.0));
    }

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);

    bencher.counter(depth).bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

#[divan::bench(args = [1, 2, 4, 8, 16])]
fn overhead_nested_multiplications(bencher: Bencher, depth: usize) {
    // Create deeply nested multiplications: (((x * 2) * 2) * 2)...
    let mut expr = Expr::variable("x");
    for _ in 0..depth {
        expr = Expr::mul(expr, Expr::constant(2.0));
    }

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    bencher.counter(depth).bench(|| {
        divan::black_box(expr.evaluate(&vars).unwrap());
    });
}

// Direct computation benchmarks (raw Rust) for comparison
#[divan::bench]
fn raw_computation_polynomial_degree_2(bencher: Bencher) {
    let x = 3.0f64;

    bencher.bench(|| {
        let result = x * x + 2.0 * x + 1.0;
        divan::black_box(result);
    });
}

#[divan::bench]
fn raw_computation_polynomial_degree_4(bencher: Bencher) {
    let x = 2.0f64;

    bencher.bench(|| {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        let result = 3.0 * x4 - 2.0 * x3 + x2 - 5.0 * x + 7.0;
        divan::black_box(result);
    });
}

#[divan::bench]
fn raw_computation_transcendental(bencher: Bencher) {
    let x = 1.0f64;

    bencher.bench(|| {
        let result = x.sin() + x.cos() + x.exp() + x.ln();
        divan::black_box(result);
    });
}

// Batch evaluation to amortize call overhead
#[divan::bench(args = [1, 10, 100, 1000])]
fn batch_vs_individual_polynomial(bencher: Bencher, size: usize) {
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    bencher.counter(size).bench(|| {
        divan::black_box(expr.evaluate_batch("x", &values).unwrap());
    });
}

#[divan::bench(args = [1, 10, 100, 1000])]
fn individual_calls_polynomial(bencher: Bencher, size: usize) {
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    bencher.counter(size).bench(|| {
        for &val in &values {
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), val);
            divan::black_box(expr.evaluate(&vars).unwrap());
        }
    });
}

// JIT compilation overhead analysis (compilation only, not execution due to thread safety)
#[cfg(feature = "jit")]
#[divan::bench]
fn jit_compilation_overhead(bencher: Bencher) {
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    bencher.bench(|| {
        let compiler = GeneralJITCompiler::new().unwrap();
        let jit_func = compiler
            .compile_expression(&expr, &["x".to_string()], &[], &HashMap::new())
            .unwrap();
        divan::black_box(jit_func);
    });
}

// Memory allocation overhead analysis
#[divan::bench]
fn allocation_overhead_simple(bencher: Bencher) {
    bencher.bench(|| {
        let expr = Expr::add(Expr::constant(1.0), Expr::constant(2.0));
        divan::black_box(expr);
    });
}

#[divan::bench]
fn allocation_overhead_complex(bencher: Bencher) {
    bencher.bench(|| {
        let x = Expr::variable("x");
        let expr = Expr::add(
            Expr::mul(
                Expr::pow(x.clone(), Expr::constant(2.0)),
                Expr::sin(x.clone()),
            ),
            Expr::exp(x),
        );
        divan::black_box(expr);
    });
}
