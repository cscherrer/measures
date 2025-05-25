//! Benchmark comparison: egglog optimization performance
//!
//! Compares expression complexity before and after egglog optimization
//! to measure the effectiveness of mathematical simplification rules.

use measures::exponential_family::egglog_optimizer::EgglogOptimize;
use measures::exponential_family::symbolic_ir::Expr;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = vec![
        ("x + 0", create_expr_x_plus_0()),
        ("x * 0", create_expr_x_times_0()),
        ("ln(exp(x))", create_expr_ln_exp_x()),
        ("distributive", create_distributive_expr()),
        ("log properties", create_log_properties_expr()),
        ("trig identity", create_trig_identity_expr()),
        ("wide expression", create_wide_expression()),
        ("20 terms", create_scalable_expression(20)),
        ("like terms", create_like_terms_expr()),
        ("mixed ops", create_mixed_operations()),
    ];

    println!("Test Case          | Orig | Opt | Time (μs) | Reduction | Quality");
    println!("-------------------|------|-----|-----------|-----------|--------");

    let mut total_time = Duration::new(0, 0);
    let mut total_reductions = 0;
    let mut successful_optimizations = 0;

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();

        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();

        let optimized_complexity = optimized.complexity();
        let complexity_reduction = original_complexity.saturating_sub(optimized_complexity);
        let reduction_percent = if original_complexity > 0 {
            (complexity_reduction * 100) / original_complexity
        } else {
            0
        };

        let quality = evaluate_optimization_quality(&expr, &optimized);

        println!(
            "{:<18} | {:>4} | {:>3} | {:>9.0} | {:>8}% | {:>7}",
            name,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64,
            reduction_percent,
            quality
        );

        total_time += duration;
        total_reductions += complexity_reduction;
        if complexity_reduction > 0 {
            successful_optimizations += 1;
        }
    }

    println!("-------------------|------|-----|-----------|-----------|--------");
    println!(
        "TOTALS             |      |     | {:>9.0} | {:>8} | {:>7}",
        total_time.as_micros() as f64,
        total_reductions,
        successful_optimizations
    );

    println!("\nSummary:");
    println!(
        "  Average time: {:.0}μs",
        total_time.as_micros() as f64 / 10.0
    );
    println!("  Total reduction: {total_reductions}");
    println!("  Success rate: {successful_optimizations}/10");

    Ok(())
}

fn evaluate_optimization_quality(original: &Expr, optimized: &Expr) -> &'static str {
    let vars = HashMap::from([
        ("x".to_string(), 2.5),
        ("a".to_string(), 1.5),
        ("b".to_string(), 3.0),
        ("y".to_string(), 1.2),
        ("z".to_string(), 0.8),
    ]);

    match (original.evaluate(&vars), optimized.evaluate(&vars)) {
        (Ok(orig), Ok(opt)) => {
            let error = (orig - opt).abs();
            if error < 1e-12 {
                "Perfect"
            } else if error < 1e-6 {
                "Good"
            } else if error < 1e-3 {
                "Fair"
            } else {
                "Poor"
            }
        }
        _ => "Error",
    }
}

fn create_expr_x_plus_0() -> Expr {
    Expr::Add(
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Const(0.0)),
    )
}

fn create_expr_x_times_0() -> Expr {
    Expr::Mul(
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Const(0.0)),
    )
}

fn create_expr_ln_exp_x() -> Expr {
    Expr::Ln(Box::new(Expr::Exp(Box::new(Expr::Var("x".to_string())))))
}

fn create_distributive_expr() -> Expr {
    // (a * x) + (b * x) -> (a + b) * x
    Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Var("b".to_string())),
            Box::new(Expr::Var("x".to_string())),
        )),
    )
}

fn create_log_properties_expr() -> Expr {
    // ln(a) + ln(b) -> ln(a * b)
    Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Var("a".to_string())))),
        Box::new(Expr::Ln(Box::new(Expr::Var("b".to_string())))),
    )
}

fn create_trig_identity_expr() -> Expr {
    // sin²(x) + cos²(x) -> 1
    Expr::Add(
        Box::new(Expr::Pow(
            Box::new(Expr::Sin(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Const(2.0)),
        )),
        Box::new(Expr::Pow(
            Box::new(Expr::Cos(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Const(2.0)),
        )),
    )
}

fn create_wide_expression() -> Expr {
    // x + x + x + x + x + x + x + x
    let mut expr = Expr::Var("x".to_string());
    for _ in 0..7 {
        expr = Expr::Add(Box::new(expr), Box::new(Expr::Var("x".to_string())));
    }
    expr
}

fn create_scalable_expression(size: usize) -> Expr {
    // Create expression with 'size' terms: x + 0 + x + 0 + ...
    let mut expr = Expr::Var("x".to_string());
    for i in 1..size {
        if i % 2 == 0 {
            expr = Expr::Add(Box::new(expr), Box::new(Expr::Var("x".to_string())));
        } else {
            expr = Expr::Add(Box::new(expr), Box::new(Expr::Const(0.0)));
        }
    }
    expr
}

fn create_like_terms_expr() -> Expr {
    // 2x + 3x + x -> 6x
    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Var("x".to_string())),
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(3.0)),
                Box::new(Expr::Var("x".to_string())),
            )),
        )),
        Box::new(Expr::Var("x".to_string())),
    )
}

fn create_mixed_operations() -> Expr {
    // ln(exp(x) * exp(y)) + sin²(z) + cos²(z)
    Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Mul(
            Box::new(Expr::Exp(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Exp(Box::new(Expr::Var("y".to_string())))),
        )))),
        Box::new(Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Sin(Box::new(Expr::Var("z".to_string())))),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Cos(Box::new(Expr::Var("z".to_string())))),
                Box::new(Expr::Const(2.0)),
            )),
        )),
    )
}
