//! Comprehensive benchmark for egglog optimization performance
//!
//! This benchmark measures:
//! - Optimization time for various expression types
//! - Optimization quality (complexity reduction)
//! - Memory usage during optimization
//! - Scalability with expression size
//!
//! Run with: cargo run --example egglog_benchmark --features jit --release

use measures::exponential_family::symbolic_ir::Expr;
use measures::exponential_family::egglog_optimizer::{EgglogOptimizer, EgglogOptimize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Egglog Optimization Benchmark");
    println!("=================================\n");

    // Run different benchmark categories
    benchmark_basic_optimizations()?;
    benchmark_advanced_optimizations()?;
    benchmark_complex_expressions()?;
    benchmark_scalability()?;
    benchmark_real_world_cases()?;

    println!("\n🏁 Benchmark Complete!");
    Ok(())
}

/// Benchmark basic algebraic optimizations
fn benchmark_basic_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Basic Optimizations Benchmark");
    println!("---------------------------------");

    let test_cases = vec![
        ("x + 0", create_expr_x_plus_0()),
        ("x * 0", create_expr_x_times_0()),
        ("x * 1", create_expr_x_times_1()),
        ("x^1", create_expr_x_power_1()),
        ("x^0", create_expr_x_power_0()),
        ("ln(exp(x))", create_expr_ln_exp_x()),
        ("exp(ln(x))", create_expr_exp_ln_x()),
        ("-(-(x))", create_expr_double_neg()),
    ];

    let mut total_time = Duration::new(0, 0);
    let mut total_optimizations = 0;
    let mut total_complexity_reduction = 0;

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();
        
        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();
        
        let optimized_complexity = optimized.complexity();
        let complexity_reduction = original_complexity.saturating_sub(optimized_complexity);
        
        println!("  {:<15} | {:>3} → {:>3} | {:>8.2}μs | {}% reduction", 
            name,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64,
            if original_complexity > 0 { 
                (complexity_reduction * 100) / original_complexity 
            } else { 0 }
        );

        total_time += duration;
        total_optimizations += 1;
        total_complexity_reduction += complexity_reduction;
    }

    println!("  Average time: {:.2}μs", total_time.as_micros() as f64 / total_optimizations as f64);
    println!("  Total complexity reduction: {}", total_complexity_reduction);
    println!();

    Ok(())
}

/// Benchmark advanced mathematical optimizations
fn benchmark_advanced_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Advanced Optimizations Benchmark");
    println!("------------------------------------");

    let test_cases = vec![
        ("Distributive", create_distributive_expr()),
        ("Log properties", create_log_properties_expr()),
        ("Exp properties", create_exp_properties_expr()),
        ("Trig identity", create_trig_identity_expr()),
        ("Power laws", create_power_laws_expr()),
        ("Polynomial", create_polynomial_expr()),
    ];

    let mut total_time = Duration::new(0, 0);
    let mut total_optimizations = 0;

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();
        
        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();
        
        let optimized_complexity = optimized.complexity();
        let complexity_reduction = original_complexity.saturating_sub(optimized_complexity);
        
        println!("  {:<15} | {:>3} → {:>3} | {:>8.2}μs | {}% reduction", 
            name,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64,
            if original_complexity > 0 { 
                (complexity_reduction * 100) / original_complexity 
            } else { 0 }
        );

        total_time += duration;
        total_optimizations += 1;
    }

    println!("  Average time: {:.2}μs", total_time.as_micros() as f64 / total_optimizations as f64);
    println!();

    Ok(())
}

/// Benchmark complex nested expressions
fn benchmark_complex_expressions() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Complex Expressions Benchmark");
    println!("---------------------------------");

    let test_cases = vec![
        ("Nested poly", create_nested_polynomial()),
        ("Mixed ops", create_mixed_operations()),
        ("Deep nesting", create_deep_nesting()),
        ("Wide expr", create_wide_expression()),
    ];

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();
        
        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();
        
        let optimized_complexity = optimized.complexity();
        let complexity_reduction = original_complexity.saturating_sub(optimized_complexity);
        
        println!("  {:<15} | {:>3} → {:>3} | {:>8.2}μs | {}% reduction", 
            name,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64,
            if original_complexity > 0 { 
                (complexity_reduction * 100) / original_complexity 
            } else { 0 }
        );
    }
    println!();

    Ok(())
}

/// Benchmark scalability with expression size
fn benchmark_scalability() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Scalability Benchmark");
    println!("-------------------------");

    let sizes = vec![5, 10, 20, 50];
    
    for size in sizes {
        let expr = create_scalable_expression(size);
        let original_complexity = expr.complexity();
        
        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();
        
        let optimized_complexity = optimized.complexity();
        
        println!("  Size {:>2} terms   | {:>3} → {:>3} | {:>8.2}μs", 
            size,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64
        );
    }
    println!();

    Ok(())
}

/// Benchmark real-world probability distribution expressions
fn benchmark_real_world_cases() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 Real-World Cases Benchmark");
    println!("------------------------------");

    let test_cases = vec![
        ("Normal PDF", create_normal_pdf()),
        ("Gamma PDF", create_gamma_pdf()),
        ("Beta PDF", create_beta_pdf()),
        ("Mixture", create_mixture_model()),
    ];

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();
        
        let start = Instant::now();
        let optimized = expr.optimize_with_egglog()?;
        let duration = start.elapsed();
        
        let optimized_complexity = optimized.complexity();
        let complexity_reduction = original_complexity.saturating_sub(optimized_complexity);
        
        println!("  {:<15} | {:>3} → {:>3} | {:>8.2}μs | {}% reduction", 
            name,
            original_complexity,
            optimized_complexity,
            duration.as_micros() as f64,
            if original_complexity > 0 { 
                (complexity_reduction * 100) / original_complexity 
            } else { 0 }
        );

        // Verify functional equivalence
        let vars = HashMap::from([
            ("x".to_string(), 2.5),
            ("mu".to_string(), 1.0),
            ("sigma".to_string(), 1.5),
            ("alpha".to_string(), 2.0),
            ("beta".to_string(), 3.0),
        ]);

        match (expr.evaluate(&vars), optimized.evaluate(&vars)) {
            (Ok(orig), Ok(opt)) => {
                let error = (orig - opt).abs();
                if error > 1e-10 {
                    println!("    ⚠️  Accuracy warning: {:.2e} error", error);
                }
            }
            _ => println!("    ❌ Evaluation error"),
        }
    }
    println!();

    Ok(())
}

// Expression creation functions
fn create_expr_x_plus_0() -> Expr {
    Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(0.0)))
}

fn create_expr_x_times_0() -> Expr {
    Expr::Mul(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(0.0)))
}

fn create_expr_x_times_1() -> Expr {
    Expr::Mul(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(1.0)))
}

fn create_expr_x_power_1() -> Expr {
    Expr::Pow(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(1.0)))
}

fn create_expr_x_power_0() -> Expr {
    Expr::Pow(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(0.0)))
}

fn create_expr_ln_exp_x() -> Expr {
    Expr::Ln(Box::new(Expr::Exp(Box::new(Expr::Var("x".to_string())))))
}

fn create_expr_exp_ln_x() -> Expr {
    Expr::Exp(Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string())))))
}

fn create_expr_double_neg() -> Expr {
    Expr::Neg(Box::new(Expr::Neg(Box::new(Expr::Var("x".to_string())))))
}

fn create_distributive_expr() -> Expr {
    // (a * x) + (b * x) -> (a + b) * x
    Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("x".to_string()))
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Var("b".to_string())),
            Box::new(Expr::Var("x".to_string()))
        ))
    )
}

fn create_log_properties_expr() -> Expr {
    // ln(a) + ln(b) -> ln(a * b)
    Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Var("a".to_string())))),
        Box::new(Expr::Ln(Box::new(Expr::Var("b".to_string()))))
    )
}

fn create_exp_properties_expr() -> Expr {
    // exp(a) * exp(b) -> exp(a + b)
    Expr::Mul(
        Box::new(Expr::Exp(Box::new(Expr::Var("a".to_string())))),
        Box::new(Expr::Exp(Box::new(Expr::Var("b".to_string()))))
    )
}

fn create_trig_identity_expr() -> Expr {
    // sin²(x) + cos²(x) -> 1
    Expr::Add(
        Box::new(Expr::Pow(
            Box::new(Expr::Sin(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Const(2.0))
        )),
        Box::new(Expr::Pow(
            Box::new(Expr::Cos(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Const(2.0))
        ))
    )
}

fn create_power_laws_expr() -> Expr {
    // x^a * x^b -> x^(a+b)
    Expr::Mul(
        Box::new(Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("a".to_string()))
        )),
        Box::new(Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("b".to_string()))
        ))
    )
}

fn create_polynomial_expr() -> Expr {
    // 2x² + 3x + 1
    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Pow(
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Const(2.0))
                ))
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(3.0)),
                Box::new(Expr::Var("x".to_string()))
            ))
        )),
        Box::new(Expr::Const(1.0))
    )
}

fn create_nested_polynomial() -> Expr {
    // ((x + 1)² + 2(x + 1) + 3)
    let x_plus_1 = Expr::Add(
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Const(1.0))
    );
    
    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Pow(Box::new(x_plus_1.clone()), Box::new(Expr::Const(2.0)))),
            Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x_plus_1)))
        )),
        Box::new(Expr::Const(3.0))
    )
}

fn create_mixed_operations() -> Expr {
    // ln(exp(x) * exp(y)) + sin²(z) + cos²(z)
    Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Mul(
            Box::new(Expr::Exp(Box::new(Expr::Var("x".to_string())))),
            Box::new(Expr::Exp(Box::new(Expr::Var("y".to_string()))))
        )))),
        Box::new(Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Sin(Box::new(Expr::Var("z".to_string())))),
                Box::new(Expr::Const(2.0))
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Cos(Box::new(Expr::Var("z".to_string())))),
                Box::new(Expr::Const(2.0))
            ))
        ))
    )
}

fn create_deep_nesting() -> Expr {
    // ln(exp(ln(exp(ln(exp(x))))))
    let mut expr = Expr::Var("x".to_string());
    for _ in 0..3 {
        expr = Expr::Exp(Box::new(expr));
        expr = Expr::Ln(Box::new(expr));
    }
    expr
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

fn create_normal_pdf() -> Expr {
    // -0.5 * ((x - mu) / sigma)² - ln(sigma) - 0.5 * ln(2π)
    let x = Expr::Var("x".to_string());
    let mu = Expr::Var("mu".to_string());
    let sigma = Expr::Var("sigma".to_string());
    
    let diff = Expr::Sub(Box::new(x), Box::new(mu));
    let standardized = Expr::Div(Box::new(diff), Box::new(sigma.clone()));
    let quadratic = Expr::Pow(Box::new(standardized), Box::new(Expr::Const(2.0)));
    
    Expr::Sub(
        Box::new(Expr::Sub(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(-0.5)),
                Box::new(quadratic)
            )),
            Box::new(Expr::Ln(Box::new(sigma)))
        )),
        Box::new(Expr::Const(0.5 * (2.0 * std::f64::consts::PI).ln()))
    )
}

fn create_gamma_pdf() -> Expr {
    // (alpha - 1) * ln(x) - x / beta - ln(Γ(alpha)) - alpha * ln(beta)
    let x = Expr::Var("x".to_string());
    let alpha = Expr::Var("alpha".to_string());
    let beta = Expr::Var("beta".to_string());
    
    Expr::Sub(
        Box::new(Expr::Sub(
            Box::new(Expr::Mul(
                Box::new(Expr::Sub(Box::new(alpha.clone()), Box::new(Expr::Const(1.0)))),
                Box::new(Expr::Ln(Box::new(x.clone())))
            )),
            Box::new(Expr::Div(Box::new(x), Box::new(beta.clone())))
        )),
        Box::new(Expr::Mul(
            Box::new(alpha),
            Box::new(Expr::Ln(Box::new(beta)))
        ))
    )
}

fn create_beta_pdf() -> Expr {
    // (alpha - 1) * ln(x) + (beta - 1) * ln(1 - x)
    let x = Expr::Var("x".to_string());
    let alpha = Expr::Var("alpha".to_string());
    let beta = Expr::Var("beta".to_string());
    
    Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Sub(Box::new(alpha), Box::new(Expr::Const(1.0)))),
            Box::new(Expr::Ln(Box::new(x.clone())))
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Sub(Box::new(beta), Box::new(Expr::Const(1.0)))),
            Box::new(Expr::Ln(Box::new(Expr::Sub(
                Box::new(Expr::Const(1.0)),
                Box::new(x)
            ))))
        ))
    )
}

fn create_mixture_model() -> Expr {
    // ln(0.5 * exp(normal1) + 0.5 * exp(normal2))
    let normal1 = create_normal_pdf();
    let normal2 = create_normal_pdf(); // In practice, would have different parameters
    
    Expr::Ln(Box::new(Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::Exp(Box::new(normal1)))
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::Exp(Box::new(normal2)))
        ))
    )))
} 