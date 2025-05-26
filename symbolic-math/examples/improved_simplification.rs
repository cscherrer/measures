//! Improved Simplification Example
//!
//! This example demonstrates the enhanced simplification capabilities
//! and compares performance with the original implementation.
//!
//! Run with: cargo run --example improved_simplification

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::Expr;

fn main() {
    println!("🔧 Improved Simplification Demonstration");
    println!("========================================\n");

    test_algebraic_identities();
    test_advanced_patterns();
    test_performance_improvements();
}

fn test_algebraic_identities() {
    println!("📐 Algebraic Identity Simplifications");
    println!("-------------------------------------");

    let test_cases = vec![
        ("x + x", create_x_plus_x()),
        ("x - x", create_x_minus_x()),
        ("x * x", create_x_times_x()),
        ("x / x", create_x_div_x()),
        ("(x^2)^3", create_nested_power()),
        ("ln(x^2)", create_ln_power()),
        ("exp(2 * ln(x))", create_exp_ln()),
        ("sqrt(x^2)", create_sqrt_square()),
        ("-(-x)", create_double_negative()),
    ];

    for (name, expr) in test_cases {
        let original_complexity = expr.complexity();
        let simplified = expr.clone().simplify();
        let new_complexity = simplified.complexity();

        let reduction = if original_complexity > 0 {
            (original_complexity - new_complexity) as f64 / original_complexity as f64 * 100.0
        } else {
            0.0
        };

        println!(
            "{:15} | {} → {} | {:5.1}% reduction",
            name, expr, simplified, reduction
        );
    }
    println!();
}

fn test_advanced_patterns() {
    println!("🧠 Advanced Pattern Recognition");
    println!("------------------------------");

    // Test trigonometric constants
    let sin_zero = Expr::sin(Expr::constant(0.0));
    let cos_zero = Expr::cos(Expr::constant(0.0));

    println!("sin(0) = {} → {}", sin_zero.clone(), sin_zero.simplify());
    println!("cos(0) = {} → {}", cos_zero.clone(), cos_zero.simplify());

    // Test logarithmic identities
    let ln_one = Expr::ln(Expr::constant(1.0));
    let exp_zero = Expr::exp(Expr::constant(0.0));

    println!("ln(1) = {} → {}", ln_one.clone(), ln_one.simplify());
    println!("exp(0) = {} → {}", exp_zero.clone(), exp_zero.simplify());

    // Test power identities
    let one_to_x = Expr::pow(Expr::constant(1.0), Expr::variable("x"));
    println!("1^x = {} → {}", one_to_x.clone(), one_to_x.simplify());

    println!();
}

fn test_performance_improvements() {
    println!("⚡ Performance Comparison");
    println!("------------------------");

    let complex_expressions = vec![
        ("Redundant arithmetic", create_redundant_arithmetic()),
        ("Nested powers", create_deeply_nested_powers()),
        ("Mixed operations", create_mixed_operations()),
        ("Trigonometric", create_trig_expression()),
    ];

    let iterations = 1000;

    for (name, expr) in complex_expressions {
        let original_complexity = expr.complexity();

        // Measure simplification time
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = expr.clone().simplify();
        }
        let simplify_time = start.elapsed();

        let simplified = expr.clone().simplify();
        let new_complexity = simplified.complexity();

        // Measure evaluation time before simplification
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        vars.insert("y".to_string(), 3.0);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = expr.evaluate(&vars);
        }
        let original_eval_time = start.elapsed();

        // Measure evaluation time after simplification
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simplified.evaluate(&vars);
        }
        let simplified_eval_time = start.elapsed();

        let complexity_reduction = if original_complexity > 0 {
            (original_complexity - new_complexity) as f64 / original_complexity as f64 * 100.0
        } else {
            0.0
        };

        let eval_speedup = if simplified_eval_time.as_nanos() > 0 {
            original_eval_time.as_nanos() as f64 / simplified_eval_time.as_nanos() as f64
        } else {
            1.0
        };

        println!(
            "{:15} | {:2} → {:2} ops ({:5.1}%) | Simplify: {:6.2} μs | Eval speedup: {:4.2}x",
            name,
            original_complexity,
            new_complexity,
            complexity_reduction,
            simplify_time.as_micros(),
            eval_speedup
        );
    }

    println!();
}

// Helper functions to create test expressions

fn create_x_plus_x() -> Expr {
    Expr::add(Expr::variable("x"), Expr::variable("x"))
}

fn create_x_minus_x() -> Expr {
    Expr::sub(Expr::variable("x"), Expr::variable("x"))
}

fn create_x_times_x() -> Expr {
    Expr::mul(Expr::variable("x"), Expr::variable("x"))
}

fn create_x_div_x() -> Expr {
    Expr::div(Expr::variable("x"), Expr::variable("x"))
}

fn create_nested_power() -> Expr {
    let inner = Expr::pow(Expr::variable("x"), Expr::constant(2.0));
    Expr::pow(inner, Expr::constant(3.0))
}

fn create_ln_power() -> Expr {
    let power = Expr::pow(Expr::variable("x"), Expr::constant(2.0));
    Expr::ln(power)
}

fn create_exp_ln() -> Expr {
    let ln_x = Expr::ln(Expr::variable("x"));
    let two_ln_x = Expr::mul(Expr::constant(2.0), ln_x);
    Expr::exp(two_ln_x)
}

fn create_sqrt_square() -> Expr {
    let x_squared = Expr::pow(Expr::variable("x"), Expr::constant(2.0));
    Expr::sqrt(x_squared)
}

fn create_double_negative() -> Expr {
    let neg_x = Expr::neg(Expr::variable("x"));
    Expr::neg(neg_x)
}

fn create_redundant_arithmetic() -> Expr {
    // ((x + 0) * 1) + (x - x) + (0 * y)
    let x = Expr::variable("x");
    let y = Expr::variable("y");

    let term1 = Expr::mul(
        Expr::add(x.clone(), Expr::constant(0.0)),
        Expr::constant(1.0),
    );
    let term2 = Expr::sub(x.clone(), x);
    let term3 = Expr::mul(Expr::constant(0.0), y);

    Expr::add(Expr::add(term1, term2), term3)
}

fn create_deeply_nested_powers() -> Expr {
    // ((x^2)^3)^2 = x^12
    let x = Expr::variable("x");
    let x_squared = Expr::pow(x, Expr::constant(2.0));
    let x_to_sixth = Expr::pow(x_squared, Expr::constant(3.0));
    Expr::pow(x_to_sixth, Expr::constant(2.0))
}

fn create_mixed_operations() -> Expr {
    // ln(exp(x)) + sqrt(y^2) - (x - x)
    let x = Expr::variable("x");
    let y = Expr::variable("y");

    let term1 = Expr::ln(Expr::exp(x.clone()));
    let term2 = Expr::sqrt(Expr::pow(y, Expr::constant(2.0)));
    let term3 = Expr::sub(x.clone(), x);

    Expr::sub(Expr::add(term1, term2), term3)
}

fn create_trig_expression() -> Expr {
    // sin(0) + cos(0) + sin(x) * cos(x)
    let sin_zero = Expr::sin(Expr::constant(0.0));
    let cos_zero = Expr::cos(Expr::constant(0.0));
    let sin_x = Expr::sin(Expr::variable("x"));
    let cos_x = Expr::cos(Expr::variable("x"));

    Expr::add(Expr::add(sin_zero, cos_zero), Expr::mul(sin_x, cos_x))
}
