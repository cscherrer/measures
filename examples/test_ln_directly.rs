//! Test ln implementation directly
//!
//! Creates symbolic expressions that require runtime ln(x) computation
//! to verify the accuracy of our JIT-compiled ln function.

use measures::core::{HasLogDensity, LogDensity, Measure};
use measures::distributions::continuous::Normal;
use measures::exponential_family::jit::CustomSymbolicLogDensity;
use measures::exponential_family::jit::JITCompiler;
use measures::exponential_family::traits::ExponentialFamily;
use measures::traits::DotProduct;
use std::collections::HashMap;
use symbolic_math::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test expression: ln(x) - x
    let expr = Expr::Sub(
        Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string())))),
        Box::new(Expr::Var("x".to_string())),
    );

    let symbolic = CustomSymbolicLogDensity {
        expression: expr,
        parameters: HashMap::new(),
        variables: vec!["x".to_string()],
    };

    let test_values = vec![0.5, 1.0, 2.0, 3.0, 5.0, 10.0];

    // Test symbolic evaluation
    println!("Testing symbolic evaluation of ln(x) - x:");
    for &x in &test_values {
        match symbolic.evaluate_single("x", x) {
            Ok(symbolic_result) => {
                let expected = (x as f64).ln() - x as f64;
                let difference = (symbolic_result - expected).abs();
                println!(
                    "x={:.1}: symbolic={:.10}, expected={:.10}, diff={:.2e}",
                    x, symbolic_result, expected, difference
                );
            }
            Err(e) => {
                println!("x={:.1}: symbolic evaluation failed: {}", x, e);
            }
        }
    }

    // Test JIT compilation
    println!("\nTesting JIT compilation:");
    match JITCompiler::new()?.compile_custom_expression(&symbolic) {
        Ok(jit_fn) => {
            let mut max_error = 0.0_f64;
            let mut total_error = 0.0_f64;

            for &x in &test_values {
                let rust_result = (x as f64).ln() - x as f64;
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();

                max_error = max_error.max(difference);
                total_error += difference;

                println!(
                    "x={:.1}: rust={:.10}, jit={:.10}, diff={:.2e}",
                    x, rust_result, jit_result, difference
                );
            }

            let avg_error = total_error / test_values.len() as f64;
            println!("Max error: {:.2e}, Avg error: {:.2e}", max_error, avg_error);

            // Test edge cases
            println!("\nEdge cases:");
            let edge_cases = vec![0.1, 0.01, 0.001, 1.0, 1.001, 1.01, 1.1];
            for &x in &edge_cases {
                let rust_result = (x as f64).ln() - x as f64;
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();
                println!("x={:.3}: diff={:.2e}", x, difference);
            }
        }
        Err(e) => {
            println!("JIT compilation failed: {}", e);
        }
    }

    // Test complex expression: ln(x) + ln(x+1)
    println!("\nTesting complex expression: ln(x) + ln(x+1)");
    let complex_expr = Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string())))),
        Box::new(Expr::Ln(Box::new(Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(1.0)),
        )))),
    );

    let complex_symbolic = CustomSymbolicLogDensity {
        expression: complex_expr,
        parameters: HashMap::new(),
        variables: vec!["x".to_string()],
    };

    match JITCompiler::new()?.compile_custom_expression(&complex_symbolic) {
        Ok(jit_fn) => {
            for &x in &[1.0, 2.0, 5.0] {
                let rust_result = (x as f64).ln() + (x as f64 + 1.0).ln();
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();
                println!("x={:.1}: diff={:.2e}", x, difference);
            }
        }
        Err(e) => {
            println!("Complex expression JIT compilation failed: {}", e);
        }
    }

    Ok(())
}
