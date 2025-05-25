//! Test our ln implementation directly by creating expressions that require runtime ln calls
//!
//! This test creates symbolic expressions that specifically need to compute ln(x) at runtime,
//! not just use pre-computed logarithmic constants.

use measures::exponential_family::symbolic_ir::{Expr, SymbolicLogDensity};
use measures::exponential_family::jit::JITCompiler;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Testing ln Implementation Directly");
    println!("=====================================");
    
    // Create a symbolic expression that requires ln(x) computation
    // Example: f(x) = ln(x) - x  (this is NOT a normal distribution)
    let expr = Expr::Sub(
        Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string())))),
        Box::new(Expr::Var("x".to_string()))
    );
    
    let symbolic = SymbolicLogDensity {
        expression: expr,
        parameters: HashMap::new(),
        variables: vec!["x".to_string()],
    };
    
    println!("Testing expression: ln(x) - x");
    
    // Test symbolic evaluation first
    println!("\n🔧 Testing symbolic evaluation:");
    let test_values = vec![0.5, 1.0, 2.0, 3.0, 5.0, 10.0];
    
    for &x in &test_values {
        match symbolic.evaluate_single("x", x) {
            Ok(symbolic_result) => {
                let expected = (x as f64).ln() - x as f64;  // Rust's built-in ln
                let difference = (symbolic_result - expected).abs();
                
                println!("x={:.1}: symbolic={:.10}, expected={:.10}, diff={:.2e}", 
                    x, symbolic_result, expected, difference);
            }
            Err(e) => {
                println!("x={:.1}: symbolic evaluation failed: {}", x, e);
            }
        }
    }
    
    // Now test JIT compilation
    println!("\n🚀 Testing JIT compilation:");
    
    match JITCompiler::new()?.compile_custom_expression(&symbolic) {
        Ok(jit_fn) => {
            println!("✅ JIT compilation succeeded!");
            
            println!("\nComparing JIT vs Rust's built-in ln:");
            println!("Value\t\tRust ln(x)-x\t\tJIT ln(x)-x\t\tDifference");
            println!("-----\t\t------------\t\t-----------\t\t----------");
            
            let mut max_error = 0.0_f64;
            let mut total_error = 0.0_f64;
            
            for &x in &test_values {
                let rust_result = (x as f64).ln() - x as f64;
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();
                
                max_error = max_error.max(difference);
                total_error += difference;
                
                println!("{:.1}\t\t{:.10}\t\t{:.10}\t\t{:.2e}", 
                    x, rust_result, jit_result, difference);
            }
            
            let avg_error = total_error / test_values.len() as f64;
            
            println!("\n📊 ln Implementation Accuracy:");
            println!("Maximum error: {:.2e}", max_error);
            println!("Average error: {:.2e}", avg_error);
            
            if max_error < 1e-10 {
                println!("✅ Excellent! Our CLIF ln implementation matches Rust's builtin.");
            } else if max_error < 1e-6 {
                println!("✅ Good accuracy! Small differences due to different algorithms.");
            } else {
                println!("⚠️  Significant differences. Our ln implementation needs improvement.");
            }
            
            // Test edge cases
            println!("\n🎯 Testing edge cases:");
            let edge_cases = vec![0.1, 0.01, 0.001, 1.0, 1.001, 1.01, 1.1];
            
            for &x in &edge_cases {
                let rust_result = (x as f64).ln() - x as f64;
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();
                
                println!("x={:.3}: rust={:.12}, jit={:.12}, diff={:.2e}", 
                    x, rust_result, jit_result, difference);
            }
            
        }
        Err(e) => {
            println!("❌ JIT compilation failed: {}", e);
        }
    }
    
    // Test another expression that uses ln multiple times
    println!("\n🔬 Testing more complex ln expression: ln(x) + ln(x+1)");
    
    let complex_expr = Expr::Add(
        Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string())))),
        Box::new(Expr::Ln(Box::new(Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(1.0))
        ))))
    );
    
    let complex_symbolic = SymbolicLogDensity {
        expression: complex_expr,
        parameters: HashMap::new(),
        variables: vec!["x".to_string()],
    };
    
    match JITCompiler::new()?.compile_custom_expression(&complex_symbolic) {
        Ok(jit_fn) => {
            println!("✅ Complex expression JIT compilation succeeded!");
            
            for &x in &[1.0, 2.0, 5.0] {
                let rust_result = (x as f64).ln() + (x as f64 + 1.0).ln();
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - rust_result).abs();
                
                println!("x={:.1}: rust={:.10}, jit={:.10}, diff={:.2e}", 
                    x, rust_result, jit_result, difference);
            }
        }
        Err(e) => {
            println!("❌ Complex expression JIT compilation failed: {}", e);
        }
    }
    
    Ok(())
} 