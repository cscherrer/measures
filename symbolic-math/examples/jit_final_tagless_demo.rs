//! JIT Final Tagless Demo
//!
//! This example demonstrates the JITEval interpreter that directly compiles
//! final tagless expressions to native machine code, bypassing the Expr AST entirely.
//!
//! This represents the ultimate performance for symbolic mathematics in Rust,
//! providing zero-cost abstractions with native speed evaluation.

#[cfg(feature = "jit")]
use symbolic_math::final_tagless::{JITEval, MathExpr};

#[cfg(feature = "jit")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 JIT Final Tagless Demo - Ultimate Performance Symbolic Math");
    println!("================================================================");
    
    // Example 1: Simple polynomial - x^2 + 2*x + 1
    println!("\n📊 Example 1: Polynomial x^2 + 2*x + 1");
    
    let x = JITEval::var::<f64>("x");
    let one = JITEval::constant::<f64>(1.0);
    let two = JITEval::constant::<f64>(2.0);
    let x_var = JITEval::var::<f64>("x");
    let x_var2 = JITEval::var::<f64>("x");
    
    let x_squared = JITEval::pow::<f64>(x_var, two);
    let two_x = JITEval::mul::<f64, f64, f64>(JITEval::constant::<f64>(2.0), x_var2);
    let polynomial = JITEval::add::<f64, f64, f64>(JITEval::add::<f64, f64, f64>(x_squared, two_x), one);
    
    let compiled_poly = JITEval::compile(polynomial)?;
    
    println!("Expression: {}", compiled_poly.source_expression);
    println!("Compilation stats: {:?}", compiled_poly.compilation_stats);
    
    // Test the polynomial
    for x_val in [0.0, 1.0, 2.0, 3.0, -1.0] {
        let result = compiled_poly.call_single(x_val);
        let expected = (x_val + 1.0).powi(2); // (x+1)^2
        println!("  f({x_val:4.1}) = {result:8.3} (expected: {expected:8.3})");
    }
    
    // Example 2: Transcendental function - exp(sin(x))
    println!("\n🌊 Example 2: Transcendental exp(sin(x))");
    
    fn transcendental_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
        E::exp(E::sin(x))
    }
    
    let transcendental = transcendental_expr::<JITEval>(JITEval::var::<f64>("x"));
    let compiled_trans = JITEval::compile(transcendental)?;
    
    println!("Expression: {}", compiled_trans.source_expression);
    println!("Compilation stats: {:?}", compiled_trans.compilation_stats);
    
    // Test the transcendental function
    for x_val in [0.0, std::f64::consts::PI / 6.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0] {
        let result = compiled_trans.call_single(x_val);
        println!("  exp(sin({x_val:4.2})) = {result:8.3}");
    }
    
    // Example 3: Performance comparison
    println!("\n⚡ Example 3: Performance Comparison");
    
    // Simple expression: x^2 + 1
    fn simple_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
        let one = E::constant(1.0);
        E::add(E::pow(x, E::constant(2.0)), one)
    }
    
    let jit_expr = simple_expr::<JITEval>(JITEval::var::<f64>("x"));
    let compiled_simple = JITEval::compile(jit_expr)?;
    
    // Benchmark JIT evaluation
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for i in 0..1_000_000 {
        sum += compiled_simple.call_single(i as f64);
    }
    let jit_time = start.elapsed();
    
    println!("JIT evaluation (1M calls): {jit_time:?}");
    println!("Average per call: {:.2} ns", jit_time.as_nanos() as f64 / 1_000_000.0);
    println!("Sum (to prevent optimization): {sum}");
    
    // Compare with native Rust
    let start = std::time::Instant::now();
    let mut sum_native = 0.0;
    for i in 0..1_000_000 {
        let x = i as f64;
        sum_native += x * x + 1.0;
    }
    let native_time = start.elapsed();
    
    println!("Native Rust (1M calls): {native_time:?}");
    println!("Average per call: {:.2} ns", native_time.as_nanos() as f64 / 1_000_000.0);
    println!("Sum (to prevent optimization): {sum_native}");
    
    let speedup_ratio = native_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    println!("JIT vs Native ratio: {speedup_ratio:.2}x");
    
    if speedup_ratio > 0.5 {
        println!("🎉 JIT performance is excellent! Very close to native speed.");
    } else {
        println!("📈 JIT has overhead but still provides good performance for complex expressions.");
    }
    
    println!("\n✨ Final Tagless + JIT = Zero-Cost Symbolic Mathematics! ✨");
    
    Ok(())
}

#[cfg(not(feature = "jit"))]
fn main() {
    println!("This example requires the 'jit' feature to be enabled.");
    println!("Run with: cargo run --example jit_final_tagless_demo --features jit");
} 