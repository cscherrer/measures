//! Final Tagless Approach Showcase
//!
//! This example demonstrates the power of the final tagless approach for symbolic computation.
//! It shows how the same expression definition can be interpreted in multiple ways:
//! - Direct evaluation for maximum performance
//! - JIT compilation for native code generation
//!
//! The final tagless approach solves the expression problem and provides zero-cost abstractions.

use std::time::Instant;
use symbolic_math::final_tagless::{DirectEval, MathExpr, StatisticalExpr};

#[cfg(feature = "jit")]
use symbolic_math::final_tagless::JITEval;

/// A simple linear function for performance testing
fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    let two = E::constant(2.0);
    let three = E::constant(3.0);

    // 2*x + 3
    E::add(E::mul(two, x), three)
}

/// A more complex expression demonstrating various operations
fn complex_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
where
    E::Repr<f64>: Clone,
{
    // ln(exp(x) + sqrt(x^2 + 1))
    E::ln(E::add(
        E::exp(x.clone()),
        E::sqrt(E::add(E::pow(x, E::constant(2.0)), E::constant(1.0))),
    ))
}

/// Demonstrate statistical functions using extension traits
fn statistical_example<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    // softplus(logistic(x))
    E::softplus(E::logistic(x))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Final Tagless Symbolic Math Showcase");
    println!("========================================");

    // 1. Direct Evaluation Example
    println!("\n=== 1. Direct Evaluation (Zero-cost) ===");
    let start = Instant::now();
    let result_direct = linear::<DirectEval>(DirectEval::var("x", 2.0));
    let direct_time = start.elapsed();
    println!("linear(2.0) = {result_direct}");
    println!("Time: {direct_time:?}");

    // 2. Complex Expression Example
    println!("\n=== 2. Complex Expression ===");
    let start = Instant::now();
    let complex_result = complex_expr::<DirectEval>(DirectEval::var("x", 1.0));
    let complex_time = start.elapsed();
    println!("complex_expr(1.0) = {complex_result}");
    println!("Time: {complex_time:?}");

    // 3. Statistical Extensions
    println!("\n=== 3. Statistical Extensions ===");
    let start = Instant::now();
    let stat_result = statistical_example::<DirectEval>(DirectEval::var("x", 0.0));
    let stat_time = start.elapsed();
    println!("statistical_example(0.0) = {stat_result}");
    println!("Time: {stat_time:?}");

    // 4. JIT Compilation Example
    #[cfg(feature = "jit")]
    {
        println!("\n=== 4. JIT Compilation ===");

        // Compile linear function
        let jit_expr = linear::<JITEval>(JITEval::var("x"));
        let start = Instant::now();
        let compiled = JITEval::compile_single_var(jit_expr, "x")?;
        let compile_time = start.elapsed();
        println!("JIT compilation time: {compile_time:?}");

        // Test the compiled function
        let start = Instant::now();
        let jit_result = compiled.call_single(2.0);
        let jit_time = start.elapsed();
        println!("JIT linear(2.0) = {jit_result}");
        println!("JIT execution time: {jit_time:?}");

        // Compile complex expression
        let complex_jit_expr = complex_expr::<JITEval>(JITEval::var("x"));
        let complex_compiled = JITEval::compile_single_var(complex_jit_expr, "x")?;
        let complex_jit_result = complex_compiled.call_single(1.0);
        println!("JIT complex_expr(1.0) = {complex_jit_result}");

        // Verify results match
        println!("Direct vs JIT linear: {result_direct} vs {jit_result}");
        println!("Direct vs JIT complex: {complex_result} vs {complex_jit_result}");
    }

    // 5. Performance Comparison
    println!("\n=== 5. Performance Comparison ===");

    // Benchmark DirectEval (should be very fast)
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..10000 {
        sum += linear::<DirectEval>(DirectEval::var("x", f64::from(i)));
    }
    let direct_time = start.elapsed();
    println!("DirectEval (10k iterations): {direct_time:?}, sum: {sum}");

    #[cfg(feature = "jit")]
    {
        // Benchmark JIT (should be even faster for repeated calls)
        let jit_expr = linear::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x")?;

        let start = Instant::now();
        let mut jit_sum = 0.0;
        for i in 0..10000 {
            jit_sum += compiled.call_single(f64::from(i));
        }
        let jit_time = start.elapsed();
        println!("JIT (10k iterations): {jit_time:?}, sum: {jit_sum}");

        let speedup = direct_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
        println!("JIT speedup: {speedup:.2}x");
    }

    println!("\nFinal tagless approach demonstrates:");
    println!("✅ Zero-cost abstractions with DirectEval");
    #[cfg(feature = "jit")]
    println!("✅ Native code generation with JITEval");
    println!("✅ Easy extension (expression problem solved)");
    println!("✅ Type safety at compile time");
    println!("✅ Composable DSL components");

    // 6. Type Safety Demonstration
    println!("\n=== 6. Type Safety & Extensibility ===");
    println!("✅ Compile-time type checking");
    println!("✅ Zero runtime overhead for DirectEval");
    println!("✅ Easy addition of new operations (StatisticalExpr)");
    println!("✅ Easy addition of new interpreters");
    println!("✅ Solves the expression problem elegantly");
    println!("✅ Composable DSL components");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_tagless_direct_eval() {
        // Test that final tagless produces correct results
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::mul(E::constant(2.0), x), E::constant(1.0))
        }

        let result = test_expr::<DirectEval>(DirectEval::var("x", 5.0));
        assert_eq!(result, 11.0); // 2*5 + 1 = 11
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_final_tagless_jit_eval() {
        // Test that JIT produces correct results
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::mul(E::constant(2.0), x), E::constant(1.0))
        }

        let jit_expr = test_expr::<JITEval>(JITEval::var("x"));
        let compiled = JITEval::compile_single_var(jit_expr, "x").unwrap();
        let result = compiled.call_single(5.0);
        assert_eq!(result, 11.0); // 2*5 + 1 = 11
    }

    #[test]
    fn test_statistical_extensions() {
        // Test statistical extension trait
        let result = statistical_example::<DirectEval>(DirectEval::var("x", 0.0));
        // At x=0, logistic(0) = 0.5, softplus(0.5) ≈ 0.974
        assert!((result - 0.9740769841801067).abs() < 1e-10);
    }
}
