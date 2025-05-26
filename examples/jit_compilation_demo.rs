//! JIT Compilation Demo
//!
//! Demonstrates the optimization system for exponential family distributions
//! using zero-overhead optimization and symbolic math.

use measures::LogDensityBuilder;
use measures::distributions::continuous::Normal;
use measures::exponential_family::jit::ZeroOverheadOptimizer;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let normal = Normal::new(2.0_f64, 1.5_f64);
    let test_x = 2.5_f64;
    let num_iterations = 1_000_000;

    // Standard evaluation (baseline)
    let start = Instant::now();
    let mut result_standard = 0.0;
    for _ in 0..num_iterations {
        result_standard = normal.log_density().at(&test_x);
    }
    let time_standard = start.elapsed();
    println!("Standard: {result_standard:.10} in {time_standard:?}");

    // Zero-overhead optimization
    let optimized_fn = normal.clone().zero_overhead_optimize();
    let start = Instant::now();
    let mut result_optimized = 0.0;
    for _ in 0..num_iterations {
        result_optimized = optimized_fn(&test_x);
    }
    let time_optimized = start.elapsed();
    let speedup_optimized = time_standard.as_nanos() as f64 / time_optimized.as_nanos() as f64;
    println!(
        "Zero-overhead: {result_optimized:.10} in {time_optimized:?} ({speedup_optimized:.1}x speedup)"
    );

    // Verify accuracy
    let error: f64 = (result_optimized - result_standard).abs();
    println!("Accuracy: error = {error:.2e}");

    // Demonstrate symbolic math functionality
    #[cfg(feature = "symbolic")]
    {
        use std::collections::HashMap;
        use symbolic_math::{Expr, jit::GeneralJITCompiler};

        // Create a simple mathematical expression: -0.5 * x^2
        let expr = Expr::Mul(
            Box::new(Expr::Const(-0.5)),
            Box::new(Expr::Pow(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );

        println!("Symbolic expression: -0.5 * x^2");

        #[cfg(feature = "jit")]
        {
            match GeneralJITCompiler::new() {
                Ok(compiler) => {
                    match compiler.compile_expression(
                        &expr,
                        &["x".to_string()],
                        &[],
                        &HashMap::new(),
                    ) {
                        Ok(jit_function) => {
                            let symbolic_result = jit_function.call_single(test_x);
                            println!("Symbolic JIT result: {symbolic_result:.10}");
                        }
                        Err(e) => {
                            println!("Symbolic JIT compilation failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("JIT compiler creation failed: {e}");
                }
            }
        }
    }

    #[cfg(not(feature = "symbolic"))]
    {
        println!("Symbolic feature not enabled. Run with --features symbolic,jit for full demo");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_runs() {
        // Just verify the demo can run without panicking
        let result = main();
        assert!(result.is_ok());
    }
}
