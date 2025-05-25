//! JIT Compilation Demo
//!
//! Demonstrates the JIT compilation system for exponential family distributions
//! using custom symbolic IR and Cranelift.

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::Normal;
use measures::exponential_family::jit::{CustomJITOptimizer, ZeroOverheadOptimizer};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let normal = Normal::new(2.0, 1.5);
    let test_x = 2.5;
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

    // Custom symbolic IR
    let symbolic = normal.custom_symbolic_log_density();
    let symbolic_result = symbolic.evaluate_single("x", test_x)?;
    println!(
        "Symbolic: {:.10} (complexity: {})",
        symbolic_result,
        symbolic.expression.complexity()
    );

    // JIT compilation
    #[cfg(feature = "jit")]
    {
        match normal.compile_custom_jit() {
            Ok(jit_function) => {
                let stats = jit_function.stats();
                println!(
                    "JIT compilation: {} bytes, {} CLIF instructions",
                    stats.code_size_bytes, stats.clif_instructions
                );

                let start = Instant::now();
                let mut result_jit = 0.0;
                for _ in 0..num_iterations {
                    result_jit = jit_function.call(test_x);
                }
                let time_jit = start.elapsed();
                let speedup_jit = time_standard.as_nanos() as f64 / time_jit.as_nanos() as f64;

                println!("JIT: {result_jit:.10} in {time_jit:?} ({speedup_jit:.1}x speedup)");

                let error = (result_jit - result_standard).abs();
                println!("Accuracy: error = {error:.2e}");
            }
            Err(e) => {
                println!("JIT compilation failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        println!("JIT feature not enabled. Run with --features jit");
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
