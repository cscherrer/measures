//! JIT Compilation Demo
//!
//! This example demonstrates the complete JIT compilation system for exponential
//! family distributions using our custom symbolic IR and Cranelift.
//!
//! The system provides three levels of optimization:
//! 1. Standard evaluation (baseline)
//! 2. Zero-overhead optimization (compile-time specialization)
//! 3. JIT compilation (runtime code generation)

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::Normal;
use measures::exponential_family::jit::{CustomJITOptimizer, ZeroOverheadOptimizer};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 JIT Compilation Demo for Exponential Family Distributions");
    println!("==============================================================\n");

    // Create a Normal distribution
    let normal = Normal::new(2.0, 1.5);
    println!("📊 Distribution: Normal(μ=2.0, σ=1.5)");
    println!("🎯 Test point: x = 2.5\n");

    let test_x = 2.5;
    let num_iterations = 1_000_000;

    // 1. Standard evaluation (baseline)
    println!("1️⃣  Standard Evaluation");
    println!("   ─────────────────────");
    let start = Instant::now();
    let mut result_standard = 0.0;
    for _ in 0..num_iterations {
        result_standard = normal.log_density().at(&test_x);
    }
    let time_standard = start.elapsed();
    println!("   Result: {result_standard:.10}");
    println!(
        "   Time:   {:?} ({:.2} ns/call)",
        time_standard,
        time_standard.as_nanos() as f64 / f64::from(num_iterations)
    );
    println!();

    // 2. Zero-overhead optimization
    println!("2️⃣  Zero-Overhead Optimization");
    println!("   ──────────────────────────");
    let optimized_fn = normal.clone().zero_overhead_optimize();
    let start = Instant::now();
    let mut result_optimized = 0.0;
    for _ in 0..num_iterations {
        result_optimized = optimized_fn(&test_x);
    }
    let time_optimized = start.elapsed();
    let speedup_optimized = time_standard.as_nanos() as f64 / time_optimized.as_nanos() as f64;
    println!("   Result:  {result_optimized:.10}");
    println!(
        "   Time:    {:?} ({:.2} ns/call)",
        time_optimized,
        time_optimized.as_nanos() as f64 / f64::from(num_iterations)
    );
    println!("   Speedup: {speedup_optimized:.1}x");
    println!();

    // 3. Custom symbolic IR demonstration
    println!("3️⃣  Custom Symbolic IR");
    println!("   ───────────────────");
    let symbolic = normal.custom_symbolic_log_density();
    println!("   Expression: {}", symbolic.expression);
    println!("   Variables:  {:?}", symbolic.expression.variables());
    println!("   Parameters: {:?}", symbolic.parameters);
    println!(
        "   Complexity: {} operations",
        symbolic.expression.complexity()
    );

    // Test symbolic evaluation
    let symbolic_result = symbolic.evaluate_single("x", test_x)?;
    println!("   Symbolic result: {symbolic_result:.10}");
    println!();

    // 4. JIT compilation
    #[cfg(feature = "jit")]
    {
        println!("4️⃣  JIT Compilation");
        println!("   ───────────────");

        match normal.compile_custom_jit() {
            Ok(jit_function) => {
                println!("   ✅ JIT compilation successful!");
                println!("   📝 Source: {}", jit_function.source_expression);

                let stats = jit_function.stats();
                println!("   📊 Compilation Stats:");
                println!("      • Code size: {} bytes", stats.code_size_bytes);
                println!("      • CLIF instructions: {}", stats.clif_instructions);
                println!("      • Compilation time: {} μs", stats.compilation_time_us);
                println!("      • Embedded constants: {}", stats.embedded_constants);
                println!("      • Estimated speedup: {:.1}x", stats.estimated_speedup);

                // Benchmark JIT function
                let start = Instant::now();
                let mut result_jit = 0.0;
                for _ in 0..num_iterations {
                    result_jit = jit_function.call(test_x);
                }
                let time_jit = start.elapsed();
                let speedup_jit = time_standard.as_nanos() as f64 / time_jit.as_nanos() as f64;

                println!("   🏃 Performance:");
                println!("      • Result: {result_jit:.10}");
                println!(
                    "      • Time: {:?} ({:.2} ns/call)",
                    time_jit,
                    time_jit.as_nanos() as f64 / f64::from(num_iterations)
                );
                println!("      • Speedup vs standard: {speedup_jit:.1}x");
                println!(
                    "      • Speedup vs zero-overhead: {:.1}x",
                    time_optimized.as_nanos() as f64 / time_jit.as_nanos() as f64
                );

                // Verify correctness
                let error = (result_jit - result_standard).abs();
                println!("   ✓ Accuracy: error = {error:.2e}");

                if error < 1e-10 {
                    println!("   🎉 JIT result matches standard evaluation!");
                } else {
                    println!("   ⚠️  JIT result differs from standard evaluation");
                }
            }
            Err(e) => {
                println!("   ❌ JIT compilation failed: {e}");
                println!("   💡 This is expected if libm functions are not properly linked");
            }
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        println!("4️⃣  JIT Compilation");
        println!("   ───────────────");
        println!("   ⚠️  JIT feature not enabled. Run with --features jit to see JIT compilation.");
    }

    println!("\n🎯 Summary");
    println!("   ───────");
    println!("   • Standard evaluation provides the baseline");
    println!("   • Zero-overhead optimization eliminates runtime dispatch");
    println!("   • JIT compilation generates native machine code for maximum performance");
    println!("   • Our custom symbolic IR enables full expression introspection");
    println!("   • Cranelift provides safe, fast code generation");

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
