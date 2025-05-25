//! Inline JIT Optimization Demo
//!
//! This example demonstrates the new inline JIT optimization that eliminates
//! function pointer overhead by using inlinable closures with embedded constants.
//!
//! The inline JIT approach should significantly reduce the overhead compared to
//! the standard JIT compilation while maintaining the benefits of optimization.
//!
//! Run with: cargo run --example inline_jit_demo --features jit --release

use measures::core::LogDensityBuilder;
use measures::distributions::continuous::Normal;
use measures::exponential_family::jit::{CustomJITOptimizer, StaticInlineJITOptimizer, ZeroOverheadOptimizer};
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Inline JIT Optimization Demo");
    println!("================================\n");

    // Create a Normal distribution
    let normal = Normal::new(2.0, 1.5);
    println!("📊 Distribution: Normal(μ=2.0, σ=1.5)");
    println!("🎯 Test point: x = 2.5\n");

    let test_x = 2.5;
    let num_iterations = 1_000_000;

    // Generate varying inputs to prevent constant folding
    let test_inputs: Vec<f64> = (0..num_iterations)
        .map(|i| 2.5 + (i as f64) * 0.001 / num_iterations as f64)
        .collect();

    // 1. Standard evaluation (baseline)
    println!("1️⃣  Standard Evaluation");
    println!("   ─────────────────────");
    let start = Instant::now();
    let mut result_standard = 0.0;
    for &x in &test_inputs {
        result_standard += black_box(normal.log_density().at(&black_box(x)));
    }
    let time_standard = start.elapsed();
    println!("   Result sum: {result_standard:.10}");
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
    for &x in &test_inputs {
        result_optimized += black_box(optimized_fn(&black_box(x)));
    }
    let time_optimized = start.elapsed();
    let speedup_optimized = time_standard.as_nanos() as f64 / time_optimized.as_nanos() as f64;
    println!("   Result sum: {result_optimized:.10}");
    println!(
        "   Time:   {:?} ({:.2} ns/call)",
        time_optimized,
        time_optimized.as_nanos() as f64 / f64::from(num_iterations)
    );
    println!("   Speedup vs standard: {speedup_optimized:.2}x");
    println!();

    // 3. Inline JIT optimization (NEW!)
    println!("3️⃣  Static Inline JIT Optimization (NEW!)");
    println!("   ──────────────────────────────────────");
    match normal.compile_static_inline_jit() {
        Ok(static_inline_jit_function) => {
            println!("   ✅ Static Inline JIT compilation successful!");
            println!("   📝 Source: {}", static_inline_jit_function.source_expression);

            let stats = static_inline_jit_function.stats();
            println!("   📊 Compilation Stats:");
            println!("      • Code size: {} bytes", stats.code_size_bytes);
            println!("      • CLIF instructions: {}", stats.clif_instructions);
            println!("      • Compilation time: {} μs", stats.compilation_time_us);
            println!("      • Embedded constants: {}", stats.embedded_constants);
            println!("      • Estimated speedup: {:.1}x", stats.estimated_speedup);

            // Benchmark static inline JIT function
            let start = Instant::now();
            let mut result_static_inline_jit = 0.0;
            for &x in &test_inputs {
                result_static_inline_jit += black_box(static_inline_jit_function.call(black_box(x)));
            }
            let time_static_inline_jit = start.elapsed();
            let speedup_static_inline_jit = time_standard.as_nanos() as f64 / time_static_inline_jit.as_nanos() as f64;

            println!("   🏃 Performance:");
            println!("      • Result sum: {result_static_inline_jit:.10}");
            println!(
                "      • Time: {:?} ({:.2} ns/call)",
                time_static_inline_jit,
                time_static_inline_jit.as_nanos() as f64 / f64::from(num_iterations)
            );
            println!("      • Speedup vs standard: {speedup_static_inline_jit:.2}x");
            println!(
                "      • Speedup vs zero-overhead: {:.2}x",
                time_optimized.as_nanos() as f64 / time_static_inline_jit.as_nanos() as f64
            );

            // Verify correctness (compare averages)
            let avg_static = result_static_inline_jit / num_iterations as f64;
            let avg_standard = result_standard / num_iterations as f64;
            let error = (avg_static - avg_standard).abs();
            println!("      • Accuracy: {error:.2e} error");
            println!();
        }
        Err(e) => {
            println!("   ❌ Static Inline JIT compilation failed: {e}");
            println!();
        }
    }

    // 4. Standard JIT compilation (for comparison)
    #[cfg(feature = "jit")]
    {
        println!("4️⃣  Standard JIT Compilation");
        println!("   ──────────────────────────");

        match normal.compile_custom_jit() {
            Ok(jit_function) => {
                println!("   ✅ Standard JIT compilation successful!");
                println!("   📝 Source: {}", jit_function.source_expression);

                let stats = jit_function.stats();
                println!("   📊 Compilation Stats:");
                println!("      • Code size: {} bytes", stats.code_size_bytes);
                println!("      • CLIF instructions: {}", stats.clif_instructions);
                println!("      • Compilation time: {} μs", stats.compilation_time_us);
                println!("      • Embedded constants: {}", stats.embedded_constants);
                println!("      • Estimated speedup: {:.1}x", stats.estimated_speedup);

                // Benchmark standard JIT function
                let start = Instant::now();
                let mut result_jit = 0.0;
                for &x in &test_inputs {
                    result_jit += black_box(jit_function.call(black_box(x)));
                }
                let time_jit = start.elapsed();
                let speedup_jit = time_standard.as_nanos() as f64 / time_jit.as_nanos() as f64;

                println!("   🏃 Performance:");
                println!("      • Result sum: {result_jit:.10}");
                println!(
                    "      • Time: {:?} ({:.2} ns/call)",
                    time_jit,
                    time_jit.as_nanos() as f64 / f64::from(num_iterations)
                );
                println!("      • Speedup vs standard: {speedup_jit:.2}x");

                // Verify correctness (compare averages)
                let avg_jit = result_jit / num_iterations as f64;
                let avg_standard = result_standard / num_iterations as f64;
                let error = (avg_jit - avg_standard).abs();
                println!("      • Accuracy: {error:.2e} error");
                println!();
            }
            Err(e) => {
                println!("   ❌ Standard JIT compilation failed: {e}");
                println!();
            }
        }
    }

    // Summary
    println!("📈 Performance Summary");
    println!("=====================");
    println!("The static inline JIT optimization eliminates ALL function call overhead");
    println!("by using static dispatch with embedded constants - no heap allocation!");
    println!();
    println!("Expected improvements:");
    println!("• 🎯 Static Inline JIT should match or beat zero-overhead performance");
    println!("• 🎯 Much faster than Box<dyn Fn> approach (eliminates vtable lookup)");
    println!("• 🎯 Much faster than standard JIT (eliminates function pointer overhead)");
    println!("• 🎯 Perfect accuracy maintained");
    println!("• 🎯 Zero compilation overhead (just closure creation)");
    println!();
    println!("This demonstrates true zero-overhead abstractions in Rust!");

    Ok(())
} 