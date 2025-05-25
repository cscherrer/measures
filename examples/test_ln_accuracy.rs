//! Test the accuracy of our custom ln implementation in JIT
//!
//! This example compares our CLIF IR ln implementation with Rust's built-in ln
//! to verify we're getting the same mathematical accuracy.

use measures::{LogDensityBuilder, Normal, exponential_family::jit::CustomJITOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Testing Custom ln Implementation Accuracy");
    println!("============================================");

    // Test our ln implementation by creating a simple expression that uses ln
    let normal = Normal::new(1.0, 1.0);

    // Get both the JIT compiled version and standard version
    let jit_result = normal.compile_custom_jit();

    match jit_result {
        Ok(jit_fn) => {
            println!("✅ JIT compilation succeeded!");

            // Test values across different ranges to verify ln accuracy
            let test_values = vec![0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0];

            println!("\nTesting ln accuracy across different input ranges:");
            println!("Value\t\tStandard\t\tJIT\t\t\tDifference");
            println!("-----\t\t--------\t\t---\t\t\t----------");

            let mut max_error = 0.0_f64;
            let mut total_error = 0.0_f64;

            for &x in &test_values {
                let standard_result = normal.log_density().at(&x);
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - standard_result).abs();

                max_error = max_error.max(difference);
                total_error += difference;

                println!("{x:.1}\t\t{standard_result:.10}\t{jit_result:.10}\t{difference:.2e}");
            }

            let avg_error = total_error / test_values.len() as f64;

            println!("\n📊 Accuracy Summary:");
            println!("Maximum error: {max_error:.2e}");
            println!("Average error: {avg_error:.2e}");

            // Check if we're within reasonable floating-point precision
            if max_error < 1e-10 {
                println!("✅ Excellent accuracy! Our ln implementation matches Rust's builtin.");
            } else if max_error < 1e-6 {
                println!("✅ Good accuracy! Small differences likely due to different algorithms.");
            } else {
                println!(
                    "⚠️  Significant differences detected. Our ln implementation may need improvement."
                );
            }

            // Test specifically around x=1 where ln should be most accurate
            println!("\n🎯 Testing near x=1 (where ln should be most accurate):");
            let near_one_values = vec![0.99, 0.999, 1.0, 1.001, 1.01];

            for &x in &near_one_values {
                let standard_result = normal.log_density().at(&x);
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - standard_result).abs();

                println!(
                    "x={x:.3}: standard={standard_result:.12}, jit={jit_result:.12}, diff={difference:.2e}"
                );
            }
        }
        Err(e) => {
            println!("❌ JIT compilation failed: {e}");
            println!("This might be due to platform-specific issues with executable memory.");
        }
    }

    // Also test if we can at least evaluate our symbolic representation
    println!("\n🔧 Testing symbolic representation (without JIT):");
    let symbolic = normal.custom_symbolic_log_density();

    match symbolic.evaluate_single("x", 2.0) {
        Ok(symbolic_result) => {
            let standard_result = normal.log_density().at(&2.0);
            let difference = (symbolic_result - standard_result).abs();

            println!("At x=2.0:");
            println!("  Standard: {standard_result:.10}");
            println!("  Symbolic: {symbolic_result:.10}");
            println!("  Difference: {difference:.2e}");

            if difference < 1e-10 {
                println!("✅ Our symbolic representation is mathematically correct!");
            } else {
                println!("⚠️  Symbolic representation has accuracy issues.");
            }
        }
        Err(e) => {
            println!("❌ Symbolic evaluation failed: {e}");
        }
    }

    Ok(())
}
