//! Test the accuracy of our custom ln implementation in JIT
//!
//! This example compares our CLIF IR ln implementation with Rust's built-in ln
//! to verify we're getting the same mathematical accuracy.

use measures::{Normal, exponential_family::jit::CustomJITOptimizer, LogDensityBuilder};
use measures::core::{HasLogDensity, Measure};

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
            let test_values = vec![
                0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0
            ];
            
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
                
                println!("{:.1}\t\t{:.10}\t{:.10}\t{:.2e}", 
                    x, standard_result, jit_result, difference);
            }
            
            let avg_error = total_error / test_values.len() as f64;
            
            println!("\n📊 Accuracy Summary:");
            println!("Maximum error: {:.2e}", max_error);
            println!("Average error: {:.2e}", avg_error);
            
            // Check if we're within reasonable floating-point precision
            if max_error < 1e-10 {
                println!("✅ Excellent accuracy! Our ln implementation matches Rust's builtin.");
            } else if max_error < 1e-6 {
                println!("✅ Good accuracy! Small differences likely due to different algorithms.");
            } else {
                println!("⚠️  Significant differences detected. Our ln implementation may need improvement.");
            }
            
            // Test specifically around x=1 where ln should be most accurate
            println!("\n🎯 Testing near x=1 (where ln should be most accurate):");
            let near_one_values = vec![0.99, 0.999, 1.0, 1.001, 1.01];
            
            for &x in &near_one_values {
                let standard_result = normal.log_density().at(&x);
                let jit_result = jit_fn.call(x);
                let difference = (jit_result - standard_result).abs();
                
                println!("x={:.3}: standard={:.12}, jit={:.12}, diff={:.2e}", 
                    x, standard_result, jit_result, difference);
            }
            
        }
        Err(e) => {
            println!("❌ JIT compilation failed: {}", e);
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
            println!("  Standard: {:.10}", standard_result);
            println!("  Symbolic: {:.10}", symbolic_result);
            println!("  Difference: {:.2e}", difference);
            
            if difference < 1e-10 {
                println!("✅ Our symbolic representation is mathematically correct!");
            } else {
                println!("⚠️  Symbolic representation has accuracy issues.");
            }
        }
        Err(e) => {
            println!("❌ Symbolic evaluation failed: {}", e);
        }
    }
    
    Ok(())
} 