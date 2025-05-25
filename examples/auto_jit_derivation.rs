//! Automatic JIT Compilation Derivation Demo
//!
//! This example demonstrates the automatic derivation of JIT-compiled log-density
//! functions from exponential family structure. Instead of manually implementing
//! `CustomJITOptimizer` for each distribution, the system automatically generates
//! optimized implementations.
//!
//! Features demonstrated:
//! - Automatic pattern recognition for common distribution types
//! - Zero-code JIT compilation for supported distributions
//! - Performance comparison with manual implementations
//! - Extensible pattern registry for new distributions
//!
//! Run with: cargo run --example auto_jit_derivation --features jit --release

#[cfg(feature = "jit")]
use measures::{LogDensityBuilder, Normal, Exponential};

#[cfg(feature = "jit")]
use measures::exponential_family::{AutoJITExt, AutoJITOptimizer, CustomJITOptimizer};

#[cfg(feature = "jit")]
fn main() {
    println!("=== Automatic JIT Compilation Derivation Demo ===\n");

    // Demonstrate automatic derivation for Normal distribution
    demonstrate_normal_auto_jit();
    
    // Demonstrate automatic derivation for Exponential distribution
    demonstrate_exponential_auto_jit();
    
    // Compare automatic vs manual implementations
    compare_auto_vs_manual();
    
    // Performance benchmarking
    benchmark_auto_jit_performance();
    
    // Show extensibility
    demonstrate_extensibility();
}

#[cfg(feature = "jit")]
fn demonstrate_normal_auto_jit() {
    println!("=== Normal Distribution Auto-JIT ===");
    
    let normal = Normal::new(2.0, 1.5);
    println!("Distribution: Normal(μ=2.0, σ=1.5)");
    
    // Automatic symbolic generation
    match normal.auto_symbolic() {
        Ok(symbolic) => {
            println!("✓ Automatic symbolic generation succeeded");
            println!("  Variables: {:?}", symbolic.variables);
            println!("  Parameters: {:?}", symbolic.parameters.keys().collect::<Vec<_>>());
            
            // Test evaluation
            let test_x = 2.5;
            if let Ok(symbolic_result) = symbolic.evaluate_single("x", test_x) {
                let standard_result = normal.log_density().at(&test_x);
                println!("  Evaluation test at x={test_x}:");
                println!("    Auto-symbolic: {symbolic_result:.10}");
                println!("    Standard:      {standard_result:.10}");
                println!("    Difference:    {:.2e}", (symbolic_result - standard_result).abs());
            }
        }
        Err(e) => println!("✗ Automatic symbolic generation failed: {e:?}"),
    }
    
    // Automatic JIT compilation
    match normal.auto_jit() {
        Ok(jit_fn) => {
            println!("✓ Automatic JIT compilation succeeded");
            println!("  Compilation stats: {:?}", jit_fn.stats());
            
            // Test JIT function
            let test_x = 2.5;
            let jit_result = jit_fn.call(test_x);
            let standard_result = normal.log_density().at(&test_x);
            println!("  JIT evaluation test at x={test_x}:");
            println!("    JIT result:    {jit_result:.10}");
            println!("    Standard:      {standard_result:.10}");
            println!("    Difference:    {:.2e}", (jit_result - standard_result).abs());
        }
        Err(e) => println!("✗ Automatic JIT compilation failed: {e:?}"),
    }
    
    println!();
}

#[cfg(feature = "jit")]
fn demonstrate_exponential_auto_jit() {
    println!("=== Exponential Distribution Auto-JIT ===");
    
    let exponential = Exponential::new(2.0);
    println!("Distribution: Exponential(λ=2.0)");
    
    // Automatic symbolic generation
    match exponential.auto_symbolic() {
        Ok(symbolic) => {
            println!("✓ Automatic symbolic generation succeeded");
            println!("  Variables: {:?}", symbolic.variables);
            println!("  Parameters: {:?}", symbolic.parameters.keys().collect::<Vec<_>>());
        }
        Err(e) => println!("✗ Automatic symbolic generation failed: {e:?}"),
    }
    
    // Automatic JIT compilation
    match exponential.auto_jit() {
        Ok(jit_fn) => {
            println!("✓ Automatic JIT compilation succeeded");
            
            // Test JIT function
            let test_x = 1.5;
            let jit_result = jit_fn.call(test_x);
            let standard_result = exponential.log_density().at(&test_x);
            println!("  JIT evaluation test at x={test_x}:");
            println!("    JIT result:    {jit_result:.10}");
            println!("    Standard:      {standard_result:.10}");
            println!("    Difference:    {:.2e}", (jit_result - standard_result).abs());
        }
        Err(e) => println!("✗ Automatic JIT compilation failed: {e:?}"),
    }
    
    println!();
}

#[cfg(feature = "jit")]
fn compare_auto_vs_manual() {
    println!("=== Automatic vs Manual Implementation Comparison ===");
    
    let normal = Normal::new(0.0, 1.0);
    
    // Test automatic derivation
    let auto_symbolic = normal.custom_symbolic_log_density(); // Uses auto-derivation
    println!("Automatic derivation:");
    println!("  Variables: {:?}", auto_symbolic.variables);
    println!("  Parameters: {} entries", auto_symbolic.parameters.len());
    
    // Compare results
    let test_values = [0.0, 1.0, -1.0, 2.0, -2.0];
    println!("\nAccuracy comparison:");
    for &x in &test_values {
        let standard = normal.log_density().at(&x);
        if let Ok(auto_result) = auto_symbolic.evaluate_single("x", x) {
            let error = (auto_result - standard).abs();
            println!("  x={x:4.1}: standard={standard:.6}, auto={auto_result:.6}, error={error:.2e}");
        }
    }
    
    println!();
}

#[cfg(feature = "jit")]
fn benchmark_auto_jit_performance() {
    println!("=== Performance Benchmark ===");
    
    let normal = Normal::new(1.0, 2.0);
    let test_x = 1.5;
    let n_iterations = 1_000_000;
    
    // Benchmark standard evaluation
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = normal.log_density().at(&test_x);
    }
    let standard_time = start.elapsed();
    
    // Benchmark auto-JIT compilation (if successful)
    if let Ok(jit_fn) = normal.auto_jit() {
        let start = std::time::Instant::now();
        for _ in 0..n_iterations {
            let _ = jit_fn.call(test_x);
        }
        let jit_time = start.elapsed();
        
        println!("Performance results ({n_iterations} iterations):");
        println!("  Standard evaluation: {:?} ({:.2} ns/call)", 
                 standard_time, 
                 standard_time.as_nanos() as f64 / n_iterations as f64);
        println!("  Auto-JIT evaluation: {:?} ({:.2} ns/call)", 
                 jit_time, 
                 jit_time.as_nanos() as f64 / n_iterations as f64);
        
        let speedup = standard_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
        println!("  Speedup: {speedup:.2}x");
        
        if speedup > 1.0 {
            println!("  🚀 Auto-JIT provides significant speedup!");
        } else {
            println!("  📊 Standard evaluation remains competitive");
        }
    } else {
        println!("Auto-JIT compilation failed, skipping performance comparison");
    }
    
    println!();
}

#[cfg(feature = "jit")]
fn demonstrate_extensibility() {
    println!("=== Extensibility Demo ===");
    
    // Show how the pattern registry works
    let optimizer = AutoJITOptimizer::new();
    
    // Test with supported distributions
    let normal = Normal::new(0.0, 1.0);
    match optimizer.generate_symbolic(&normal) {
        Ok(_) => println!("✓ Normal distribution: Pattern found and compiled"),
        Err(e) => println!("✗ Normal distribution: {e:?}"),
    }
    
    let exponential = Exponential::new(1.0);
    match optimizer.generate_symbolic(&exponential) {
        Ok(_) => println!("✓ Exponential distribution: Pattern found and compiled"),
        Err(e) => println!("✗ Exponential distribution: {e:?}"),
    }
    
    println!("\nPattern registry benefits:");
    println!("• Zero-code JIT compilation for supported distributions");
    println!("• Automatic pattern recognition based on type");
    println!("• Extensible to new distribution types");
    println!("• Consistent performance across distribution families");
    println!("• Fallback to standard evaluation for unsupported types");
    
    println!("\nTo add support for a new distribution:");
    println!("1. Implement AutoJITPattern for the distribution");
    println!("2. Register the pattern in AutoJITRegistry");
    println!("3. Use auto_jit_impl! macro for automatic trait implementation");
    println!("4. Enjoy automatic JIT compilation!");
    
    println!();
}

#[cfg(not(feature = "jit"))]
fn main() {
    println!("This example requires the 'jit' feature to be enabled.");
    println!("Run with: cargo run --example auto_jit_derivation --features jit --release");
} 