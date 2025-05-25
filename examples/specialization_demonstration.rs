//! Specialization Demonstration
//!
//! This example shows how specialization would enable automatic optimization
//! for exponential family relative density computation. Currently, we use
//! zero-overhead optimization as the best available alternative.
//!
//! Run with: cargo run --example `specialization_demonstration` --features jit

use measures::exponential_family::jit::ZeroOverheadOptimizer;
use measures::{LogDensityBuilder, Normal};

fn main() {
    println!("🔬 === Zero-Overhead Optimization (Current Best Solution) === 🔬\n");

    let normal1 = Normal::new(0.0, 1.0);
    let normal2 = Normal::new(1.0, 1.5);
    let x = 0.5;

    // Current approach: Zero-overhead optimization
    println!("🚀 Current Approach: Zero-Overhead Optimization");
    println!("   Uses: zero_overhead_optimize_wrt() optimization");
    println!("   API:  normal1.zero_overhead_optimize_wrt(normal2)");
    println!();

    let standard_result: f64 = normal1.log_density().wrt(normal2.clone()).at(&x);
    let optimized_fn = normal1.clone().zero_overhead_optimize_wrt(normal2.clone());
    let optimized_result: f64 = optimized_fn(&x);

    println!("Standard computation:  {standard_result:.10}");
    println!("Zero-overhead result:  {optimized_result:.10}");
    println!(
        "Difference:            {:.2e}",
        (standard_result - optimized_result).abs()
    );

    println!("\n✨ === How Specialization Would Improve This === ✨");
    println!();
    println!("🎯 With Rust specialization (when available):");
    println!("   • Automatic optimization in builder pattern");
    println!("   • No need for explicit optimization calls");
    println!("   • Zero-cost abstractions maintained");
    println!();
    println!("📝 Hypothetical specialized API:");
    println!("   normal1.log_density().wrt(normal2).at(&x)  // Auto-optimized!");
    println!();
    println!("🔧 Current workaround:");
    println!("   • Use zero-overhead optimization for performance");
    println!("   • Builder pattern for convenience");
    println!("   • Manual selection between approaches");

    println!("\n🎉 === Current State Summary === 🎉");
    println!("✅ Zero-overhead optimization provides excellent performance");
    println!("✅ Builder pattern provides consistent API");
    println!("⏳ Automatic dispatch awaits Rust specialization");
    println!("🚀 Performance is excellent with current approach!");
}
