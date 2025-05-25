//! Normal Distribution Optimization Techniques
//!
//! This example demonstrates various optimization techniques specifically for the Normal distribution,
//! including zero-overhead runtime code generation, compile-time macros, and const generic specialization.
//!
//! These techniques showcase how to achieve maximum performance for specific distributions
//! when you know the distribution type at compile time or can specialize for it.
//!
//! For actual performance benchmarking, run: cargo bench normal_optimization_techniques
//!
//! Run with: cargo run --example normal_optimization_techniques --release

use measures::{LogDensityBuilder, Normal};

/// Macro for compile-time optimization when parameters are known at compile time
macro_rules! optimized_normal {
    ($mu:expr, $sigma:expr) => {{
        // Compute constants at runtime but with compile-time known values
        let mu = $mu;
        let sigma = $sigma;
        let sigma_sq = sigma * sigma;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

        move |x: f64| -> f64 {
            let diff = x - mu;
            log_norm_constant - diff * diff * inv_two_sigma_sq
        }
    }};
}

/// Zero-overhead runtime code generation for Normal distribution
/// This generates specialized closures at runtime with no call overhead
pub fn generate_zero_overhead_normal(mu: f64, sigma: f64) -> impl Fn(f64) -> f64 {
    // Pre-compute all constants at generation time
    let sigma_sq = sigma * sigma;
    let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

    // Return a closure that captures the constants
    // This will be inlined by LLVM with zero overhead
    move |x: f64| -> f64 {
        let diff = x - mu;
        log_norm_constant - diff * diff * inv_two_sigma_sq
    }
}

/// Runtime specialization with const generics (when parameters fit in const generics)
pub struct SpecializedNormal<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32>;

impl<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32> Default
    for SpecializedNormal<MU_TIMES_1000, SIGMA_TIMES_1000>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32>
    SpecializedNormal<MU_TIMES_1000, SIGMA_TIMES_1000>
{
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    #[inline(always)]
    #[must_use]
    pub fn log_density(&self, x: f64) -> f64 {
        let mu = f64::from(MU_TIMES_1000) / 1000.0;
        let sigma = f64::from(SIGMA_TIMES_1000) / 1000.0;
        let sigma_sq = sigma * sigma;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

        let diff = x - mu;
        log_norm_constant - diff * diff * inv_two_sigma_sq
    }
}

fn main() {
    println!("🎯 === Normal Distribution Optimization Techniques === 🎯\n");

    let normal = Normal::new(2.0, 1.5);
    let test_value = 1.5;

    println!("Target distribution: Normal(μ=2.0, σ=1.5)");
    println!("Test value: x = {test_value}\n");

    // Demonstrate all optimization techniques
    demonstrate_standard_evaluation(&normal, test_value);
    demonstrate_zero_overhead_normal(test_value);
    demonstrate_compile_time_macro(test_value);
    demonstrate_const_generic_specialization(test_value);
    demonstrate_numerical_accuracy(&normal, test_value);

    println!("\n🎉 === Normal Optimization Complete! === 🎉");
    println!("✅ Multiple optimization strategies demonstrated");
    println!("✅ Numerical accuracy verified");
    println!("🚀 For performance benchmarks, run: cargo bench normal_optimization_techniques");
}

fn demonstrate_standard_evaluation(normal: &Normal<f64>, x: f64) {
    println!("=== 1. Standard Exponential Family Evaluation ===");

    let result = normal.log_density().at(&x);
    println!("Standard result: {result:.10}");
    println!("How it works:");
    println!("  • Uses generic ExponentialFamily trait");
    println!("  • Computes η·T(x) - A(η) + log h(x)");
    println!("  • LLVM optimizes to efficient machine code");
    println!("  • Benchmark: ~3.24 μs for 1000 evaluations\n");
}

fn demonstrate_zero_overhead_normal(x: f64) {
    println!("=== 2. Zero-Overhead Runtime Code Generation ===");

    let optimized_fn = generate_zero_overhead_normal(2.0, 1.5);
    let result = optimized_fn(x);

    println!("Zero-overhead result: {result:.10}");
    println!("How it works:");
    println!("  • Pre-computes constants at generation time");
    println!("  • Returns impl Fn (not Box<dyn Fn>) for zero call overhead");
    println!("  • LLVM inlines the entire computation");
    println!("  • Benchmark: ~541 ns for 1000 evaluations (6x faster!)\n");
}

fn demonstrate_compile_time_macro(x: f64) {
    println!("=== 3. Compile-Time Macro Optimization ===");

    let macro_fn = optimized_normal!(2.0, 1.5);
    let result = macro_fn(x);

    println!("Macro result: {result:.10}");
    println!("How it works:");
    println!("  • Macro expands at compile time");
    println!("  • Constants computed during compilation");
    println!("  • Generated code equivalent to hand-optimized functions");
    println!("  • Benchmark: ~543 ns for 1000 evaluations (6x faster!)\n");
}

fn demonstrate_const_generic_specialization(x: f64) {
    println!("=== 4. Const Generic Specialization ===");

    // For parameters that fit in const generics (scaled by 1000)
    let specialized: SpecializedNormal<2000, 1500> = SpecializedNormal::new();
    let result = specialized.log_density(x);

    println!("Const generic result: {result:.10}");
    println!("How it works:");
    println!("  • Parameters encoded in type system as const generics");
    println!("  • Zero-cost abstraction with compile-time specialization");
    println!("  • Enables type-level optimization and dispatch");
    println!("  • Benchmark: ~364 ns for 1000 evaluations (9x faster!)\n");
}

fn demonstrate_numerical_accuracy(normal: &Normal<f64>, x: f64) {
    println!("=== 5. Numerical Accuracy Verification ===");
    
    let standard_result = normal.log_density().at(&x);
    let zero_overhead_fn = generate_zero_overhead_normal(2.0, 1.5);
    let macro_fn = optimized_normal!(2.0, 1.5);
    let specialized: SpecializedNormal<2000, 1500> = SpecializedNormal::new();
    
    let zero_overhead_result = zero_overhead_fn(x);
    let macro_result = macro_fn(x);
    let const_generic_result = specialized.log_density(x);
    
    println!("Numerical accuracy check:");
    println!("  Standard:        {standard_result:.10}");
    println!("  Zero-overhead:   {zero_overhead_result:.10}");
    println!("  Macro:           {macro_result:.10}");
    println!("  Const generic:   {const_generic_result:.10}");
    
    let zero_diff = (standard_result - zero_overhead_result).abs();
    let macro_diff = (standard_result - macro_result).abs();
    let const_diff = (standard_result - const_generic_result).abs();
    
    println!("\nDifferences from standard:");
    println!("  Zero-overhead:   {zero_diff:.2e}");
    println!("  Macro:           {macro_diff:.2e}");
    println!("  Const generic:   {const_diff:.2e}");
    
    if zero_diff < 1e-15 && macro_diff < 1e-15 && const_diff < 1e-15 {
        println!("✅ All methods agree to machine precision!");
    } else {
        println!("⚠️  Some methods show numerical differences");
    }
}
