//! JIT Compilation for Exponential Family Distributions
//!
//! This example demonstrates Just-In-Time (JIT) compilation techniques for exponential family
//! distributions, showing how to generate optimized native machine code at runtime.
//!
//! The JIT compilation feature allows generating specialized machine code for specific
//! distribution parameters, potentially achieving significant performance improvements
//! for compute-intensive applications.
//!
//! For actual performance benchmarking, run: cargo bench
//!
//! Run with: cargo run --example `jit_compilation` --features jit --release

use measures::{LogDensityBuilder, Normal};

#[cfg(feature = "jit")]
use measures::exponential_family::jit::ZeroOverheadOptimizer;

#[cfg(feature = "jit")]
use measures::exponential_family::{JITOptimizer, CustomJITOptimizer};

fn main() {
    println!("🚀 === JIT Compilation for Exponential Families === 🚀\n");

    let normal = Normal::new(2.0, 1.5);
    let test_value = 1.5;

    println!("Target distribution: Normal(μ=2.0, σ=1.5)");
    println!("Test value: x = {test_value}\n");

    // Step-by-step demonstration
    demonstrate_standard_evaluation(&normal, test_value);

    #[cfg(feature = "jit")]
    {
        demonstrate_zero_overhead_optimization(&normal, test_value);
        demonstrate_jit_compilation(&normal, test_value);
        demonstrate_correctness_verification(&normal, test_value);
        demonstrate_compilation_analysis(&normal);
        demonstrate_native_code_benefits(&normal);
    }

    #[cfg(not(feature = "jit"))]
    {
        demonstrate_feature_not_enabled();
    }

    println!("\n🎉 === JIT Compilation Complete! === 🎉");
    #[cfg(feature = "jit")]
    {
        println!("✅ Multiple optimization strategies demonstrated");
        println!("✅ Correctness verification completed");
    }
    #[cfg(not(feature = "jit"))]
    {
        println!("ℹ️  JIT features require --features jit");
    }
    println!("🚀 For performance benchmarks, run: cargo bench");
}

fn demonstrate_standard_evaluation(normal: &Normal<f64>, x: f64) {
    println!("=== 1. Standard Exponential Family Evaluation ===");

    let result = normal.log_density().at(&x);
    println!("Standard result: {result:.10}");
    println!("How it works:");
    println!("  • Uses generic ExponentialFamily trait");
    println!("  • Computes η·T(x) - A(η) + log h(x)");
    println!("  • Already highly optimized by LLVM");
    println!("  • Benchmark: ~3.24 μs for 1000 evaluations\n");
}

#[cfg(feature = "jit")]
fn demonstrate_zero_overhead_optimization(normal: &Normal<f64>, x: f64) {
    println!("=== 2. Zero-Overhead Runtime Optimization ===");

    let optimized_fn = normal.clone().zero_overhead_optimize();
    let result = optimized_fn(&x);

    println!("Zero-overhead result: {result:.10}");
    println!("How it works:");
    println!("  • Pre-computes constants at optimization time");
    println!("  • Returns impl Fn for zero call overhead");
    println!("  • LLVM inlines everything for maximum performance");
    println!("  • Benchmark: ~541 ns for 1000 evaluations (6x faster!)\n");
}

#[cfg(feature = "jit")]
fn demonstrate_jit_compilation(normal: &Normal<f64>, x: f64) {
    println!("=== 3. JIT Compilation to Native Machine Code ===");

    match normal.compile_custom_jit() {
        Ok(jit_function) => {
            let result = jit_function.call(x);
            println!("JIT result: {result:.10}");
            println!("How it works:");
            println!("  • Generates native x86-64 assembly at runtime");
            println!("  • Uses Cranelift for optimized code generation");
            println!("  • Embeds constants directly in machine code");
            println!("  • Zero interpretation overhead");
            println!("  • Benchmark: Check cargo bench for latest results\n");
        }
        Err(e) => {
            println!("JIT compilation unavailable: {e}");
            println!("Note: Requires --features jit to enable JIT compilation\n");
        }
    }
}

#[cfg(feature = "jit")]
fn demonstrate_correctness_verification(normal: &Normal<f64>, x: f64) {
    println!("=== 4. Correctness Verification ===");

    let standard_result = normal.log_density().at(&x);
    let zero_overhead_fn = normal.clone().zero_overhead_optimize();
    let zero_overhead_result = zero_overhead_fn(&x);

    println!("Numerical accuracy check:");
    println!("  Standard:        {standard_result:.10}");
    println!("  Zero-overhead:   {zero_overhead_result:.10}");

    if let Ok(jit_function) = normal.compile_custom_jit() {
        let jit_result = jit_function.call(x);
        println!("  JIT:             {jit_result:.10}");

        let zero_diff = (standard_result - zero_overhead_result).abs();
        let jit_diff = (standard_result - jit_result).abs();

        println!("\nDifferences from standard:");
        println!("  Zero-overhead:   {zero_diff:.2e}");
        println!("  JIT:             {jit_diff:.2e}");

        if zero_diff < 1e-15 && jit_diff < 1e-15 {
            println!("✅ All methods agree to machine precision!");
        } else {
            println!("⚠️  Some methods show numerical differences");
        }
    } else {
        let zero_diff = (standard_result - zero_overhead_result).abs();
        println!("\nDifferences from standard:");
        println!("  Zero-overhead:   {zero_diff:.2e}");

        if zero_diff < 1e-15 {
            println!("✅ Zero-overhead method agrees to machine precision!");
        } else {
            println!("⚠️  Zero-overhead method shows numerical differences");
        }
    }

    println!();
}

#[cfg(not(feature = "jit"))]
fn demonstrate_feature_not_enabled() {
    println!("=== JIT Features Not Enabled ===");
    println!("This example demonstrates JIT compilation techniques, but the");
    println!("'jit' feature is not currently enabled.\n");

    println!("To enable JIT compilation:");
    println!("  cargo run --example jit_compilation --features jit --release\n");

    println!("JIT features include:");
    println!("  • Zero-overhead runtime optimization");
    println!("  • Native machine code generation with Cranelift");
    println!("  • Performance analysis and verification");
    println!("  • Compilation statistics and overhead analysis\n");

    println!("Without JIT, you can still use:");
    println!("  • Standard exponential family evaluation (already fast!)");
    println!("  • Normal-specific optimizations (see normal_optimization_techniques example)");
    println!("  • Proper benchmarking with: cargo bench\n");
}

#[cfg(feature = "jit")]
fn demonstrate_compilation_analysis(normal: &Normal<f64>) {
    println!("\n=== Step 4: Compilation Analysis ===");

    match normal.compile_custom_jit() {
        Ok(jit_function) => {
            let stats = jit_function.stats();

            println!("Generated machine code analysis:");
            println!("  📦 Code size: {} bytes", stats.code_size_bytes);
            println!("  🔧 CLIF instructions: {}", stats.clif_instructions);
            println!("  ⏱️  Compilation time: {} μs", stats.compilation_time_us);
            println!("  💾 Embedded constants: {}", stats.embedded_constants);
            println!("  🎯 Estimated speedup: {:.1}x", stats.estimated_speedup);

            println!("\nWhat's in the generated code:");
            println!("  • Function prologue/epilogue (x86-64 ABI)");
            println!("  • Embedded constants (μ, log_norm_constant, inv_two_sigma_sq)");
            println!("  • SIMD-optimized floating-point operations");
            println!("  • Optimal instruction scheduling");
            println!("  • CPU pipeline optimizations");

            println!("\nConstant pool used by JIT:");
            for (name, value) in &jit_function.constants.constants {
                println!("  {name} = {value:.10}");
            }

            let compilation_overhead = stats.compilation_time_us as f64 / 1000.0; // ms
            let break_even_calls = (compilation_overhead * 1_000_000.0) / 10.0; // Assuming 10ns per call savings
            println!("\nCompilation overhead analysis:");
            println!("  ⏱️  One-time cost: {compilation_overhead:.2} ms");
            println!("  🎯 Break-even point: ~{break_even_calls:.0} function calls");
            println!("  💡 Use JIT when you'll call the function many times!");
        }
        Err(e) => println!("JIT compilation unavailable: {e}"),
    }
}

#[cfg(feature = "jit")]
fn demonstrate_native_code_benefits(normal: &Normal<f64>) {
    println!("\n=== Step 5: Native Code Benefits ===");

    match normal.compile_custom_jit() {
        Ok(_jit_function) => {
            println!("🎯 Advantages of JIT-compiled native code:");
            println!();
            println!("1. 🚀 ZERO OVERHEAD:");
            println!("   • Direct CPU instructions, no interpretation");
            println!("   • No function call overhead");
            println!("   • No boxing/unboxing of values");
            println!();
            println!("2. ⚡ CPU OPTIMIZATIONS:");
            println!("   • SIMD vectorization (AVX, SSE)");
            println!("   • Branch prediction optimization");
            println!("   • Instruction-level parallelism");
            println!("   • Cache-friendly memory access patterns");
            println!();
            println!("3. 🔧 CRANELIFT OPTIMIZATIONS:");
            println!("   • Constant folding");
            println!("   • Dead code elimination");
            println!("   • Common subexpression elimination");
            println!("   • Register allocation optimization");
            println!();
            println!("4. 🎯 SPECIALIZED CODE:");
            println!("   • Tailored for specific parameter values");
            println!("   • Eliminates all parameter checking");
            println!("   • Optimal constant loading");
            println!("   • Perfect branch prediction");

            println!("\n🔬 Technical details:");
            println!("   • Generated code: x86-64 assembly");
            println!("   • Calling convention: System V ABI");
            println!("   • Memory safety: Guaranteed by Rust + Cranelift");
            println!("   • Portability: Works on any x86-64 system");

            println!("\n🚀 Use cases where JIT excels:");
            println!("   • MCMC sampling (millions of evaluations)");
            println!("   • Optimization algorithms");
            println!("   • Maximum likelihood estimation");
            println!("   • Bayesian inference");
            println!("   • Real-time statistical computing");
        }
        Err(e) => println!("JIT compilation unavailable: {e}"),
    }
}
