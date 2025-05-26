//! JIT Final Tagless Demo - Enhanced Edition
//!
//! This demo showcases the enhanced final tagless JIT implementation with feature parity
//! to the existing JIT system, including multiple function signatures, compilation statistics,
//! and various call methods.

use std::time::Instant;
use symbolic_math::final_tagless::{JITEval, MathExpr};

fn main() {
    println!("🚀 Enhanced JIT Final Tagless Demo");
    println!("=====================================\n");

    demo_single_variable();
    demo_data_parameter();
    demo_data_parameters();
    demo_embedded_constants();
    demo_compilation_stats();
    demo_performance_comparison();
}

fn demo_single_variable() {
    println!("📊 Single Variable Functions: f(x) -> f64");
    println!("-------------------------------------------");

    // Polynomial: f(x) = 2x³ - 3x² + x + 1
    fn polynomial<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let two = E::constant(2.0);
        let three = E::constant(3.0);
        let one = E::constant(1.0);

        E::add(
            E::add(
                E::sub(
                    E::mul(two, E::pow(x.clone(), E::constant(3.0))),
                    E::mul(three, E::pow(x.clone(), E::constant(2.0))),
                ),
                x,
            ),
            one,
        )
    }

    let jit_expr = polynomial::<JITEval>(JITEval::var("x"));
    let compiled = JITEval::compile_single_var(jit_expr, "x").expect("Compilation should succeed");

    println!("Expression: 2x³ - 3x² + x + 1");
    println!("f(0) = {}", compiled.call_single(0.0));
    println!("f(1) = {}", compiled.call_single(1.0));
    println!("f(2) = {}", compiled.call_single(2.0));
    println!("Signature: {:?}\n", compiled.signature);
}

fn demo_data_parameter() {
    println!("📈 Data + Parameter Functions: f(x, θ) -> f64");
    println!("----------------------------------------------");

    // Linear model: f(x, θ) = θ * x + 1
    fn linear_model<E: MathExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
        E::add(E::mul(theta, x), E::constant(1.0))
    }

    let jit_expr = linear_model::<JITEval>(JITEval::var("x"), JITEval::var("theta"));
    let compiled =
        JITEval::compile_data_param(jit_expr, "x", "theta").expect("Compilation should succeed");

    println!("Expression: θ * x + 1");
    println!("f(2, 3) = {}", compiled.call_data_param(2.0, 3.0));
    println!("f(5, 0.5) = {}", compiled.call_data_param(5.0, 0.5));
    println!("Signature: {:?}\n", compiled.signature);
}

fn demo_data_parameters() {
    println!("📊 Data + Multiple Parameters: f(x, θ₁, θ₂, ...) -> f64");
    println!("--------------------------------------------------------");

    // Quadratic model: f(x, a, b, c) = a*x² + b*x + c
    fn quadratic<E: MathExpr>(
        x: E::Repr<f64>,
        a: E::Repr<f64>,
        b: E::Repr<f64>,
        c: E::Repr<f64>,
    ) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        E::add(
            E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)),
            c,
        )
    }

    let jit_expr = quadratic::<JITEval>(
        JITEval::var("x"),
        JITEval::var("a"),
        JITEval::var("b"),
        JITEval::var("c"),
    );
    let param_vars = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let compiled = JITEval::compile_data_params(jit_expr, "x", &param_vars)
        .expect("Compilation should succeed");

    println!("Expression: a*x² + b*x + c");
    println!(
        "f(2, [1, -2, 1]) = {}",
        compiled.call_data_params(2.0, &[1.0, -2.0, 1.0])
    );
    println!(
        "f(3, [0.5, 1, -1]) = {}",
        compiled.call_data_params(3.0, &[0.5, 1.0, -1.0])
    );
    println!("Signature: {:?}\n", compiled.signature);
}

fn demo_embedded_constants() {
    println!("🔢 Embedded Constants for Performance");
    println!("-------------------------------------");

    // Circle area: f(r) = π * r²
    fn circle_area<E: MathExpr>(r: E::Repr<f64>) -> E::Repr<f64> {
        let pi_var = E::var::<f64>("pi"); // Will be replaced by constant
        E::mul(pi_var, E::pow(r, E::constant(2.0)))
    }

    let jit_expr = circle_area::<JITEval>(JITEval::var("r"));
    let mut constants = std::collections::HashMap::new();
    constants.insert("pi".to_string(), std::f64::consts::PI);

    let compiled = JITEval::compile_with_constants(jit_expr, &["r".to_string()], &[], &constants)
        .expect("Compilation should succeed");

    println!("Expression: π * r² (with π embedded as constant)");
    println!("Area(r=1) = {}", compiled.call_single(1.0));
    println!("Area(r=2) = {}", compiled.call_single(2.0));
    println!(
        "Embedded constants: {}\n",
        compiled.compilation_stats.embedded_constants
    );
}

fn demo_compilation_stats() {
    println!("📈 Compilation Statistics");
    println!("-------------------------");

    // Complex expression with transcendentals: f(x) = exp(sin(x)) + ln(cos(x))
    fn complex_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        E::add(E::exp(E::sin(x.clone())), E::ln(E::cos(x)))
    }

    let jit_expr = complex_expr::<JITEval>(JITEval::var("x"));
    let compiled = JITEval::compile(jit_expr).expect("Compilation should succeed");

    let stats = &compiled.compilation_stats;
    println!("Expression: exp(sin(x)) + ln(cos(x))");
    println!("Code size: {} bytes", stats.code_size_bytes);
    println!("CLIF instructions: {}", stats.clif_instructions);
    println!("Compilation time: {} μs", stats.compilation_time_us);
    println!("Estimated speedup: {:.1}x", stats.estimated_speedup);
    println!("Source: {}\n", compiled.source_expression);
}

fn demo_performance_comparison() {
    println!("⚡ Performance Comparison");
    println!("------------------------");

    // Test expression: f(x) = x³ + 2x² - x + 1
    fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        E::add(
            E::add(
                E::sub(
                    E::add(
                        E::pow(x.clone(), E::constant(3.0)),
                        E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))),
                    ),
                    x,
                ),
                E::constant(1.0),
            ),
            E::constant(0.0), // Just to make it more complex
        )
    }

    let jit_expr = test_expr::<JITEval>(JITEval::var("x"));
    let compiled = JITEval::compile_single_var(jit_expr, "x").expect("Compilation should succeed");

    // Native Rust equivalent for comparison
    fn native_fn(x: f64) -> f64 {
        x.powi(3) + 2.0 * x.powi(2) - x + 1.0
    }

    let test_values = [0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 10.0];

    println!("Testing accuracy against native Rust:");
    for &x in &test_values {
        let jit_result = compiled.call_single(x);
        let native_result = native_fn(x);
        let diff = (jit_result - native_result).abs();
        println!("x={x:4.1}: JIT={jit_result:10.6}, Native={native_result:10.6}, Diff={diff:.2e}");
    }

    // Performance benchmark
    const ITERATIONS: usize = 100_000;

    // JIT performance
    let start = Instant::now();
    let mut jit_sum = 0.0;
    for i in 0..ITERATIONS {
        jit_sum += compiled.call_single(i as f64 * 0.01);
    }
    let jit_time = start.elapsed();

    // Native performance
    let start = Instant::now();
    let mut native_sum = 0.0;
    for i in 0..ITERATIONS {
        native_sum += native_fn(i as f64 * 0.01);
    }
    let native_time = start.elapsed();

    println!("\nPerformance Benchmark ({ITERATIONS} iterations):");
    println!("JIT time:    {jit_time:?} (sum: {jit_sum:.6})");
    println!("Native time: {native_time:?} (sum: {native_sum:.6})");

    let jit_ns_per_call = jit_time.as_nanos() as f64 / ITERATIONS as f64;
    let native_ns_per_call = native_time.as_nanos() as f64 / ITERATIONS as f64;
    let ratio = jit_ns_per_call / native_ns_per_call;

    println!("JIT:    {jit_ns_per_call:.2} ns/call");
    println!("Native: {native_ns_per_call:.2} ns/call");
    println!(
        "JIT/Native ratio: {:.2}x ({:.0}% of native speed)",
        ratio,
        100.0 / ratio
    );
}
