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
    demo_custom_symbolic();
    demo_compilation_stats();
    demo_performance_comparison();
}

fn demo_single_variable() {
    println!("📊 Single Variable Functions: f(x) -> f64");
    println!("-------------------------------------------");

    // Polynomial: f(x) = 2x³ - 3x² + x + 1
    let x1 = JITEval::var::<f64>("x");
    let x2 = JITEval::var::<f64>("x");
    let x3 = JITEval::var::<f64>("x");
    let polynomial = JITEval::add::<f64, f64, f64>(
        JITEval::add::<f64, f64, f64>(
            JITEval::sub::<f64, f64, f64>(
                JITEval::mul::<f64, f64, f64>(
                    JITEval::constant::<f64>(2.0),
                    JITEval::pow::<f64>(x1, JITEval::constant::<f64>(3.0)),
                ),
                JITEval::mul::<f64, f64, f64>(
                    JITEval::constant::<f64>(3.0),
                    JITEval::pow::<f64>(x2, JITEval::constant::<f64>(2.0)),
                ),
            ),
            x3,
        ),
        JITEval::constant::<f64>(1.0),
    );

    let compiled =
        JITEval::compile_single_var(polynomial, "x").expect("Compilation should succeed");

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
    let x = JITEval::var::<f64>("x");
    let theta = JITEval::var::<f64>("theta");
    let linear_model = JITEval::add::<f64, f64, f64>(
        JITEval::mul::<f64, f64, f64>(theta, x),
        JITEval::constant::<f64>(1.0),
    );

    let compiled = JITEval::compile_data_param(linear_model, "x", "theta")
        .expect("Compilation should succeed");

    println!("Expression: θ * x + 1");
    println!("f(2, 3) = {}", compiled.call_data_param(2.0, 3.0));
    println!("f(5, 0.5) = {}", compiled.call_data_param(5.0, 0.5));
    println!("Signature: {:?}\n", compiled.signature);
}

fn demo_data_parameters() {
    println!("📊 Data + Multiple Parameters: f(x, θ₁, θ₂, ...) -> f64");
    println!("--------------------------------------------------------");

    // Quadratic model: f(x, a, b, c) = a*x² + b*x + c
    let x1 = JITEval::var::<f64>("x");
    let x2 = JITEval::var::<f64>("x");
    let a = JITEval::var::<f64>("a");
    let b = JITEval::var::<f64>("b");
    let c = JITEval::var::<f64>("c");

    let quadratic = JITEval::add::<f64, f64, f64>(
        JITEval::add::<f64, f64, f64>(
            JITEval::mul::<f64, f64, f64>(
                a,
                JITEval::pow::<f64>(x1, JITEval::constant::<f64>(2.0)),
            ),
            JITEval::mul::<f64, f64, f64>(b, x2),
        ),
        c,
    );

    let param_vars = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let compiled = JITEval::compile_data_params(quadratic, "x", &param_vars)
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
    let r = JITEval::var::<f64>("r");
    let pi_var = JITEval::var::<f64>("pi"); // Will be replaced by constant
    let circle_area = JITEval::mul::<f64, f64, f64>(
        pi_var,
        JITEval::pow::<f64>(r, JITEval::constant::<f64>(2.0)),
    );

    let mut constants = std::collections::HashMap::new();
    constants.insert("pi".to_string(), std::f64::consts::PI);

    let compiled =
        JITEval::compile_with_constants(circle_area, &["r".to_string()], &[], &constants)
            .expect("Compilation should succeed");

    println!("Expression: π * r² (with π embedded as constant)");
    println!("Area(r=1) = {}", compiled.call_single(1.0));
    println!("Area(r=2) = {}", compiled.call_single(2.0));
    println!(
        "Embedded constants: {}\n",
        compiled.compilation_stats.embedded_constants
    );
}

fn demo_custom_symbolic() {
    println!("🎯 Custom Symbolic Log-Density");
    println!("-------------------------------");

    // Standard normal log-density (without normalization): f(x) = -0.5 * x²
    let x = JITEval::var::<f64>("x");
    let log_density = JITEval::mul::<f64, f64, f64>(
        JITEval::constant::<f64>(-0.5),
        JITEval::pow::<f64>(x, JITEval::constant::<f64>(2.0)),
    );

    let parameters = std::collections::HashMap::new();
    let compiled = JITEval::compile_custom_symbolic(log_density, parameters)
        .expect("Compilation should succeed");

    println!("Expression: -0.5 * x² (standard normal log-density)");
    println!("log_density(0) = {}", compiled.call_single(0.0));
    println!("log_density(1) = {}", compiled.call_single(1.0));
    println!("log_density(2) = {}", compiled.call_single(2.0));
    println!("Signature: {:?}\n", compiled.signature);
}

fn demo_compilation_stats() {
    println!("📈 Compilation Statistics");
    println!("-------------------------");

    // Complex expression with transcendentals: f(x) = exp(sin(x)) + ln(cos(x))
    let x1 = JITEval::var::<f64>("x");
    let x2 = JITEval::var::<f64>("x");
    let complex_expr = JITEval::add::<f64, f64, f64>(
        JITEval::exp::<f64>(JITEval::sin::<f64>(x1)),
        JITEval::ln::<f64>(JITEval::cos::<f64>(x2)),
    );

    let compiled = JITEval::compile(complex_expr).expect("Compilation should succeed");

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
    let x1 = JITEval::var::<f64>("x");
    let x2 = JITEval::var::<f64>("x");
    let x3 = JITEval::var::<f64>("x");
    let test_expr = JITEval::add::<f64, f64, f64>(
        JITEval::add::<f64, f64, f64>(
            JITEval::sub::<f64, f64, f64>(
                JITEval::pow::<f64>(x1, JITEval::constant::<f64>(3.0)),
                x2,
            ),
            JITEval::mul::<f64, f64, f64>(
                JITEval::constant::<f64>(2.0),
                JITEval::pow::<f64>(x3, JITEval::constant::<f64>(2.0)),
            ),
        ),
        JITEval::constant::<f64>(1.0),
    );

    let compiled = JITEval::compile(test_expr).expect("Compilation should succeed");

    // Native Rust equivalent
    fn native_fn(x: f64) -> f64 {
        x.powi(3) + 2.0 * x.powi(2) - x + 1.0
    }

    // Benchmark JIT
    let test_value = 2.5;
    let iterations = 1_000_000;

    let start = Instant::now();
    let mut jit_result = 0.0;
    for _ in 0..iterations {
        jit_result = compiled.call_single(test_value);
    }
    let jit_time = start.elapsed();

    // Benchmark native
    let start = Instant::now();
    let mut native_result = 0.0;
    for _ in 0..iterations {
        native_result = native_fn(test_value);
    }
    let native_time = start.elapsed();

    println!("Expression: x³ + 2x² - x + 1");
    println!("Test value: {test_value}");
    println!("Iterations: {iterations}");
    println!();
    println!("JIT result: {jit_result}");
    println!("JIT time: {jit_time:?}");
    println!(
        "JIT per call: {:.2} ns",
        jit_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!();
    println!("Native result: {native_result}");
    println!("Native time: {native_time:?}");
    println!(
        "Native per call: {:.2} ns",
        native_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!();
    println!(
        "JIT vs Native ratio: {:.2}x",
        jit_time.as_nanos() as f64 / native_time.as_nanos() as f64
    );
    println!(
        "Results match: {}",
        (jit_result - native_result).abs() < 1e-10
    );
}
