//! Exponential Family vs Non-Exponential Family Example
//!
//! This example demonstrates that our trait system correctly handles both:
//! - Exponential family distributions (Normal) - get automatic implementation
//! - Non-exponential family distributions (Cauchy) - require manual implementation
//!
//! Both types work seamlessly with the same API.
//!
//! Run with: cargo run --example exponential_vs_non_exponential

use measures::{Cauchy, LogDensityBuilder, Normal};

fn main() {
    println!("🎯 === Exponential Family vs Non-Exponential Family === 🎯\n");

    // Create distributions
    let normal = Normal::new(0.0, 1.0);    // Exponential family
    let cauchy = Cauchy::new(0.0, 1.0);    // NOT exponential family
    let test_point = 0.5;

    println!("Normal distribution: N(μ=0.0, σ=1.0) [Exponential Family]");
    println!("Cauchy distribution: Cauchy(x₀=0.0, γ=1.0) [NOT Exponential Family]");
    println!("Test point: x = {test_point}\n");

    demonstrate_basic_log_density(&normal, &cauchy, test_point);
    demonstrate_relative_densities(&normal, &cauchy, test_point);
    demonstrate_type_level_dispatch(&normal, &cauchy);
    demonstrate_shared_api(&normal, &cauchy, test_point);

    println!("\n🎉 === Example Complete! === 🎉");
    println!("✅ Both exponential families and non-exponential families work seamlessly");
    println!("✅ Type-level dispatch correctly routes to appropriate implementations");
    println!("✅ Same API works for all distribution types");
    println!("🚀 This demonstrates the power of our trait-based architecture!");
}

fn demonstrate_basic_log_density(normal: &Normal<f64>, cauchy: &Cauchy<f64>, x: f64) {
    println!("=== 1. Basic Log-Density Computation ===");

    // Both use the same API, but different implementations
    let normal_log_density: f64 = normal.log_density().at(&x);
    let cauchy_log_density: f64 = cauchy.log_density().at(&x);

    println!("Normal log-density at x={x}: {normal_log_density:.6}");
    println!("Cauchy log-density at x={x}: {cauchy_log_density:.6}");
    
    // Show the difference
    let difference = normal_log_density - cauchy_log_density;
    println!("Difference (Normal - Cauchy): {difference:.6}");
    println!("Note: Different values because they're different distributions\n");
}

fn demonstrate_relative_densities(normal: &Normal<f64>, cauchy: &Cauchy<f64>, x: f64) {
    println!("=== 2. Relative Density Computation ===");

    // Compute relative densities using the builder pattern
    let normal_wrt_cauchy: f64 = normal.log_density().wrt(cauchy.clone()).at(&x);
    let cauchy_wrt_normal: f64 = cauchy.log_density().wrt(normal.clone()).at(&x);

    println!("log(Normal/Cauchy) at x={x}: {normal_wrt_cauchy:.6}");
    println!("log(Cauchy/Normal) at x={x}: {cauchy_wrt_normal:.6}");
    
    // Verify they're negatives of each other
    let sum = normal_wrt_cauchy + cauchy_wrt_normal;
    println!("Sum (should be ~0): {sum:.10}");
    println!("This demonstrates the mathematical relationship: log(A/B) = -log(B/A)\n");
}

fn demonstrate_type_level_dispatch(normal: &Normal<f64>, cauchy: &Cauchy<f64>) {
    use measures::core::{MeasureMarker, types::TypeLevelBool};
    
    println!("=== 3. Type-Level Dispatch Verification ===");

    // Check type-level markers
    let normal_is_exp_fam = <Normal<f64> as MeasureMarker>::IsExponentialFamily::VALUE;
    let cauchy_is_exp_fam = <Cauchy<f64> as MeasureMarker>::IsExponentialFamily::VALUE;
    
    let normal_is_primitive = <Normal<f64> as MeasureMarker>::IsPrimitive::VALUE;
    let cauchy_is_primitive = <Cauchy<f64> as MeasureMarker>::IsPrimitive::VALUE;

    println!("Normal distribution:");
    println!("  IsExponentialFamily: {normal_is_exp_fam}");
    println!("  IsPrimitive: {normal_is_primitive}");
    
    println!("Cauchy distribution:");
    println!("  IsExponentialFamily: {cauchy_is_exp_fam}");
    println!("  IsPrimitive: {cauchy_is_primitive}");
    
    println!("\nType-level dispatch routes:");
    println!("  Normal → Automatic exponential family implementation");
    println!("  Cauchy → Manual HasLogDensity implementation\n");
}

fn demonstrate_shared_api(normal: &Normal<f64>, cauchy: &Cauchy<f64>, x: f64) {
    println!("=== 4. Shared API Demonstration ===");

    // Both distributions work with the same API calls
    let normal_result: f64 = normal.log_density().at(&x);
    let cauchy_result: f64 = cauchy.log_density().at(&x);

    println!("Using identical API calls:");
    println!("  normal.log_density().at(&{x}) = {normal_result:.6}");
    println!("  cauchy.log_density().at(&{x}) = {cauchy_result:.6}");

    // Create f32 versions for proper type matching
    let normal_f32 = Normal::new(0.0f32, 1.0f32);
    let cauchy_f32 = Cauchy::new(0.0f32, 1.0f32);
    let x_f32 = x as f32;

    // Both work with different numeric types
    let normal_f32_result: f32 = normal_f32.log_density().at(&x_f32);
    let cauchy_f32_result: f32 = cauchy_f32.log_density().at(&x_f32);

    println!("\nWith f32 precision:");
    println!("  Normal f32 result: {normal_f32_result:.6}");
    println!("  Cauchy f32 result: {cauchy_f32_result:.6}");

    // Verify consistency between f64 and f32
    let normal_diff = (normal_result - normal_f32_result as f64).abs();
    let cauchy_diff = (cauchy_result - cauchy_f32_result as f64).abs();
    
    println!("\nPrecision differences (f64 vs f32):");
    println!("  Normal: {normal_diff:.8}");
    println!("  Cauchy: {cauchy_diff:.8}");
    
    println!("\nKey insight: Both distributions use identical API despite different implementations!");
} 