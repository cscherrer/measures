//! Final Tagless Distributions Demo
//!
//! This example demonstrates the final tagless approach for probability distributions,
//! showcasing zero-cost abstractions and multiple interpreters for distribution computations.

use measures_distributions::final_tagless::*;
use symbolic_math::final_tagless::{DirectEval, MathExpr, PrettyPrint};

#[cfg(feature = "jit")]
use symbolic_math::final_tagless::JITEval;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Final Tagless Distributions Demo ===\n");

    // 1. Direct Evaluation Demo
    println!("1. Direct Evaluation with DistributionEval:");
    demo_direct_evaluation();
    println!();

    // 2. Pretty Printing Demo
    println!("2. Pretty Printing Demo:");
    demo_pretty_printing();
    println!();

    // 3. Distribution Pattern Library Demo
    println!("3. Distribution Pattern Library:");
    demo_pattern_library();
    println!();

    // 4. Comparison with Traditional Approach
    println!("4. Comparison with Traditional Distributions:");
    demo_traditional_comparison();
    println!();

    // 5. JIT Compilation Demo (if available)
    #[cfg(feature = "jit")]
    {
        println!("5. JIT Compilation Demo:");
        demo_jit_compilation()?;
        println!();
    }

    // 6. Performance Comparison
    println!("6. Performance Comparison:");
    demo_performance_comparison();
    println!();

    println!("Demo completed successfully!");
    Ok(())
}

/// Demonstrate direct evaluation with DistributionEval
fn demo_direct_evaluation() {
    // Normal distribution log-densities
    let points = [0.0, 1.0, -1.0, 2.0];
    println!("  Normal(μ=0, σ=1) log-densities:");
    for &x in &points {
        let result = DistributionEval::normal_log_density(x, 0.0, 1.0);
        println!("    f({:4.1}) = {:8.4}", x, result);
    }

    // Exponential distribution log-densities
    println!("  Exponential(λ=1.5) log-densities:");
    let exp_points = [0.0, 0.5, 1.0, 2.0, 3.0];
    for &x in &exp_points {
        let result = DistributionEval::exponential_log_density(x, 1.5);
        println!("    f({:4.1}) = {:8.4}", x, result);
    }

    // Gamma distribution log-densities
    println!("  Gamma(α=2, β=1) log-densities:");
    for &x in &exp_points {
        if x > 0.0 {
            let result = DistributionEval::gamma_log_density(x, 2.0, 1.0);
            println!("    f({:4.1}) = {:8.4}", x, result);
        }
    }
}

/// Demonstrate pretty printing of distribution expressions
fn demo_pretty_printing() {
    // Normal distribution expression
    let x = PrettyPrint::var("x");
    let mu = PrettyPrint::var("mu");
    let sigma = PrettyPrint::var("sigma");

    let normal_expr = patterns::normal_log_density::<PrettyPrint>(x, mu, sigma);
    println!("  Normal log-density expression:");
    println!("    {}", normal_expr);

    // Standard normal expression
    let x = PrettyPrint::var("x");
    let std_normal_expr = patterns::standard_normal_log_density::<PrettyPrint>(x);
    println!("  Standard normal log-density expression:");
    println!("    {}", std_normal_expr);

    // Exponential distribution expression
    let x = PrettyPrint::var("x");
    let rate = PrettyPrint::var("rate");

    let exp_expr = patterns::exponential_log_density::<PrettyPrint>(x, rate);
    println!("  Exponential log-density expression:");
    println!("    {}", exp_expr);

    // Cauchy distribution expression
    let x = PrettyPrint::var("x");
    let location = PrettyPrint::var("x0");
    let scale = PrettyPrint::var("gamma");

    let cauchy_expr = patterns::cauchy_log_density::<PrettyPrint>(x, location, scale);
    println!("  Cauchy log-density expression:");
    println!("    {}", cauchy_expr);
}

/// Demonstrate distribution pattern library
fn demo_pattern_library() {
    let test_points = [0.0, 1.0, -1.0, 2.0];

    // Standard normal pattern
    println!("  Standard Normal log-densities:");
    for &point in &test_points {
        let x = DirectEval::constant(point);
        let density = patterns::standard_normal_log_density::<DirectEval>(x);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }

    // Normal distribution with parameters
    println!("  Normal(μ=0.5, σ=2.0) log-densities:");
    let mu = DirectEval::constant(0.5);
    let sigma = DirectEval::constant(2.0);
    for &point in &test_points {
        let x = DirectEval::constant(point);
        let density = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }

    // Exponential distribution
    println!("  Exponential(λ=0.8) log-densities:");
    let rate = DirectEval::constant(0.8);
    let exp_points = [0.1, 0.5, 1.0, 2.0, 5.0];
    for &point in &exp_points {
        let x = DirectEval::constant(point);
        let density = patterns::exponential_log_density::<DirectEval>(x, rate);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }

    // Cauchy distribution
    println!("  Cauchy(x₀=0, γ=1) log-densities:");
    let location = DirectEval::constant(0.0);
    let scale = DirectEval::constant(1.0);
    for &point in &test_points {
        let x = DirectEval::constant(point);
        let density = patterns::cauchy_log_density::<DirectEval>(x, location, scale);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }
}

/// Demonstrate comparison with traditional distribution implementations
fn demo_traditional_comparison() {
    use measures_core::HasLogDensity;
    use measures_distributions::{Exponential, Normal};

    let test_points = [0.0, 1.0, -1.0, 2.0];

    println!("  Comparison: Normal(μ=0, σ=1)");
    let normal_traditional = Normal::new(0.0, 1.0);

    for &point in &test_points {
        // Traditional approach
        let traditional_result = normal_traditional.log_density_wrt_root(&point);

        // Final tagless approach
        let x = DirectEval::constant(point);
        let mu = DirectEval::constant(0.0);
        let sigma = DirectEval::constant(1.0);
        let final_tagless_result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);

        let diff = (traditional_result - final_tagless_result).abs();
        println!(
            "    x={:4.1}: Traditional={:8.4}, FinalTagless={:8.4}, Diff={:.2e}",
            point, traditional_result, final_tagless_result, diff
        );
    }

    println!("  Comparison: Exponential(λ=1.5)");
    let exp_traditional = Exponential::new(1.5);
    let exp_points = [0.1, 0.5, 1.0, 2.0];

    for &point in &exp_points {
        // Traditional approach
        let traditional_result = exp_traditional.log_density_wrt_root(&point);

        // Final tagless approach
        let x = DirectEval::constant(point);
        let rate = DirectEval::constant(1.5);
        let final_tagless_result = patterns::exponential_log_density::<DirectEval>(x, rate);

        let diff = (traditional_result - final_tagless_result).abs();
        println!(
            "    x={:4.1}: Traditional={:8.4}, FinalTagless={:8.4}, Diff={:.2e}",
            point, traditional_result, final_tagless_result, diff
        );
    }
}

/// Demonstrate JIT compilation (if available)
#[cfg(feature = "jit")]
fn demo_jit_compilation() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    println!("  Compiling normal log-density with JIT...");

    // Create JIT expression for normal log-density
    let x = JITEval::var::<f64>("x");
    let mu = JITEval::var::<f64>("mu");
    let sigma = JITEval::var::<f64>("sigma");

    let jit_expr = patterns::normal_log_density::<JITEval>(x, mu, sigma);

    // Compile with data + parameters signature
    let start_compile = Instant::now();
    let compiled =
        JITEval::compile_data_params(jit_expr, "x", &["mu".to_string(), "sigma".to_string()])?;
    let compile_time = start_compile.elapsed();

    println!(
        "  Normal distribution compilation completed in {:?}",
        compile_time
    );

    // Test the compiled function
    let test_points = [0.0, 1.0, -1.0, 2.0];
    let mu_val = 0.0;
    let sigma_val = 1.0;

    println!(
        "  JIT-compiled Normal(μ={}, σ={}) results:",
        mu_val, sigma_val
    );
    for &x_val in &test_points {
        let result = compiled.call_data_params(x_val, &[mu_val, sigma_val]);
        println!("    f({:4.1}) = {:8.4}", x_val, result);
    }

    // Compile exponential distribution
    println!("  Compiling exponential log-density with JIT...");
    let x = JITEval::var::<f64>("x");
    let rate = JITEval::var::<f64>("rate");

    let exp_jit_expr = patterns::exponential_log_density::<JITEval>(x, rate);

    let start_compile = Instant::now();
    let exp_compiled = JITEval::compile_data_param(exp_jit_expr, "x", "rate")?;
    let compile_time = start_compile.elapsed();

    println!(
        "  Exponential distribution compilation completed in {:?}",
        compile_time
    );

    let rate_val = 1.5;
    println!("  JIT-compiled Exponential(λ={}) results:", rate_val);
    let exp_points = [0.1, 0.5, 1.0, 2.0];
    for &x_val in &exp_points {
        let result = exp_compiled.call_data_param(x_val, rate_val);
        println!("    f({:4.1}) = {:8.4}", x_val, result);
    }

    Ok(())
}

/// Demonstrate performance comparison between interpreters
fn demo_performance_comparison() {
    use std::time::Instant;

    let iterations = 100000;
    let test_points = [0.0, 1.0, -1.0, 2.0, 0.5];

    println!(
        "  Performance comparison ({} iterations per point):",
        iterations
    );

    // DirectEval performance
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let x = DirectEval::constant(point);
            let mu = DirectEval::constant(0.0);
            let sigma = DirectEval::constant(1.0);
            let _result = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
        }
    }
    let direct_time = start.elapsed();
    let direct_ns = direct_time.as_nanos() / (iterations as u128 * test_points.len() as u128);

    println!("    DirectEval:         {:6} ns per call", direct_ns);

    // DistributionEval performance (optimized)
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let _result = DistributionEval::normal_log_density(point, 0.0, 1.0);
        }
    }
    let dist_time = start.elapsed();
    let dist_ns = dist_time.as_nanos() / (iterations as u128 * test_points.len() as u128);

    println!("    DistributionEval:   {:6} ns per call", dist_ns);

    // Traditional distribution performance
    use measures_core::HasLogDensity;
    use measures_distributions::Normal;

    let normal = Normal::new(0.0, 1.0);
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let _result = normal.log_density_wrt_root(&point);
        }
    }
    let traditional_time = start.elapsed();
    let traditional_ns =
        traditional_time.as_nanos() / (iterations as u128 * test_points.len() as u128);

    println!("    Traditional:        {:6} ns per call", traditional_ns);

    // Native Rust implementation for comparison
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let two_pi = 2.0 * std::f64::consts::PI;
            let _result = -0.5 * two_pi.ln() - 0.5 * point * point;
        }
    }
    let native_time = start.elapsed();
    let native_ns = native_time.as_nanos() / (iterations as u128 * test_points.len() as u128);

    println!("    Native Rust:        {:6} ns per call", native_ns);

    // Calculate speedup ratios
    let direct_ratio = direct_ns as f64 / native_ns as f64;
    let dist_ratio = dist_ns as f64 / native_ns as f64;
    let traditional_ratio = traditional_ns as f64 / native_ns as f64;

    println!("  Overhead ratios (vs native):");
    println!("    DirectEval:         {:.2}x", direct_ratio);
    println!("    DistributionEval:   {:.2}x", dist_ratio);
    println!("    Traditional:        {:.2}x", traditional_ratio);

    if dist_ns < direct_ns {
        let speedup = direct_ns as f64 / dist_ns as f64;
        println!(
            "  DistributionEval is {:.2}x faster than DirectEval",
            speedup
        );
    }

    if dist_ns < traditional_ns {
        let speedup = traditional_ns as f64 / dist_ns as f64;
        println!(
            "  DistributionEval is {:.2}x faster than Traditional",
            speedup
        );
    }
}
