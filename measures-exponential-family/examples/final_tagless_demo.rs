//! Final Tagless Exponential Family Demo
//!
//! This example demonstrates the final tagless approach for exponential family
//! distributions, showcasing zero-cost abstractions and multiple interpreters.

use measures_exponential_family::final_tagless::*;
use symbolic_math::final_tagless::{DirectEval, PrettyPrint, MathExpr};

#[cfg(feature = "jit")]
use symbolic_math::final_tagless::JITEval;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Final Tagless Exponential Family Demo ===\n");

    // 1. Direct Evaluation Demo
    println!("1. Direct Evaluation with ExpFamEval:");
    demo_direct_evaluation();
    println!();

    // 2. Pretty Printing Demo
    println!("2. Pretty Printing Demo:");
    demo_pretty_printing();
    println!();

    // 3. Exponential Family Operations Demo
    println!("3. Exponential Family Operations:");
    demo_exp_fam_operations();
    println!();

    // 4. Pattern Library Demo
    println!("4. Pattern Library Demo:");
    demo_pattern_library();
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

/// Demonstrate direct evaluation with ExpFamEval
fn demo_direct_evaluation() {
    // Basic dot product computation
    let params = [1.5, 2.0, -0.5];
    let stats = [2.0, 1.0, 3.0];
    let dot_product = ExpFamEval::dot_product_array(&params, &stats);
    println!("  Dot product [1.5, 2.0, -0.5] · [2.0, 1.0, 3.0] = {}", dot_product);

    // Sum of sufficient statistics
    let stats1 = [1.0, 2.0];
    let stats2 = [3.0, 4.0];
    let stats3 = [0.5, 1.5];
    let stats_list = [&stats1[..], &stats2[..], &stats3[..]];
    let sum_stats = ExpFamEval::sum_stats_array(&stats_list);
    println!("  Sum of sufficient statistics: {:?}", sum_stats);

    // Standard normal log-density at x=0
    let x = ExpFamEval::var("x", 0.0);
    let log_density = patterns::standard_normal_log_density::<ExpFamEval>(x);
    println!("  Standard normal log-density at x=0: {:.6}", log_density);
}

/// Demonstrate pretty printing of expressions
fn demo_pretty_printing() {
    // Normal log-density expression
    let x = PrettyPrint::var("x");
    let mu = PrettyPrint::var("mu");
    let sigma = PrettyPrint::var("sigma");
    
    let normal_expr = patterns::normal_log_density::<PrettyPrint>(x, mu, sigma);
    println!("  Normal log-density expression:");
    println!("    {}", normal_expr);

    // Exponential log-density expression
    let x = PrettyPrint::var("x");
    let lambda = PrettyPrint::var("lambda");
    
    let exp_expr = patterns::exponential_log_density::<PrettyPrint>(x, lambda);
    println!("  Exponential log-density expression:");
    println!("    {}", exp_expr);

    // Logistic function
    let x = PrettyPrint::var("x");
    let logistic_expr = PrettyPrint::logistic::<f64>(x);
    println!("  Logistic function:");
    println!("    {}", logistic_expr);
}

/// Demonstrate exponential family operations
fn demo_exp_fam_operations() {
    // Dot product with DirectEval
    let natural_params = vec![
        DirectEval::constant(1.0),
        DirectEval::constant(-0.5),
        DirectEval::constant(2.0)
    ];
    let sufficient_stats = vec![
        DirectEval::constant(3.0),
        DirectEval::constant(4.0),
        DirectEval::constant(1.0)
    ];
    
    let dot_product = DirectEval::dot_product(&natural_params, &sufficient_stats);
    println!("  Dot product result: {}", dot_product);

    // Complete exponential family log-density
    let log_partition = DirectEval::constant(2.5);
    let log_base_measure = DirectEval::constant(-1.0);
    
    let exp_fam_density = DirectEval::exp_fam_log_density(
        &natural_params,
        &sufficient_stats,
        log_partition,
        log_base_measure
    );
    println!("  Exponential family log-density: {}", exp_fam_density);

    // IID version with 10 samples
    let n_samples = DirectEval::constant(10.0);
    let sum_sufficient_stats = vec![
        DirectEval::constant(30.0),  // 10 * 3.0
        DirectEval::constant(40.0),  // 10 * 4.0
        DirectEval::constant(10.0)   // 10 * 1.0
    ];
    let sum_log_base_measure = DirectEval::constant(-10.0); // 10 * -1.0
    
    let iid_density = DirectEval::iid_exp_fam_log_density(
        &natural_params,
        &sum_sufficient_stats,
        log_partition,
        n_samples,
        sum_log_base_measure
    );
    println!("  IID exponential family log-density (n=10): {}", iid_density);
}

/// Demonstrate pattern library
fn demo_pattern_library() {
    // Standard normal at different points
    let points = [0.0, 1.0, -1.0, 2.0];
    println!("  Standard normal log-densities:");
    for &point in &points {
        let x = DirectEval::constant(point);
        let density = patterns::standard_normal_log_density::<DirectEval>(x);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }

    // Normal distribution with different parameters
    println!("  Normal(mu=1.0, sigma=2.0) log-densities:");
    let mu = DirectEval::constant(1.0);
    let sigma = DirectEval::constant(2.0);
    for &point in &points {
        let x = DirectEval::constant(point);
        let density = patterns::normal_log_density::<DirectEval>(x, mu, sigma);
        println!("    f({:4.1}) = {:8.4}", point, density);
    }

    // Exponential distribution
    println!("  Exponential(lambda=0.5) log-densities:");
    let lambda = DirectEval::constant(0.5);
    let exp_points = [0.1, 0.5, 1.0, 2.0, 5.0];
    for &point in &exp_points {
        let x = DirectEval::constant(point);
        let density = patterns::exponential_log_density::<DirectEval>(x, lambda);
        println!("    f({:4.1}) = {:8.4}", point, density);
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
    let compiled = JITEval::compile_data_params(
        jit_expr, 
        "x", 
        &["mu".to_string(), "sigma".to_string()]
    )?;
    let compile_time = start_compile.elapsed();
    
    println!("  Compilation completed in {:?}", compile_time);
    
    // Test the compiled function
    let test_points = [0.0, 1.0, -1.0, 2.0];
    let mu_val = 0.5;
    let sigma_val = 1.5;
    
    println!("  JIT-compiled Normal(mu={}, sigma={}) results:", mu_val, sigma_val);
    
    // Benchmark performance
    let start_eval = Instant::now();
    let iterations = 10000;
    
    for _ in 0..iterations {
        for &x_val in &test_points {
            let _result = compiled.call_data_params(x_val, &[mu_val, sigma_val]);
        }
    }
    
    let eval_time = start_eval.elapsed();
    let ns_per_call = eval_time.as_nanos() / (iterations as u128 * test_points.len() as u128);
    
    println!("  Performance: {} ns per call ({} iterations)", ns_per_call, iterations * test_points.len());
    
    // Show actual results
    for &x_val in &test_points {
        let result = compiled.call_data_params(x_val, &[mu_val, sigma_val]);
        println!("    f({:4.1}) = {:8.4}", x_val, result);
    }
    
    Ok(())
}

/// Demonstrate performance comparison between interpreters
fn demo_performance_comparison() {
    use std::time::Instant;
    
    let iterations = 100000;
    let test_points = [0.0, 1.0, -1.0, 2.0, 0.5];
    
    println!("  Performance comparison ({} iterations per point):", iterations);
    
    // DirectEval performance
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let x = DirectEval::constant(point);
            let _result = patterns::standard_normal_log_density::<DirectEval>(x);
        }
    }
    let direct_time = start.elapsed();
    let direct_ns = direct_time.as_nanos() / (iterations as u128 * test_points.len() as u128);
    
    println!("    DirectEval:     {:6} ns per call", direct_ns);
    
    // ExpFamEval performance (optimized)
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let x = ExpFamEval::var("x", point);
            let _result = patterns::standard_normal_log_density::<ExpFamEval>(x);
        }
    }
    let expfam_time = start.elapsed();
    let expfam_ns = expfam_time.as_nanos() / (iterations as u128 * test_points.len() as u128);
    
    println!("    ExpFamEval:     {:6} ns per call", expfam_ns);
    
    // Native Rust implementation for comparison
    let start = Instant::now();
    for _ in 0..iterations {
        for &point in &test_points {
            let _result = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * point * point;
        }
    }
    let native_time = start.elapsed();
    let native_ns = native_time.as_nanos() / (iterations as u128 * test_points.len() as u128);
    
    println!("    Native Rust:    {:6} ns per call", native_ns);
    
    // Calculate speedup ratios
    let direct_ratio = direct_ns as f64 / native_ns as f64;
    let expfam_ratio = expfam_ns as f64 / native_ns as f64;
    
    println!("  Overhead ratios (vs native):");
    println!("    DirectEval:     {:.2}x", direct_ratio);
    println!("    ExpFamEval:     {:.2}x", expfam_ratio);
    
    if expfam_ns < direct_ns {
        let speedup = direct_ns as f64 / expfam_ns as f64;
        println!("  ExpFamEval is {:.2}x faster than DirectEval", speedup);
    }
} 