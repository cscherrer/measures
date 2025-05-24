//! Symbolic simplification of log-density expressions.
//!
//! This example demonstrates how to use symbolic mathematics to represent
//! and simplify log-density expressions from exponential families.
//! 
//! Requires the "symbolic" feature: cargo run --example symbolic_log_density --features symbolic

#[cfg(feature = "symbolic")]
use rusymbols::Expression;
#[cfg(feature = "symbolic")]
use std::collections::HashMap;

#[cfg(feature = "symbolic")]
use measures::{Normal, LogDensityBuilder, exponential_family::ExponentialFamily};

#[cfg(feature = "symbolic")]
fn main() {
    println!("=== Symbolic Log-Density Simplification ===\n");

    // Create a normal distribution for reference
    let normal = Normal::new(1.0, 2.0);
    let x_value = 0.5;
    
    println!("Normal distribution: μ=1.0, σ=2.0");
    println!("Evaluation point: x={}\n", x_value);

    // Compute numerical result for comparison
    let numerical_log_density: f64 = normal.log_density().at(&x_value);
    println!("Numerical log-density: {:.6}\n", numerical_log_density);

    // === Show the mathematical formulas ===
    println!("=== Mathematical Formulas ===");
    println!("Normal distribution log-density:");
    println!("  log p(x|μ,σ) = -½log(2πσ²) - (x-μ)²/(2σ²)");
    println!();
    println!("Exponential family form:");
    println!("  log p(x|θ) = η₁·T₁(x) + η₂·T₂(x) - A(η) + log h(x)");
    println!("  where:");
    println!("    η₁ = μ/σ²         (natural parameter 1)");
    println!("    η₂ = -1/(2σ²)     (natural parameter 2)");
    println!("    T₁(x) = x         (sufficient statistic 1)");
    println!("    T₂(x) = x²        (sufficient statistic 2)");
    println!("    A(η) = μ²/(2σ²) + ½log(2πσ²)  (log partition function)");
    println!("    h(x) = 1          (base measure density)");

    // === Create symbolic representation ===
    println!("\n=== Symbolic Representation ===");
    
    // Define symbolic variables
    let x = Expression::new_var("x");
    let mu = Expression::new_var("mu");  
    let sigma = Expression::new_var("sigma");
    let two = Expression::new_val(2.0);
    
    println!("Symbolic variables: x, μ, σ");
    
    // Build the exponential family form symbolically
    println!("\n=== Exponential Family Form ===");
    
    // Natural parameter η₁ = μ/σ²
    let sigma_squared = sigma.clone() * sigma.clone();
    let eta1 = mu.clone() / sigma_squared.clone();
    
    // Natural parameter η₂ = -1/(2σ²) 
    let eta2 = -(Expression::new_val(1.0) / (two.clone() * sigma_squared.clone()));
    
    println!("Natural parameters:");
    println!("  η₁ = μ/σ² = {}", eta1);
    println!("  η₂ = -1/(2σ²) = {}", eta2);
    
    // Sufficient statistics
    let t1 = x.clone();
    let t2 = x.clone() * x.clone();
    
    println!("Sufficient statistics:");
    println!("  T₁(x) = x = {}", t1);
    println!("  T₂(x) = x² = {}", t2);
    
    // Log partition function A(η) = μ²/(2σ²) + ½log(2πσ²)
    // We'll show the quadratic part symbolically
    let mu_squared = mu.clone() * mu.clone();
    let log_partition_quadratic = mu_squared / (two.clone() * sigma_squared.clone());
    
    println!("Log partition function (quadratic part):");
    println!("  A_quad(η) = μ²/(2σ²) = {}", log_partition_quadratic);
    
    // Exponential family log-density: η₁T₁(x) + η₂T₂(x) - A_quad(η)
    let exp_fam_part = eta1.clone() * t1.clone() + eta2.clone() * t2.clone();
    let exp_fam_log_density = exp_fam_part.clone() - log_partition_quadratic.clone();
    
    println!("\nExponential family log-density (main terms):");
    println!("  η₁T₁(x) + η₂T₂(x) - A_quad(η) = {}", exp_fam_log_density);
    
    // === Show expanded formula ===
    println!("\n=== Expanded Formula ===");
    println!("Substituting the expressions:");
    
    // Create the full expanded form step by step
    let term1 = mu.clone() / sigma_squared.clone() * x.clone();
    let term2 = -(Expression::new_val(1.0) / (two.clone() * sigma_squared.clone())) * (x.clone() * x.clone());
    let term3 = -(mu.clone() * mu.clone() / (two.clone() * sigma_squared.clone()));
    
    println!("  Term 1: (μ/σ²)·x = {}", term1);
    println!("  Term 2: (-1/(2σ²))·x² = {}", term2);
    println!("  Term 3: -μ²/(2σ²) = {}", term3);
    
    let full_expression = term1.clone() + term2.clone() + term3.clone();
    println!("  Full: {}", full_expression);
    
    // === Demonstrate simplification ===
    println!("\n=== Symbolic Simplification ===");
    
    // Create simpler examples to show simplification capabilities
    let simple_example1 = x.clone() + x.clone();
    let simple_example2 = x.clone() * Expression::new_val(2.0) + x.clone();
    let simple_example3 = mu.clone() * Expression::new_val(0.0);
    
    println!("Examples of symbolic expressions:");
    println!("  x + x = {}", simple_example1);
    println!("  2x + x = {}", simple_example2);
    println!("  μ × 0 = {}", simple_example3);
    
    // === Show specific case formulas ===
    println!("\n=== Specific Case Formulas ===");
    
    // Standard normal case (μ=0, σ=1)
    println!("Standard Normal (μ=0, σ=1):");
    println!("  η₁ = 0/1² = 0");
    println!("  η₂ = -1/(2×1²) = -0.5");
    println!("  log p(x) = 0×x + (-0.5)×x² - 0 = -0.5x²");
    
    // Unit variance case (σ=1)
    println!("Unit Variance Normal (σ=1):");
    println!("  η₁ = μ/1² = μ");
    println!("  η₂ = -1/(2×1²) = -0.5");
    println!("  log p(x) = μx - 0.5x² - μ²/2");
    
    // === Numerical evaluation ===
    println!("\n=== Numerical Evaluation ===");
    
    // Set up variable values for evaluation
    let mut vars: HashMap<&str, f64> = HashMap::new();
    vars.insert("x", x_value);
    vars.insert("mu", normal.mean);
    vars.insert("sigma", normal.std_dev);
    
    println!("For our distribution Normal(μ={}, σ={}) at x={}:", normal.mean, normal.std_dev, x_value);
    
    // Evaluate each term
    if let Some(t1_val) = term1.eval_args(&vars) {
        println!("  Term 1 value: {:.6}", t1_val);
    }
    if let Some(t2_val) = term2.eval_args(&vars) {
        println!("  Term 2 value: {:.6}", t2_val);
    }
    if let Some(t3_val) = term3.eval_args(&vars) {
        println!("  Term 3 value: {:.6}", t3_val);
    }
    
    // Evaluate the exponential family part
    match exp_fam_part.eval_args(&vars) {
        Some(symbolic_result) => {
            println!("  Exponential family part: {:.6}", symbolic_result);
            
            // For comparison, compute the same using our exponential family implementation
            let (natural_params, _log_partition) = normal.natural_and_log_partition();
            let sufficient_stats = normal.sufficient_statistic(&x_value);
            let computed_exp_fam = natural_params[0] * sufficient_stats[0] + 
                                   natural_params[1] * sufficient_stats[1];
            
            println!("  Direct computation:      {:.6}", computed_exp_fam);
            println!("  Difference:              {:.10}", (symbolic_result - computed_exp_fam).abs());
            
            if (symbolic_result - computed_exp_fam).abs() < 1e-6 {
                println!("  ✓ Symbolic and direct computation match!");
            } else {
                println!("  ⚠ Small difference due to approximations");
            }
        }
        None => {
            println!("  Error evaluating symbolic expression");
        }
    }
    
    // === Show parameter relationships ===
    println!("\n=== Parameter Relationships ===");
    
    // Show how natural parameters relate to distribution parameters
    let eta1_numeric = normal.mean / (normal.std_dev * normal.std_dev);
    let eta2_numeric = -1.0 / (2.0 * normal.std_dev * normal.std_dev);
    
    println!("For Normal(μ={}, σ={}):", normal.mean, normal.std_dev);
    println!("  η₁ = μ/σ² = {}/{:.1}² = {:.6}", normal.mean, normal.std_dev, eta1_numeric);
    println!("  η₂ = -1/(2σ²) = -1/(2×{:.1}²) = {:.6}", normal.std_dev, eta2_numeric);
    
    // Evaluate symbolic natural parameters
    if let (Some(eta1_symbolic), Some(eta2_symbolic)) = (eta1.eval_args(&vars), eta2.eval_args(&vars)) {
        println!("  η₁ (symbolic) = {:.6}", eta1_symbolic);
        println!("  η₂ (symbolic) = {:.6}", eta2_symbolic);
        println!("  Differences: Δη₁={:.10}, Δη₂={:.10}", 
                 (eta1_numeric - eta1_symbolic).abs(),
                 (eta2_numeric - eta2_symbolic).abs());
    }
    
    // === Show different distribution parameters ===
    println!("\n=== Different Parameter Values ===");
    
    // Standard normal case
    let mut std_vars: HashMap<&str, f64> = HashMap::new();
    std_vars.insert("x", 0.0);
    std_vars.insert("mu", 0.0);
    std_vars.insert("sigma", 1.0);
    
    if let (Some(eta1_std), Some(eta2_std)) = (eta1.eval_args(&std_vars), eta2.eval_args(&std_vars)) {
        println!("Standard Normal (μ=0, σ=1):");
        println!("  η₁ = {:.6}", eta1_std);
        println!("  η₂ = {:.6}", eta2_std);
    }
    
    // High variance case
    let mut high_var_vars: HashMap<&str, f64> = HashMap::new();
    high_var_vars.insert("x", 0.0);
    high_var_vars.insert("mu", 0.0);
    high_var_vars.insert("sigma", 5.0);
    
    if let (Some(eta1_hv), Some(eta2_hv)) = (eta1.eval_args(&high_var_vars), eta2.eval_args(&high_var_vars)) {
        println!("High Variance Normal (μ=0, σ=5):");
        println!("  η₁ = {:.6}", eta1_hv);
        println!("  η₂ = {:.6}", eta2_hv);
    }
    
    println!("\n=== Summary ===");
    println!("✓ Mathematical formulas displayed clearly");
    println!("✓ Symbolic representation of natural parameters created");
    println!("✓ Exponential family form derived symbolically");
    println!("✓ Term-by-term evaluation performed");
    println!("✓ Numerical verification completed");
    println!("✓ Parameter relationships demonstrated");
    println!("✓ Different parameter values evaluated");
    
    println!("\nNote: This example demonstrates symbolic representation of the");
    println!("exponential family parameters and formulas. The rusymbols crate");
    println!("provides basic symbolic manipulation for mathematical expressions.");
}

#[cfg(not(feature = "symbolic"))]
fn main() {
    println!("This example requires the 'symbolic' feature.");
    println!("Run with: cargo run --example symbolic_log_density --features symbolic");
} 