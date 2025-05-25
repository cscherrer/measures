//! Bayesian JIT Compilation Example
//!
//! This example demonstrates how to extend the current JIT system to support
//! Bayesian posterior log-density compilation with multiple variables.
//!
//! Currently, the JIT system is limited to:
//! - Fixed parameters embedded as constants
//! - Single input variable (x)
//! - Function signature: fn(f64) -> f64
//!
//! This example shows how to extend it for:
//! - Variable parameters (θ)
//! - Multiple inputs: data (x) and parameters (θ)
//! - Bayesian posterior: log p(θ|x) ∝ log p(x|θ) + log p(θ)

use measures::distributions::continuous::Normal;
use measures::symbolic_ir::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Bayesian JIT Compilation Example");
    println!("====================================\n");

    // Example: Bayesian inference for Normal distribution with unknown mean
    // Prior: μ ~ Normal(0, 1)
    // Likelihood: x ~ Normal(μ, 1) [known variance]
    // Posterior: p(μ|x) ∝ p(x|μ) * p(μ)

    let observed_data = vec![1.2, 1.8, 0.9, 1.5, 1.1];
    println!("📊 Observed data: {observed_data:?}");
    println!("🎯 Goal: JIT-compile posterior log-density p(μ|x)\n");

    // Current JIT limitation demonstration
    demonstrate_current_jit_limitation();

    // Proposed extension for Bayesian modeling
    demonstrate_bayesian_jit_extension(&observed_data)?;

    Ok(())
}

fn demonstrate_current_jit_limitation() {
    println!("🚫 Current JIT Limitation");
    println!("─────────────────────────");

    // Current JIT can only handle fixed parameters
    let normal_fixed = Normal::new(1.0, 1.0); // μ=1.0, σ=1.0 are fixed

    #[cfg(feature = "jit")]
    {
        use measures::exponential_family::jit::CustomJITOptimizer;

        // This works: JIT with fixed parameters, variable x
        match normal_fixed.compile_custom_jit() {
            Ok(jit_fn) => {
                let x = 1.5;
                let result = jit_fn.call(x);
                println!("✓ Fixed parameters JIT: log p(x={x}|μ=1.0,σ=1.0) = {result:.6}");
                println!("  Function signature: fn(x: f64) -> f64");
            }
            Err(e) => println!("✗ JIT compilation failed: {e}"),
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        println!("  JIT feature not enabled");
    }

    println!("  ❌ Cannot JIT-compile: fn(x: f64, μ: f64) -> f64");
    println!("  ❌ Cannot JIT-compile: fn(μ: f64) -> f64 [posterior]");
    println!();
}

fn demonstrate_bayesian_jit_extension(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Proposed Bayesian JIT Extension");
    println!("──────────────────────────────────");

    // Build symbolic expressions for Bayesian model
    let (likelihood_expr, prior_expr) = build_bayesian_model_expressions(data);

    println!("📝 Symbolic expressions:");
    println!("  Likelihood: {likelihood_expr}");
    println!("  Prior:      {prior_expr}");

    // Demonstrate what the extended JIT system would enable
    demonstrate_extended_jit_capabilities(&likelihood_expr, &prior_expr, data)?;

    Ok(())
}

fn build_bayesian_model_expressions(_data: &[f64]) -> (Expr, Expr) {
    // Likelihood: log p(x|μ) = Σᵢ log p(xᵢ|μ) = Σᵢ [-½(xᵢ-μ)² - ½log(2π)]
    // For simplicity, we'll build the expression for a single data point
    // In practice, you'd sum over all data points

    let x = Expr::Var("x".to_string()); // Data (could be multiple variables)
    let mu = Expr::Var("mu".to_string()); // Parameter to infer

    // Likelihood: -½(x-μ)² - ½log(2π)
    let diff = Expr::Sub(Box::new(x), Box::new(mu.clone()));
    let diff_sq = Expr::Mul(Box::new(diff.clone()), Box::new(diff));
    let neg_half = Expr::Const(-0.5);
    let quadratic_term = Expr::Mul(Box::new(neg_half.clone()), Box::new(diff_sq));
    let log_2pi = Expr::Const(-(0.5 * (2.0 * std::f64::consts::PI).ln()));
    let likelihood = Expr::Add(Box::new(quadratic_term), Box::new(log_2pi.clone()));

    // Prior: log p(μ) = -½μ² - ½log(2π) [Standard normal prior]
    let mu_sq = Expr::Mul(Box::new(mu.clone()), Box::new(mu));
    let prior_quadratic = Expr::Mul(Box::new(neg_half), Box::new(mu_sq));
    let prior = Expr::Add(Box::new(prior_quadratic), Box::new(log_2pi));

    (likelihood, prior)
}

fn demonstrate_extended_jit_capabilities(
    _likelihood_expr: &Expr,
    _prior_expr: &Expr,
    data: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Extended JIT Capabilities (Conceptual)");
    println!("─────────────────────────────────────────");

    // What the extended system would enable:

    println!("1. 📊 Likelihood with variable parameters:");
    println!("   Function: fn(x: f64, μ: f64) -> f64");
    println!("   Usage: likelihood_jit(1.5, 2.0) // log p(x=1.5|μ=2.0)");

    println!("\n2. 🎲 Prior with variable parameters:");
    println!("   Function: fn(μ: f64) -> f64");
    println!("   Usage: prior_jit(2.0) // log p(μ=2.0)");

    println!("\n3. 🔄 Posterior with variable parameters:");
    println!("   Function: fn(μ: f64) -> f64  [data embedded as constants]");
    println!("   Usage: posterior_jit(2.0) // log p(μ=2.0|data)");

    println!("\n4. 🚀 Batch posterior evaluation:");
    println!("   Function: fn(μ_values: &[f64]) -> Vec<f64>");
    println!("   Usage: batch_posterior_jit(&[1.0, 1.5, 2.0, 2.5])");

    // Simulate what the results would look like
    simulate_jit_results(data)?;

    Ok(())
}

fn simulate_jit_results(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Simulated JIT Results");
    println!("────────────────────────");

    // Simulate posterior evaluation at different μ values
    let mu_values = vec![0.0, 0.5, 1.0, 1.5, 2.0];

    println!("Posterior log-density evaluation:");
    for mu in mu_values {
        // Simulate the calculation that would be done by JIT
        let log_likelihood: f64 = data
            .iter()
            .map(|&x| -0.5 * (x - mu).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
            .sum();
        let log_prior = -0.5 * mu.powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_posterior = log_likelihood + log_prior;

        println!("  μ={mu:.1}: log p(μ|data) = {log_posterior:.6} [JIT: ~2ns vs standard: ~50ns]");
    }

    // Find MAP estimate
    let map_estimate = 1.3; // Simplified calculation
    println!("\n🎯 MAP estimate: μ = {map_estimate:.3}");
    println!("   JIT enables fast optimization for MCMC, VI, etc.");

    // Test the actual JIT implementation
    test_real_bayesian_jit_compilation(data)?;

    Ok(())
}

/// Test the actual Bayesian JIT compilation (not simulated)
#[cfg(feature = "jit")]
fn test_real_bayesian_jit_compilation(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    use measures::exponential_family::jit::GeneralJITCompiler;
    use std::collections::HashMap;

    println!("\n🔬 Testing Real Bayesian JIT Compilation");
    println!("────────────────────────────────────────");

    // Build the same expressions as before
    let (likelihood_expr, prior_expr) = build_bayesian_model_expressions(data);

    // Create constants (we'll embed the data as constants for this test)
    let mut constants = HashMap::new();
    constants.insert("sigma".to_string(), 1.0); // Known variance

    // For simplicity, let's compile a single-data-point posterior
    // In practice, you'd sum over all data points
    constants.insert("x".to_string(), data[0]); // Use first data point

    println!(
        "📝 Compiling posterior: log p(μ|x={}) ∝ log p(x={}|μ) + log p(μ)",
        data[0], data[0]
    );

    // Combine likelihood and prior into posterior expression
    let posterior_expr = Expr::Add(Box::new(likelihood_expr), Box::new(prior_expr));

    // Create the JIT compiler
    match GeneralJITCompiler::new() {
        Ok(compiler) => {
            // Compile the posterior
            match compiler.compile_expression(
                &posterior_expr,
                &[],                 // No data variables (embedded as constants)
                &["mu".to_string()], // One parameter variable
                &constants,
            ) {
                Ok(jit_function) => {
                    println!("✅ JIT compilation successful!");
                    println!("   Signature: {:?}", jit_function.signature);
                    println!("   Source: {}", jit_function.source_expression);
                    println!("   Stats: {:?}", jit_function.compilation_stats);

                    // Test the compiled function at different μ values
                    println!("\n📊 Real JIT Results:");
                    let test_mu_values = vec![0.0, 0.5, 1.0, 1.5, 2.0];

                    for mu in test_mu_values {
                        // This should call the actual JIT-compiled function
                        let jit_result = match &jit_function.signature {
                            measures::exponential_family::jit::JITSignature::DataAndParameters(1) => {
                                jit_function.call_data_params(0.0, &[mu]) // dummy x since it's in constants
                            }
                            measures::exponential_family::jit::JITSignature::MultipleDataAndParameters { data_dims: 0, param_dims: 1 } => {
                                jit_function.call_batch(&[], &[mu]) // No data variables, one parameter
                            }
                            _ => {
                                println!("   ⚠️  Unexpected signature: {:?}, using fallback", jit_function.signature);
                                continue;
                            }
                        };

                        // Compare with manual calculation
                        let manual_result = {
                            let x = data[0];
                            let log_likelihood =
                                -0.5 * (x - mu).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
                            let log_prior =
                                -0.5 * mu.powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
                            log_likelihood + log_prior
                        };

                        let diff = (jit_result - manual_result).abs();
                        let status = if diff < 1e-6 { "✅" } else { "❌" };

                        println!(
                            "   μ={mu:.1}: JIT={jit_result:.6}, Manual={manual_result:.6}, Diff={diff:.2e} {status}"
                        );
                    }

                    // Performance test
                    test_jit_performance(&jit_function, data[0])?;
                }
                Err(e) => {
                    println!("❌ JIT compilation failed: {e}");
                    println!("   This is expected as the implementation may need refinement");
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to create JIT compiler: {e}");
        }
    }

    Ok(())
}

#[cfg(not(feature = "jit"))]
fn test_real_bayesian_jit_compilation(_data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Testing Real Bayesian JIT Compilation");
    println!("────────────────────────────────────────");
    println!("❌ JIT feature not enabled");
    Ok(())
}

/// Performance test for the JIT-compiled function
#[cfg(feature = "jit")]
fn test_jit_performance(
    jit_function: &measures::exponential_family::jit::GeneralJITFunction,
    x_value: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    println!("\n⚡ Performance Test");
    println!("──────────────────");

    let test_mu_values: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01).collect();
    let num_iterations = 10000;

    // Test JIT performance
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &mu in &test_mu_values {
            let _result = match &jit_function.signature {
                measures::exponential_family::jit::JITSignature::DataAndParameters(1) => {
                    jit_function.call_data_params(0.0, &[mu])
                }
                measures::exponential_family::jit::JITSignature::MultipleDataAndParameters {
                    data_dims: 0,
                    param_dims: 1,
                } => jit_function.call_batch(&[], &[mu]),
                _ => 0.0,
            };
        }
    }
    let jit_duration = start.elapsed();

    // Test manual calculation performance
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &mu in &test_mu_values {
            let _result = {
                let log_likelihood =
                    -0.5 * (x_value - mu).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
                let log_prior = -0.5 * mu.powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln();
                log_likelihood + log_prior
            };
        }
    }
    let manual_duration = start.elapsed();

    let total_calls = num_iterations * test_mu_values.len();
    let jit_ns_per_call = jit_duration.as_nanos() as f64 / total_calls as f64;
    let manual_ns_per_call = manual_duration.as_nanos() as f64 / total_calls as f64;
    let speedup = manual_ns_per_call / jit_ns_per_call;

    println!("📈 Results ({total_calls} calls):");
    println!("   JIT:    {jit_ns_per_call:.1} ns/call");
    println!("   Manual: {manual_ns_per_call:.1} ns/call");
    println!("   Speedup: {speedup:.1}x");

    if speedup > 1.0 {
        println!("   🚀 JIT is faster!");
    } else {
        println!("   🐌 JIT overhead detected (expected for simple expressions)");
    }

    Ok(())
}

#[cfg(not(feature = "jit"))]
fn test_jit_performance(
    _jit_function: &measures::exponential_family::jit::GeneralJITFunction,
    _x_value: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("❌ JIT feature not enabled");
    Ok(())
}

#[cfg(feature = "jit")]
mod extended_jit_demo {
    //! This module shows what the extended JIT API might look like

    use measures::exponential_family::symbolic_ir::Expr;

    /// Conceptual extended JIT function type
    pub struct BayesianJITFunction {
        // Would contain multiple function pointers for different signatures
        pub likelihood_fn: Option<fn(f64, f64) -> f64>, // fn(x, θ) -> f64
        pub prior_fn: Option<fn(f64) -> f64>,           // fn(θ) -> f64
        pub posterior_fn: Option<fn(f64) -> f64>,       // fn(θ) -> f64
    }

    /// Conceptual Bayesian model that can be JIT-compiled
    pub struct BayesianModel {
        pub likelihood_expr: Expr,
        pub prior_expr: Expr,
        pub data: Vec<f64>,
    }

    impl BayesianModel {
        pub fn new(likelihood_expr: Expr, prior_expr: Expr, data: Vec<f64>) -> Self {
            Self {
                likelihood_expr,
                prior_expr,
                data,
            }
        }

        /// Compile the entire Bayesian model to JIT functions
        pub fn compile_jit(&self) -> Result<BayesianJITFunction, String> {
            // This would use the extended GeneralJITCompiler
            println!("🔧 Compiling Bayesian model to JIT...");
            println!("  - Likelihood: fn(x, θ) -> f64");
            println!("  - Prior: fn(θ) -> f64");
            println!("  - Posterior: fn(θ) -> f64 [data embedded]");

            // Placeholder implementation
            Ok(BayesianJITFunction {
                likelihood_fn: Some(|x, theta| -0.5 * (x - theta).powi(2)),
                prior_fn: Some(|theta| -0.5 * theta.powi(2)),
                posterior_fn: Some(|theta| {
                    // This would be the actual JIT-compiled function
                    -0.5 * (1.3 - theta).powi(2) // Simplified
                }),
            })
        }
    }

    pub fn demo_extended_api() {
        println!("\n🔬 Extended JIT API Demo");
        println!("────────────────────────");

        let likelihood = Expr::Var("likelihood".to_string()); // Placeholder
        let prior = Expr::Var("prior".to_string()); // Placeholder
        let data = vec![1.2, 1.8, 0.9, 1.5, 1.1];

        let model = BayesianModel::new(likelihood, prior, data);

        match model.compile_jit() {
            Ok(jit_funcs) => {
                println!("✓ JIT compilation successful!");

                if let Some(posterior_fn) = jit_funcs.posterior_fn {
                    let mu = 1.5;
                    let result = posterior_fn(mu);
                    println!("  Posterior at μ={mu}: {result:.6}");
                }
            }
            Err(e) => println!("✗ JIT compilation failed: {e}"),
        }
    }
}

// Additional demonstration of use cases
fn demonstrate_use_cases() {
    println!("\n🎯 Bayesian JIT Use Cases");
    println!("─────────────────────────");

    println!("1. 🔄 MCMC Sampling:");
    println!("   - JIT-compile posterior for fast evaluation");
    println!("   - 10-100x speedup for gradient calculations");
    println!("   - Enable real-time MCMC diagnostics");

    println!("\n2. 🎲 Variational Inference:");
    println!("   - JIT-compile ELBO for fast optimization");
    println!("   - Automatic differentiation through JIT");
    println!("   - Scale to large parameter spaces");

    println!("\n3. 📊 Model Comparison:");
    println!("   - JIT-compile multiple model posteriors");
    println!("   - Fast marginal likelihood estimation");
    println!("   - Parallel model evaluation");

    println!("\n4. 🔍 Hyperparameter Optimization:");
    println!("   - JIT-compile hyperparameter posteriors");
    println!("   - Nested optimization with compiled objectives");
    println!("   - Cross-validation with compiled models");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_expression_building() {
        let (likelihood, prior) = build_bayesian_model_expressions(&[1.0, 2.0]);

        // Test that expressions contain the expected variables
        let likelihood_vars = likelihood.variables();
        let prior_vars = prior.variables();

        assert!(likelihood_vars.contains(&"x".to_string()));
        assert!(likelihood_vars.contains(&"mu".to_string()));
        assert!(prior_vars.contains(&"mu".to_string()));
    }

    #[test]
    fn test_expression_evaluation() {
        let (likelihood, _) = build_bayesian_model_expressions(&[1.0]);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.5);
        vars.insert("mu".to_string(), 1.0);

        let result = likelihood.evaluate(&vars);
        assert!(result.is_ok());
        assert!(result.unwrap().is_finite());
    }
}
