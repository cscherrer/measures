//! Demonstration of the ergonomic symbolic IR interface
//!
//! This example shows how to use the enhanced IR with standard Rust notation,
//! improved display formatting, and convenient builder functions.

use measures::symbolic_ir::{builders, display};
use measures::{const_expr, var};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ergonomic Symbolic IR Demo ===\n");

    // 1. Basic expression building with natural syntax
    println!("1. Natural Rust-like syntax:");

    // Using the expr! macro for natural notation
    let x = var!("x");
    let y = var!("y");
    let mu = var!("mu");
    let sigma = var!("sigma");

    // Simple arithmetic - now looks like regular Rust!
    let linear = 2.0 * x.clone() + 3.0;
    println!("Linear: {linear}");

    let quadratic = x.clone() * x.clone() + 2.0 * x.clone() + 1.0;
    println!("Quadratic: {quadratic}");

    // Mathematical functions
    let trig_expr = x.clone().sin() + y.clone().cos();
    println!("Trigonometric: {trig_expr}");

    let log_expr = (x.clone() + 1.0).natural_log();
    println!("Logarithmic: {log_expr}");

    println!();

    // 2. Enhanced display formatting
    println!("2. Enhanced display formatting:");

    let complex_expr = (x.clone() - mu.clone()) / sigma.clone();
    let normal_kernel = (-0.5 * complex_expr.square()).exponential();

    println!("Standard display: {normal_kernel}");
    println!("Pretty display: {}", display::PrettyExpr(&normal_kernel));
    println!("LaTeX: {}", display::latex(&normal_kernel));
    println!("Python: {}", display::python(&normal_kernel));
    println!();

    // 3. Builder functions for common patterns
    println!("3. Builder functions for common patterns:");

    let normal_pdf = builders::normal_log_pdf(x.clone(), mu.clone(), sigma.clone());
    println!("Normal log-PDF: {normal_pdf}");

    let polynomial = builders::polynomial(x.clone(), &[1.0, -2.0, 1.0]); // x² - 2x + 1
    println!("Polynomial (x² - 2x + 1): {polynomial}");

    let gaussian = builders::gaussian_kernel(x.clone(), 0.0, 1.0);
    println!("Gaussian kernel: {gaussian}");

    let logistic = builders::logistic(x.clone());
    println!("Logistic function: {logistic}");

    println!();

    // 4. Evaluation with mixed types
    println!("4. Expression evaluation:");

    let expr = 2.0 * x.clone() + 3.0 * y.clone();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);
    vars.insert("y".to_string(), 2.0);

    let result = expr.evaluate(&vars)?;
    println!("2x + 3y where x=5, y=2: {result}");

    // Evaluate the normal PDF
    let mut params = HashMap::new();
    params.insert("x".to_string(), 1.0);
    params.insert("mu".to_string(), 0.0);
    params.insert("sigma".to_string(), 1.0);

    let pdf_result = normal_pdf.evaluate(&params)?;
    println!("Normal log-PDF at x=1, μ=0, σ=1: {pdf_result:.6}");

    println!();

    // 5. Simplification
    println!("5. Expression simplification:");

    let before = x.clone() * 1.0 + 0.0 * y.clone() + (x.clone() * 0.0);
    println!("Before simplification: {before}");

    let after = before.simplify();
    println!("After simplification: {after}");

    println!();

    // 6. Complex mathematical expressions
    println!("6. Complex mathematical expressions:");

    // Bayesian posterior (proportional to likelihood × prior)
    let likelihood = builders::normal_log_pdf(x.clone(), mu.clone(), sigma.clone());
    let prior = builders::normal_log_pdf(mu.clone(), 0.0, 10.0);
    let posterior = likelihood + prior;

    println!("Bayesian posterior (log scale):");
    println!("{}", display::equation("log p(μ|x)", &posterior));

    // Mixture of Gaussians
    let weight1 = const_expr!(0.3);
    let weight2 = const_expr!(0.7);
    let gauss1 = builders::gaussian_kernel(x.clone(), -2.0, 1.0);
    let gauss2 = builders::gaussian_kernel(x.clone(), 2.0, 1.5);
    let mixture = weight1 * gauss1 + weight2 * gauss2;

    println!("\nGaussian mixture:");
    println!("{}", display::equation("p(x)", &mixture));

    println!();

    // 7. Variable extraction and analysis
    println!("7. Expression analysis:");

    let complex = (x.clone() + y.clone()).square() * mu.clone().exponential() / sigma.clone();
    println!("Expression: {complex}");
    println!("Variables: {:?}", complex.variables());
    println!("Complexity: {} operations", complex.complexity());

    println!();

    // 8. Different output formats
    println!("8. Multiple output formats:");
    let expr = x.clone().square() + y.clone().sin();

    println!("Expression: x² + sin(y)");
    println!("Standard: {expr}");
    println!("LaTeX: {}", display::latex(&expr));
    println!("Python: {}", display::python(&expr));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ergonomic_interface() {
        let x = var!("x");
        let expr = 2.0 * x + 3.0;

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);

        assert_eq!(expr.evaluate(&vars).unwrap(), 13.0);
    }

    #[test]
    fn test_builder_functions() {
        let x = var!("x");
        let normal = builders::normal_log_pdf(x, 0.0, 1.0);

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);

        // At x=μ, the quadratic term should be 0, so result should be -log(σ√(2π))
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        let result = normal.evaluate(&vars).unwrap();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_display_formats() {
        let x = var!("x");
        let expr = x.square();

        let latex_output = display::latex(&expr);
        let python_output = display::python(&expr);

        println!("LaTeX output: {}", latex_output);
        println!("Python output: {}", python_output);

        // The LaTeX output should contain mathematical notation
        assert!(latex_output.contains("cdot") || latex_output.contains("x"));
        assert!(python_output.contains("**") || python_output.contains("*"));
    }
}
