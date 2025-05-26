//! Exponential Family Log-Density IR Example
//!
//! This example demonstrates how exponential family log-density expressions
//! are represented in symbolic IR and how they are simplified using egglog
//! optimization. We'll show several distributions and their optimization.

use std::collections::HashMap;
use symbolic_math::{EgglogOptimize, Expr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Exponential Family Log-Density IR Optimization Example ===\n");

    // Example 1: Normal Distribution Log-Density
    println!("1. NORMAL DISTRIBUTION");
    println!("   Formula: log p(x|μ,σ²) = -½log(2πσ²) - (x-μ)²/(2σ²)");
    demonstrate_normal_ir()?;

    // Example 2: Poisson Distribution Log-Density
    println!("\n2. POISSON DISTRIBUTION");
    println!("   Formula: log p(k|λ) = k·log(λ) - λ - log(k!)");
    demonstrate_poisson_ir()?;

    // Example 3: Gamma Distribution Log-Density
    println!("\n3. GAMMA DISTRIBUTION");
    println!("   Formula: log p(x|α,β) = (α-1)·log(x) - β·x + α·log(β) - log(Γ(α))");
    demonstrate_gamma_ir()?;

    // Example 4: Complex Exponential Family Expression
    println!("\n4. COMPLEX EXPONENTIAL FAMILY EXPRESSION");
    println!("   Showing advanced algebraic simplifications");
    demonstrate_complex_exp_fam_ir()?;

    // Example 5: IID (Independent and Identically Distributed) Case
    println!("\n5. IID EXPONENTIAL FAMILY");
    println!("   Formula: log p(x₁,...,xₙ|θ) = η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)");
    demonstrate_iid_ir()?;

    Ok(())
}

fn demonstrate_normal_ir() -> Result<(), Box<dyn std::error::Error>> {
    // Create a symbolic representation of Normal log-density
    // log p(x|μ,σ²) = -½log(2πσ²) - (x-μ)²/(2σ²)

    let before = create_normal_log_density_ir();

    println!("   BEFORE optimization:");
    println!("   {}", format_expression_full(&before));
    println!("   Complexity: {} operations", before.complexity());

    let after = before.optimize_with_egglog()?;

    println!("   AFTER egglog optimization:");
    println!("   {}", format_expression_full(&after));
    println!("   Complexity: {} operations", after.complexity());

    let reduction = before.complexity().saturating_sub(after.complexity());
    println!(
        "   Reduction: {} operations ({:.1}%)",
        reduction,
        (reduction as f64 / before.complexity() as f64) * 100.0
    );

    // Verify mathematical equivalence
    verify_equivalence(&before, &after)?;

    Ok(())
}

fn demonstrate_poisson_ir() -> Result<(), Box<dyn std::error::Error>> {
    // Create a symbolic representation of Poisson log-density
    // In exponential family form: η·T(x) - A(η) + log h(x)
    // where η = log(λ), T(x) = x, A(η) = exp(η) = λ, h(x) = 1/x!

    let before = create_poisson_log_density_ir();

    println!("   BEFORE optimization:");
    println!("   {}", format_expression_full(&before));
    println!("   Complexity: {} operations", before.complexity());

    let after = before.optimize_with_egglog()?;

    println!("   AFTER egglog optimization:");
    println!("   {}", format_expression_full(&after));
    println!("   Complexity: {} operations", after.complexity());

    let reduction = before.complexity().saturating_sub(after.complexity());
    println!(
        "   Reduction: {} operations ({:.1}%)",
        reduction,
        (reduction as f64 / before.complexity() as f64) * 100.0
    );

    verify_equivalence(&before, &after)?;

    Ok(())
}

fn demonstrate_gamma_ir() -> Result<(), Box<dyn std::error::Error>> {
    // Create a symbolic representation of Gamma log-density
    // In exponential family form: η·T(x) - A(η) + log h(x)
    // where η = [α-1, -β], T(x) = [log(x), x], A(η) = log(Γ(η₁+1)) - (η₁+1)log(-η₂)

    let before = create_gamma_log_density_ir();

    println!("   BEFORE optimization:");
    println!("   {}", format_expression_full(&before));
    println!("   Complexity: {} operations", before.complexity());

    let after = before.optimize_with_egglog()?;

    println!("   AFTER egglog optimization:");
    println!("   {}", format_expression_full(&after));
    println!("   Complexity: {} operations", after.complexity());

    let reduction = before.complexity().saturating_sub(after.complexity());
    println!(
        "   Reduction: {} operations ({:.1}%)",
        reduction,
        (reduction as f64 / before.complexity() as f64) * 100.0
    );

    verify_equivalence(&before, &after)?;

    Ok(())
}

fn demonstrate_complex_exp_fam_ir() -> Result<(), Box<dyn std::error::Error>> {
    // Create a complex exponential family expression with many simplification opportunities
    let before = create_complex_exponential_family_ir();

    println!("   BEFORE optimization:");
    println!("   {}", format_expression_full(&before));
    println!("   Complexity: {} operations", before.complexity());

    let after = before.optimize_with_egglog()?;

    println!("   AFTER egglog optimization:");
    println!("   {}", format_expression_full(&after));
    println!("   Complexity: {} operations", after.complexity());

    let reduction = before.complexity().saturating_sub(after.complexity());
    println!(
        "   Reduction: {} operations ({:.1}%)",
        reduction,
        (reduction as f64 / before.complexity() as f64) * 100.0
    );

    verify_equivalence(&before, &after)?;

    Ok(())
}

fn demonstrate_iid_ir() -> Result<(), Box<dyn std::error::Error>> {
    // Create IID exponential family expression: η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
    let before = create_iid_exponential_family_ir();

    println!("   BEFORE optimization:");
    println!("   {}", format_expression_full(&before));
    println!("   Complexity: {} operations", before.complexity());

    let after = before.optimize_with_egglog()?;

    println!("   AFTER egglog optimization:");
    println!("   {}", format_expression_full(&after));
    println!("   Complexity: {} operations", after.complexity());

    let reduction = before.complexity().saturating_sub(after.complexity());
    println!(
        "   Reduction: {} operations ({:.1}%)",
        reduction,
        (reduction as f64 / before.complexity() as f64) * 100.0
    );

    verify_equivalence(&before, &after)?;

    Ok(())
}

// Helper functions to create symbolic IR expressions

fn create_normal_log_density_ir() -> Expr {
    // log p(x|μ,σ²) = -½log(2πσ²) - (x-μ)²/(2σ²)
    // Expanded with redundant operations for demonstration

    let x = Expr::Var("x".to_string());
    let mu = Expr::Var("mu".to_string());
    let sigma_sq = Expr::Var("sigma_sq".to_string());

    // -½log(2πσ²) = -½(log(2π) + log(σ²))
    let log_2pi = Expr::Ln(Box::new(Expr::Mul(
        Box::new(Expr::Const(2.0)),
        Box::new(Expr::Const(std::f64::consts::PI)),
    )));
    let log_sigma_sq = Expr::Ln(Box::new(sigma_sq.clone()));
    let log_term = Expr::Mul(
        Box::new(Expr::Const(-0.5)),
        Box::new(Expr::Add(Box::new(log_2pi), Box::new(log_sigma_sq))),
    );

    // -(x-μ)²/(2σ²) with redundant operations
    let x_minus_mu = Expr::Sub(Box::new(x), Box::new(mu));
    let x_minus_mu_sq = Expr::Pow(Box::new(x_minus_mu), Box::new(Expr::Const(2.0)));
    let two_sigma_sq = Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(sigma_sq));
    let quadratic_term = Expr::Neg(Box::new(Expr::Div(
        Box::new(x_minus_mu_sq),
        Box::new(two_sigma_sq),
    )));

    // Add redundant operations that should be simplified
    let redundant_zero = Expr::Sub(
        Box::new(Expr::Var("dummy".to_string())),
        Box::new(Expr::Var("dummy".to_string())),
    );
    let redundant_one = Expr::Div(
        Box::new(Expr::Var("dummy2".to_string())),
        Box::new(Expr::Var("dummy2".to_string())),
    );

    Expr::Add(
        Box::new(Expr::Add(Box::new(log_term), Box::new(quadratic_term))),
        Box::new(Expr::Mul(Box::new(redundant_zero), Box::new(redundant_one))),
    )
}

fn create_poisson_log_density_ir() -> Expr {
    // Poisson: log p(k|λ) = k·log(λ) - λ - log(k!)
    // In exponential family form: η·T(k) - A(η) + log h(k)
    // where η = log(λ), T(k) = k, A(η) = exp(η), h(k) = 1/k!

    let k = Expr::Var("k".to_string());
    let lambda = Expr::Var("lambda".to_string());

    // k·log(λ) with some redundant operations
    let log_lambda = Expr::Ln(Box::new(lambda.clone()));
    let k_log_lambda = Expr::Mul(Box::new(k.clone()), Box::new(log_lambda));

    // -λ
    let neg_lambda = Expr::Neg(Box::new(lambda));

    // -log(k!) - simplified as -log_factorial(k)
    let log_k_factorial = Expr::Var("log_k_factorial".to_string());
    let neg_log_factorial = Expr::Neg(Box::new(log_k_factorial));

    // Add some operations that should simplify
    let zero_term = Expr::Mul(
        Box::new(Expr::Const(0.0)),
        Box::new(Expr::Var("anything".to_string())),
    );
    let one_mult = Expr::Mul(Box::new(Expr::Const(1.0)), Box::new(k_log_lambda.clone()));

    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Add(Box::new(one_mult), Box::new(neg_lambda))),
            Box::new(neg_log_factorial),
        )),
        Box::new(zero_term),
    )
}

fn create_gamma_log_density_ir() -> Expr {
    // Gamma: log p(x|α,β) = (α-1)·log(x) - β·x + α·log(β) - log(Γ(α))

    let x = Expr::Var("x".to_string());
    let alpha = Expr::Var("alpha".to_string());
    let beta = Expr::Var("beta".to_string());

    // (α-1)·log(x)
    let alpha_minus_1 = Expr::Sub(Box::new(alpha.clone()), Box::new(Expr::Const(1.0)));
    let log_x = Expr::Ln(Box::new(x.clone()));
    let term1 = Expr::Mul(Box::new(alpha_minus_1), Box::new(log_x));

    // -β·x
    let term2 = Expr::Neg(Box::new(Expr::Mul(Box::new(beta.clone()), Box::new(x))));

    // α·log(β)
    let log_beta = Expr::Ln(Box::new(beta));
    let term3 = Expr::Mul(Box::new(alpha.clone()), Box::new(log_beta));

    // -log(Γ(α))
    let log_gamma_alpha = Expr::Var("log_gamma_alpha".to_string());
    let term4 = Expr::Neg(Box::new(log_gamma_alpha));

    // Add redundant operations
    let exp_ln_identity = Expr::Exp(Box::new(Expr::Ln(Box::new(alpha))));
    let redundant_add = Expr::Add(Box::new(Expr::Const(0.0)), Box::new(exp_ln_identity));

    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Add(Box::new(term1), Box::new(term2))),
            Box::new(Expr::Add(Box::new(term3), Box::new(term4))),
        )),
        Box::new(Expr::Sub(
            Box::new(redundant_add),
            Box::new(Expr::Var("alpha".to_string())),
        )),
    )
}

fn create_complex_exponential_family_ir() -> Expr {
    // Create a complex expression with many simplification opportunities
    let x = Expr::Var("x".to_string());
    let eta1 = Expr::Var("eta1".to_string());
    let eta2 = Expr::Var("eta2".to_string());

    // η₁·x + η₂·x² - A(η) with redundant operations
    let x_squared = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));

    // Add many redundant operations that should be simplified
    let term1 = Expr::Add(
        Box::new(Expr::Mul(Box::new(eta1.clone()), Box::new(x.clone()))),
        Box::new(Expr::Mul(
            Box::new(Expr::Const(0.0)),
            Box::new(Expr::Var("dummy".to_string())),
        )),
    );

    let term2 = Expr::Mul(
        Box::new(Expr::Mul(Box::new(eta2), Box::new(Expr::Const(1.0)))),
        Box::new(x_squared),
    );

    // ln(exp(something)) - should simplify to something
    let log_partition = Expr::Ln(Box::new(Expr::Exp(Box::new(Expr::Add(
        Box::new(eta1),
        Box::new(Expr::Const(1.0)),
    )))));

    // (a + b) * x - a * x - b * x should simplify to 0
    let distributive_zero = Expr::Sub(
        Box::new(Expr::Mul(
            Box::new(Expr::Add(
                Box::new(Expr::Var("a".to_string())),
                Box::new(Expr::Var("b".to_string())),
            )),
            Box::new(x.clone()),
        )),
        Box::new(Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("a".to_string())),
                Box::new(x.clone()),
            )),
            Box::new(Expr::Mul(Box::new(Expr::Var("b".to_string())), Box::new(x))),
        )),
    );

    Expr::Add(
        Box::new(Expr::Sub(
            Box::new(Expr::Add(Box::new(term1), Box::new(term2))),
            Box::new(log_partition),
        )),
        Box::new(distributive_zero),
    )
}

fn create_iid_exponential_family_ir() -> Expr {
    // IID case: η·∑ᵢT(xᵢ) - n·A(η) + ∑ᵢlog h(xᵢ)
    // Simplified to: η·(T(x₁) + T(x₂) + T(x₃)) - 3·A(η) + log h(x₁) + log h(x₂) + log h(x₃)

    let eta = Expr::Var("eta".to_string());
    let n = Expr::Const(3.0); // 3 samples

    // ∑ᵢT(xᵢ) = T(x₁) + T(x₂) + T(x₃)
    let t_x1 = Expr::Var("T_x1".to_string());
    let t_x2 = Expr::Var("T_x2".to_string());
    let t_x3 = Expr::Var("T_x3".to_string());
    let sum_sufficient_stats = Expr::Add(
        Box::new(Expr::Add(Box::new(t_x1), Box::new(t_x2))),
        Box::new(t_x3),
    );

    // η·∑ᵢT(xᵢ)
    let eta_dot_sum = Expr::Mul(Box::new(eta.clone()), Box::new(sum_sufficient_stats));

    // n·A(η)
    let log_partition = Expr::Var("A_eta".to_string());
    let n_times_partition = Expr::Mul(Box::new(n), Box::new(log_partition));

    // ∑ᵢlog h(xᵢ)
    let log_h_x1 = Expr::Var("log_h_x1".to_string());
    let log_h_x2 = Expr::Var("log_h_x2".to_string());
    let log_h_x3 = Expr::Var("log_h_x3".to_string());
    let sum_base_measures = Expr::Add(
        Box::new(Expr::Add(Box::new(log_h_x1), Box::new(log_h_x2))),
        Box::new(log_h_x3),
    );

    // Add redundant operations
    let redundant_exp_ln = Expr::Sub(
        Box::new(Expr::Exp(Box::new(Expr::Ln(Box::new(eta))))),
        Box::new(Expr::Var("eta".to_string())),
    );

    Expr::Add(
        Box::new(Expr::Add(
            Box::new(Expr::Sub(
                Box::new(eta_dot_sum),
                Box::new(n_times_partition),
            )),
            Box::new(sum_base_measures),
        )),
        Box::new(redundant_exp_ln),
    )
}

// Helper functions

fn format_expression_tree(expr: &Expr, max_depth: usize) -> String {
    if max_depth == 0 {
        return "...".to_string();
    }

    match expr {
        Expr::Const(c) => format!("{c:.2}"),
        Expr::Var(name) => name.clone(),
        Expr::Add(left, right) => format!(
            "({} + {})",
            format_expression_tree(left, max_depth - 1),
            format_expression_tree(right, max_depth - 1)
        ),
        Expr::Sub(left, right) => format!(
            "({} - {})",
            format_expression_tree(left, max_depth - 1),
            format_expression_tree(right, max_depth - 1)
        ),
        Expr::Mul(left, right) => format!(
            "({} * {})",
            format_expression_tree(left, max_depth - 1),
            format_expression_tree(right, max_depth - 1)
        ),
        Expr::Div(left, right) => format!(
            "({} / {})",
            format_expression_tree(left, max_depth - 1),
            format_expression_tree(right, max_depth - 1)
        ),
        Expr::Pow(base, exp) => format!(
            "{}^{}",
            format_expression_tree(base, max_depth - 1),
            format_expression_tree(exp, max_depth - 1)
        ),
        Expr::Ln(inner) => format!("ln({})", format_expression_tree(inner, max_depth - 1)),
        Expr::Exp(inner) => format!("exp({})", format_expression_tree(inner, max_depth - 1)),
        Expr::Sqrt(inner) => format!("sqrt({})", format_expression_tree(inner, max_depth - 1)),
        Expr::Sin(inner) => format!("sin({})", format_expression_tree(inner, max_depth - 1)),
        Expr::Cos(inner) => format!("cos({})", format_expression_tree(inner, max_depth - 1)),
        Expr::Neg(inner) => format!("-({})", format_expression_tree(inner, max_depth - 1)),
    }
}

fn format_expression_full(expr: &Expr) -> String {
    match expr {
        Expr::Const(c) => format!("{c:.2}"),
        Expr::Var(name) => name.clone(),
        Expr::Add(left, right) => format!(
            "({} + {})",
            format_expression_full(left),
            format_expression_full(right)
        ),
        Expr::Sub(left, right) => format!(
            "({} - {})",
            format_expression_full(left),
            format_expression_full(right)
        ),
        Expr::Mul(left, right) => format!(
            "({} * {})",
            format_expression_full(left),
            format_expression_full(right)
        ),
        Expr::Div(left, right) => format!(
            "({} / {})",
            format_expression_full(left),
            format_expression_full(right)
        ),
        Expr::Pow(base, exp) => format!(
            "{}^{}",
            format_expression_full(base),
            format_expression_full(exp)
        ),
        Expr::Ln(inner) => format!("ln({})", format_expression_full(inner)),
        Expr::Exp(inner) => format!("exp({})", format_expression_full(inner)),
        Expr::Sqrt(inner) => format!("sqrt({})", format_expression_full(inner)),
        Expr::Sin(inner) => format!("sin({})", format_expression_full(inner)),
        Expr::Cos(inner) => format!("cos({})", format_expression_full(inner)),
        Expr::Neg(inner) => format!("-({})", format_expression_full(inner)),
    }
}

fn verify_equivalence(before: &Expr, after: &Expr) -> Result<(), Box<dyn std::error::Error>> {
    let test_values = HashMap::from([
        ("x".to_string(), 2.5),
        ("mu".to_string(), 1.0),
        ("sigma_sq".to_string(), 4.0),
        ("k".to_string(), 3.0),
        ("lambda".to_string(), 2.0),
        ("log_k_factorial".to_string(), 1.79),
        ("alpha".to_string(), 2.0),
        ("beta".to_string(), 1.5),
        ("log_gamma_alpha".to_string(), 0.0),
        ("eta".to_string(), 0.5),
        ("eta1".to_string(), 1.2),
        ("eta2".to_string(), -0.3),
        ("T_x1".to_string(), 1.0),
        ("T_x2".to_string(), 2.0),
        ("T_x3".to_string(), 1.5),
        ("A_eta".to_string(), 1.65),
        ("log_h_x1".to_string(), -0.5),
        ("log_h_x2".to_string(), -1.0),
        ("log_h_x3".to_string(), -0.7),
        ("a".to_string(), 1.0),
        ("b".to_string(), 2.0),
        ("dummy".to_string(), 5.0),
        ("dummy2".to_string(), 3.0),
        ("anything".to_string(), 10.0),
    ]);

    match (before.evaluate(&test_values), after.evaluate(&test_values)) {
        (Ok(before_val), Ok(after_val)) => {
            let error = (before_val - after_val).abs();
            if error < 1e-10 {
                println!("   ✓ Mathematical equivalence verified (error: {error:.2e})");
            } else {
                println!("   ⚠ Mathematical equivalence check failed (error: {error:.2e})");
                println!("     Before: {before_val:.6}, After: {after_val:.6}");
            }
        }
        (Err(e1), Err(e2)) => {
            println!("   ⚠ Both expressions failed to evaluate: {e1:?}, {e2:?}");
        }
        (Ok(val), Err(e)) => {
            println!("   ⚠ Only original expression evaluated: {val:.6}, optimized failed: {e:?}");
        }
        (Err(e), Ok(val)) => {
            println!("   ⚠ Only optimized expression evaluated: {val:.6}, original failed: {e:?}");
        }
    }

    Ok(())
}
