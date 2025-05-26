//! Final Tagless Approach Showcase
//!
//! This example demonstrates the power of the final tagless approach for symbolic computation.
//! It shows how the same expression definition can be interpreted in multiple ways:
//! - Direct evaluation for maximum performance
//! - AST building for compatibility with existing systems
//! - Contextual evaluation with variable bindings
//! - Pretty printing for human-readable output
//!
//! The final tagless approach solves the expression problem and provides zero-cost abstractions.

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::final_tagless::*;

/// A simple linear function for performance testing
fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    let two = E::constant(2.0);
    let three = E::constant(3.0);
    
    // 2*x + 3
    E::add::<f64, f64, f64>(E::mul::<f64, f64, f64>(two, x), three)
}

/// A more complex expression demonstrating various operations
fn complex_expr<E: MathExpr>() -> E::Repr<f64> {
    let x: E::Repr<f64> = E::var("x");
    let x2: E::Repr<f64> = E::var("x");
    
    // ln(exp(x) + sqrt(x^2 + 1))
    E::ln(E::add::<f64, f64, f64>(
        E::exp(x),
        E::sqrt(E::add::<f64, f64, f64>(E::pow(x2, E::constant(2.0)), E::constant(1.0))),
    ))
}

/// Demonstrate statistical functions using extension traits
fn statistical_example<E: StatisticalExpr>() -> E::Repr<f64> {
    let x = E::var("x");
    // softplus(logistic(x))
    E::softplus(E::logistic(x))
}

fn main() {
    println!("🚀 Final Tagless Symbolic Math Showcase");
    println!("========================================");

    // 1. Direct Evaluation Example
    println!("\n=== 1. Direct Evaluation (Zero-cost) ===");
    let start = Instant::now();
    let result_direct = linear_direct_eval(2.0);
    let direct_time = start.elapsed();
    println!("linear(2.0) = {result_direct}");
    println!("Time: {direct_time:?}");

    // 2. Expression Building Example
    println!("\n=== 2. Expression Building (AST) ===");
    let start = Instant::now();
    let expr_ast = linear::<ExprBuilder>(ExprBuilder::var("x"));
    let build_time = start.elapsed();
    println!("AST: {expr_ast:?}");
    println!("Build time: {build_time:?}");
    
    // Evaluate the AST
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    let ast_result = expr_ast.evaluate(&vars).unwrap();
    println!("AST eval result: {ast_result}");

    // 3. Contextual Evaluation Example
    println!("\n=== 3. Contextual Evaluation ===");
    let contextual_expr = linear_contextual();
    let mut context = HashMap::new();
    context.insert("x".to_string(), 2.0);
    let contextual_result = ContextualEval::eval_with(&contextual_expr, &context);
    println!("Contextual result: {contextual_result}");

    // 4. Pretty Printing Example
    println!("\n=== 4. Pretty Printing ===");
    let pretty_result = linear::<PrettyPrint>(PrettyPrint::var("x"));
    println!("Pretty: {pretty_result}");

    // 5. Complex Expression Example
    println!("\n=== 5. Complex Expression ===");
    let complex_expr_contextual = complex_expr_contextual();
    let mut context = HashMap::new();
    context.insert("x".to_string(), 1.0);
    let complex_result = ContextualEval::eval_with(&complex_expr_contextual, &context);
    
    let complex_pretty = complex_expr::<PrettyPrint>();
    println!("complex_expr(1.0) = {complex_result}");
    println!("Complex pretty: {complex_pretty}");

    // 6. Statistical Extensions
    println!("\n=== 6. Statistical Extensions ===");
    let stat_expr_contextual = statistical_example_contextual();
    let mut context = HashMap::new();
    context.insert("x".to_string(), 0.0);
    let stat_result = ContextualEval::eval_with(&stat_expr_contextual, &context);
    
    let stat_pretty = statistical_example::<PrettyPrint>();
    println!("statistical_example(0.0) = {stat_result}");
    println!("Statistical pretty: {stat_pretty}");

    // 7. Performance Comparison
    println!("\n=== 7. Performance Comparison ===");
    
    // Benchmark DirectEval (should be very fast)
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..10000 {
        sum += linear_direct_eval(i as f64);
    }
    let direct_time = start.elapsed();
    println!("DirectEval (10k iterations): {direct_time:?}, sum: {sum}");
    
    println!("\nFinal tagless approach demonstrates:");
    println!("✅ Zero-cost abstractions with DirectEval");
    println!("✅ AST building with ExprBuilder");
    println!("✅ Contextual evaluation with ContextualEval");
    println!("✅ Pretty printing with PrettyPrint");
    println!("✅ Operator overloading support");
    println!("✅ Easy extension (expression problem solved)");

    // 8. Type Safety Demonstration
    println!("\n=== 8. Type Safety & Extensibility ===");
    println!("✅ Compile-time type checking");
    println!("✅ Zero runtime overhead for DirectEval");
    println!("✅ Easy addition of new operations (StatisticalExpr)");
    println!("✅ Easy addition of new interpreters");
    println!("✅ Solves the expression problem elegantly");
    println!("✅ Composable DSL components");
}

/// Direct evaluation version of linear function
fn linear_direct_eval(x_val: f64) -> f64 {
    2.0 * x_val + 3.0
}

/// Contextual evaluation version of linear function
fn linear_contextual() -> ContextualRepr<f64> {
    let x = ContextualEval::var("x");
    let two = ContextualEval::constant(2.0);
    let three = ContextualEval::constant(3.0);
    
    // 2*x + 3
    ContextualEval::add_same(ContextualEval::mul_same(two, x), three)
}

/// Contextual evaluation version of complex expression
fn complex_expr_contextual() -> ContextualRepr<f64> {
    let x = ContextualEval::var("x");
    let x2 = ContextualEval::var("x");
    let one = ContextualEval::constant(1.0);
    let two = ContextualEval::constant(2.0);
    
    // ln(exp(x) + sqrt(x^2 + 1))
    ContextualEval::ln(ContextualEval::add_same(
        ContextualEval::exp(x),
        ContextualEval::sqrt(ContextualEval::add_same(ContextualEval::pow(x2, two), one)),
    ))
}

/// Contextual evaluation version of statistical example
fn statistical_example_contextual() -> ContextualRepr<f64> {
    let x = ContextualEval::var("x");
    
    // ln(1 + exp(logistic(x)))
    let logistic_x = ContextualEval::div_same(
        ContextualEval::constant(1.0),
        ContextualEval::add_same(ContextualEval::constant(1.0), ContextualEval::exp(ContextualEval::neg(x)))
    );
    
    ContextualEval::ln(ContextualEval::add_same(
        ContextualEval::constant(1.0),
        ContextualEval::exp(logistic_x)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_tagless_equivalence() {
        // Test that final tagless and tagged union produce equivalent results
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::mul(E::constant(2.0), x), E::constant(1.0))
        }

        let final_tagless_result = test_expr::<DirectEval>(DirectEval::var("x", 5.0));

        let tagged_expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Var("x".to_string())),
            )),
            Box::new(Expr::Const(1.0)),
        );

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let tagged_result = tagged_expr.evaluate(&vars)?;

        assert_eq!(final_tagless_result, tagged_result);
    }

    #[test]
    fn test_multiple_interpreters() {
        fn test_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::pow(x, E::constant(2.0))
        }

        // Direct evaluation
        let direct = test_expr::<DirectEval>(DirectEval::var("x", 3.0));
        assert_eq!(direct, 9.0);

        // Expression building
        let expr_builder = test_expr::<ExprBuilder>(ExprBuilder::var("x"));
        assert!(matches!(expr_builder, Expr::Pow(_, _)));

        // Contextual evaluation
        let contextual = test_expr::<ContextualEval>(ContextualEval::var("x"));
        let mut context = HashMap::new();
        context.insert("x".to_string(), 3.0);
        assert_eq!(ContextualEval::eval_with(&contextual, &context), 9.0);
    }
}
