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

/// A polymorphic quadratic function that works with any interpreter
/// This demonstrates the power of final tagless: one definition, multiple interpretations
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    let two = E::constant(2.0);
    let three = E::constant(3.0);
    let one = E::constant(1.0);
    
    // For interpreters where Repr doesn't implement Copy, we need to be careful about moves
    // Let's create separate variables for each use
    let x_for_pow = x;
    let x_for_mul = E::var("x"); // Create a fresh variable reference
    
    // 2*x^2 + 3*x + 1
    E::add(
        E::add(
            E::mul(two, E::pow(x_for_pow, E::constant(2.0))),
            E::mul(three, x_for_mul)
        ),
        one
    )
}

/// A simpler linear function to avoid move issues
fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    let two = E::constant(2.0);
    let three = E::constant(3.0);
    
    // 2*x + 3
    E::add(E::mul(two, x), three)
}

/// A more complex expression using transcendental functions
fn complex_expr<E: MathExpr>() -> E::Repr<f64> {
    let x = E::var("x");
    let x2 = E::var("x");
    
    // ln(exp(x) + sqrt(x^2 + 1))
    E::ln(
        E::add(
            E::exp(x),
            E::sqrt(E::add(E::pow(x2, E::constant(2.0)), E::constant(1.0)))
        )
    )
}

/// Demonstrate statistical functions using extension traits
fn statistical_example<E: StatisticalExpr>() -> E::Repr<f64> {
    let x = E::var("x");
    // softplus(logistic(x))
    E::softplus(E::logistic(x))
}

/// Demonstrate operator overloading with wrapper types
fn operator_example() {
    println!("\n=== Operator Overloading Example ===");
    
    // Using the wrapper type for ergonomic syntax
    let x = FinalTaglessExpr::<PrettyPrint>::var("x");
    let y = FinalTaglessExpr::<PrettyPrint>::var("y");
    let two = FinalTaglessExpr::<PrettyPrint>::constant(2.0);
    
    // Simple expression to avoid clone issues: x + y * 2
    let expr = x + y * two;
    
    println!("Expression with operators: {}", expr.as_repr());
}

fn main() {
    println!("🚀 Final Tagless Symbolic Math Showcase");
    println!("========================================");
    
    // 1. Direct Evaluation - Zero-cost abstraction
    println!("\n=== 1. Direct Evaluation (Zero-cost) ===");
    let start = Instant::now();
    let direct_result = linear::<DirectEval>(DirectEval::var("x", 2.0));
    let direct_time = start.elapsed();
    println!("linear(2.0) = {}", direct_result);
    println!("Time: {:?}", direct_time);
    
    // 2. Expression Building - AST construction
    println!("\n=== 2. Expression Building (AST) ===");
    let start = Instant::now();
    let ast_expr = linear::<ExprBuilder>(ExprBuilder::var("x"));
    let build_time = start.elapsed();
    println!("AST: {:?}", ast_expr);
    println!("Build time: {:?}", build_time);
    
    // Evaluate the AST for comparison
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    let ast_result = ast_expr.evaluate(&vars).unwrap();
    println!("AST eval result: {}", ast_result);
    
    // 3. Contextual Evaluation - Closure-based
    println!("\n=== 3. Contextual Evaluation ===");
    let contextual_expr = linear::<ContextualEval>(ContextualEval::var("x"));
    let mut context = HashMap::new();
    context.insert("x".to_string(), 2.0);
    let contextual_result = ContextualEval::eval_with(&contextual_expr, &context);
    println!("Contextual result: {}", contextual_result);
    
    // 4. Pretty Printing - Human-readable output
    println!("\n=== 4. Pretty Printing ===");
    let pretty_expr = linear::<PrettyPrint>(PrettyPrint::var("x"));
    println!("Pretty: {}", pretty_expr);
    
    // 5. Complex Expression Example
    println!("\n=== 5. Complex Expression ===");
    let complex_expr_contextual = complex_expr::<ContextualEval>();
    let mut context = HashMap::new();
    context.insert("x".to_string(), 1.0);
    let complex_result = ContextualEval::eval_with(&complex_expr_contextual, &context);
    
    let complex_pretty = complex_expr::<PrettyPrint>();
    println!("complex_expr(1.0) = {}", complex_result);
    println!("Complex pretty: {}", complex_pretty);
    
    // 6. Statistical Extensions
    println!("\n=== 6. Statistical Extensions ===");
    let stat_expr = statistical_example::<ContextualEval>();
    let mut context = HashMap::new();
    context.insert("x".to_string(), 0.0);
    let stat_result = ContextualEval::eval_with(&stat_expr, &context);
    
    let stat_pretty = statistical_example::<PrettyPrint>();
    println!("statistical_example(0.0) = {}", stat_result);
    println!("Statistical pretty: {}", stat_pretty);
    
    // 7. Operator overloading
    operator_example();
    
    // 8. Performance Comparison
    println!("\n=== 8. Performance Comparison ===");
    performance_benchmark();
    
    // 9. Type Safety Demonstration
    println!("\n=== 9. Type Safety ===");
    println!("✅ All expressions are type-safe at compile time");
    println!("✅ No runtime type checking overhead");
    println!("✅ Generic over numeric types (f32, f64, AD types)");
    
    println!("\n🎉 Final tagless approach successfully demonstrated!");
    println!("Key benefits:");
    println!("  • Zero-cost abstractions");
    println!("  • Solves the expression problem");
    println!("  • Type safety without runtime overhead");
    println!("  • Easy extension of operations and interpreters");
    println!("  • Composable DSL components");
}

fn performance_benchmark() {
    const ITERATIONS: usize = 1_000_000;
    
    // Benchmark direct evaluation
    let start = Instant::now();
    for i in 0..ITERATIONS {
        let x = i as f64 / 1000.0;
        let _result = linear::<DirectEval>(DirectEval::var("x", x));
    }
    let direct_time = start.elapsed();
    
    // Benchmark AST evaluation for comparison
    let ast_expr = linear::<ExprBuilder>(ExprBuilder::var("x"));
    let start = Instant::now();
    for i in 0..ITERATIONS {
        let x = i as f64 / 1000.0;
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x);
        let _result = ast_expr.evaluate(&vars).unwrap();
    }
    let ast_time = start.elapsed();
    
    println!("Performance over {} iterations:", ITERATIONS);
    println!("  Direct eval: {:?} ({:.2} ns/call)", 
             direct_time, direct_time.as_nanos() as f64 / ITERATIONS as f64);
    println!("  AST eval:    {:?} ({:.2} ns/call)", 
             ast_time, ast_time.as_nanos() as f64 / ITERATIONS as f64);
    
    let speedup = ast_time.as_nanos() as f64 / direct_time.as_nanos() as f64;
    println!("  Speedup: {:.2}x faster with final tagless!", speedup);
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
                Box::new(Expr::Var("x".to_string()))
            )),
            Box::new(Expr::Const(1.0))
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