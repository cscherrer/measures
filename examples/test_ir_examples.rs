use measures::exponential_family::symbolic_ir::Expr;
use measures::exponential_family::egglog_optimizer::{EgglogOptimizer, EgglogOptimize};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Examining Generated IR for Mathematical Expressions");
    println!("====================================================\n");

    // Test cases to examine
    let test_cases = vec![
        ("Simple addition", Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0))
        )),
        ("Multiplication by zero", Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0))
        )),
        ("Multiplication by one", Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(1.0))
        )),
        ("Power of one", Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(1.0))
        )),
        ("Power of zero", Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0))
        )),
        ("Square (x^2)", Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0))
        )),
        ("Logarithm of exponential", Expr::Ln(
            Box::new(Expr::Exp(Box::new(Expr::Var("x".to_string()))))
        )),
        ("Exponential of logarithm", Expr::Exp(
            Box::new(Expr::Ln(Box::new(Expr::Var("x".to_string()))))
        )),
        ("Double negation", Expr::Neg(
            Box::new(Expr::Neg(Box::new(Expr::Var("x".to_string()))))
        )),
        ("Complex expression", Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Const(2.0)),
                Box::new(Expr::Pow(
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Const(2.0))
                ))
            )),
            Box::new(Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(3.0)),
                    Box::new(Expr::Var("x".to_string()))
                )),
                Box::new(Expr::Const(1.0))
            ))
        )),
        ("Distributive law candidate", Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("a".to_string())),
                Box::new(Expr::Var("x".to_string()))
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Var("b".to_string())),
                Box::new(Expr::Var("x".to_string()))
            ))
        )),
        ("Associativity candidate", Expr::Add(
            Box::new(Expr::Add(
                Box::new(Expr::Var("a".to_string())),
                Box::new(Expr::Var("b".to_string()))
            )),
            Box::new(Expr::Var("c".to_string()))
        )),
        ("Logarithm properties", Expr::Add(
            Box::new(Expr::Ln(Box::new(Expr::Var("a".to_string())))),
            Box::new(Expr::Ln(Box::new(Expr::Var("b".to_string()))))
        )),
        ("Exponential properties", Expr::Mul(
            Box::new(Expr::Exp(Box::new(Expr::Var("a".to_string())))),
            Box::new(Expr::Exp(Box::new(Expr::Var("b".to_string()))))
        )),
        ("Trigonometric identity", Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Sin(Box::new(Expr::Var("x".to_string())))),
                Box::new(Expr::Const(2.0))
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Cos(Box::new(Expr::Var("x".to_string())))),
                Box::new(Expr::Const(2.0))
            ))
        )),
    ];

    for (description, expr) in test_cases {
        println!("📊 Testing: {}", description);
        println!("   Original: {}", format_expr(&expr));
        println!("   Complexity: {}", expr.complexity());
        
        // Test basic simplification
        let basic_simplified = expr.clone().simplify();
        println!("   Basic simplified: {}", format_expr(&basic_simplified));
        println!("   Basic complexity: {}", basic_simplified.complexity());
        
        // Test egglog optimization
        match expr.optimize_with_egglog() {
            Ok(egglog_optimized) => {
                println!("   Egglog optimized: {}", format_expr(&egglog_optimized));
                println!("   Egglog complexity: {}", egglog_optimized.complexity());
                
                // Test if they're functionally equivalent
                let vars = HashMap::from([
                    ("x".to_string(), 2.5),
                    ("a".to_string(), 1.5),
                    ("b".to_string(), 3.0),
                    ("c".to_string(), 0.5),
                ]);
                
                match (expr.evaluate(&vars), egglog_optimized.evaluate(&vars)) {
                    (Ok(orig), Ok(opt)) => {
                        let error = (orig - opt).abs();
                        println!("   Functional equivalence: {:.2e} error", error);
                    }
                    _ => println!("   Functional equivalence: evaluation error"),
                }
            }
            Err(e) => {
                println!("   Egglog optimization failed: {}", e);
            }
        }
        
        println!();
    }

    // Test the egglog IR generation directly
    println!("🔧 Testing Egglog IR Generation");
    println!("===============================\n");
    
    let simple_expr = Expr::Add(
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Const(0.0))
    );
    
    // Create an optimizer to test IR generation
    let mut optimizer = EgglogOptimizer::new()?;
    println!("Expression: {}", format_expr(&simple_expr));
    
    // Test the optimization process
    match optimizer.optimize(&simple_expr) {
        Ok(optimized) => {
            println!("Optimized: {}", format_expr(&optimized));
        }
        Err(e) => {
            println!("Optimization failed: {}", e);
        }
    }

    Ok(())
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Const(c) => format!("{}", c),
        Expr::Var(name) => name.clone(),
        Expr::Add(left, right) => format!("({} + {})", format_expr(left), format_expr(right)),
        Expr::Sub(left, right) => format!("({} - {})", format_expr(left), format_expr(right)),
        Expr::Mul(left, right) => format!("({} * {})", format_expr(left), format_expr(right)),
        Expr::Div(left, right) => format!("({} / {})", format_expr(left), format_expr(right)),
        Expr::Pow(base, exp) => format!("{}^{}", format_expr(base), format_expr(exp)),
        Expr::Ln(inner) => format!("ln({})", format_expr(inner)),
        Expr::Exp(inner) => format!("exp({})", format_expr(inner)),
        Expr::Sqrt(inner) => format!("sqrt({})", format_expr(inner)),
        Expr::Sin(inner) => format!("sin({})", format_expr(inner)),
        Expr::Cos(inner) => format!("cos({})", format_expr(inner)),
        Expr::Neg(inner) => format!("-({})", format_expr(inner)),
    }
} 