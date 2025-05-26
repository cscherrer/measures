//! Symbolic Expression Optimization
//!
//! This module provides optimization for symbolic mathematical expressions using
//! the EggLog equality saturation engine. It applies mathematical rewrite rules
//! to simplify expressions while preserving mathematical equivalence.

use crate::Expr;
use egglog::EGraph;

/// Advanced optimizer using egglog for symbolic expression simplification
pub struct EgglogOptimizer {
    egraph: EGraph,
    expr_counter: usize,
}

impl EgglogOptimizer {
    /// Create a new egglog optimizer with conservative mathematical rewrite rules
    pub fn new() -> Result<Self, egglog::Error> {
        let mut egraph = EGraph::default();

        // Define the mathematical expression language with CONSERVATIVE rewrite rules
        // to avoid memory explosion
        egraph.parse_and_run_program(
            None,
            r"
            (datatype Math
                (Const f64)
                (Var String)
                (Add Math Math)
                (Sub Math Math)
                (Mul Math Math)
                (Div Math Math)
                (Pow Math Math)
                (Ln Math)
                (Exp Math)
                (Sqrt Math)
                (Sin Math)
                (Cos Math)
                (Neg Math))
            
            ; CONSERVATIVE arithmetic identities - only simplifying rules
            (rewrite (Add ?x (Const 0.0)) ?x)
            (rewrite (Add (Const 0.0) ?x) ?x)
            (rewrite (Sub ?x (Const 0.0)) ?x)
            (rewrite (Sub ?x ?x) (Const 0.0))
            (rewrite (Mul ?x (Const 1.0)) ?x)
            (rewrite (Mul (Const 1.0) ?x) ?x)
            (rewrite (Mul ?x (Const 0.0)) (Const 0.0))
            (rewrite (Mul (Const 0.0) ?x) (Const 0.0))
            (rewrite (Div ?x (Const 1.0)) ?x)
            (rewrite (Div (Const 0.0) ?x) (Const 0.0))
            (rewrite (Pow ?x (Const 1.0)) ?x)
            (rewrite (Pow ?x (Const 0.0)) (Const 1.0))
            (rewrite (Pow (Const 1.0) ?x) (Const 1.0))
            
            ; CONSERVATIVE logarithm and exponential identities - only inverse pairs
            (rewrite (Ln (Exp ?x)) ?x)
            (rewrite (Exp (Ln ?x)) ?x)
            (rewrite (Ln (Const 1.0)) (Const 0.0))
            (rewrite (Exp (Const 0.0)) (Const 1.0))
            
            ; CONSERVATIVE power identities - only safe simplifications
            (rewrite (Pow ?x (Const 2.0)) (Mul ?x ?x))
            (rewrite (Sqrt ?x) (Pow ?x (Const 0.5)))
            
            ; CONSERVATIVE negation identities
            (rewrite (Neg (Neg ?x)) ?x)
            (rewrite (Neg (Const 0.0)) (Const 0.0))
            (rewrite (Add ?x (Neg ?x)) (Const 0.0))
            (rewrite (Sub ?x ?y) (Add ?x (Neg ?y)))
            
            ; ADVANCED mathematical identities - more sophisticated optimizations
            ; Distributive law: (a * x) + (b * x) -> (a + b) * x
            (rewrite (Add (Mul ?a ?x) (Mul ?b ?x)) (Mul (Add ?a ?b) ?x))
            (rewrite (Add (Mul ?x ?a) (Mul ?x ?b)) (Mul ?x (Add ?a ?b)))
            
            ; Logarithm properties: ln(a) + ln(b) -> ln(a * b)
            (rewrite (Add (Ln ?a) (Ln ?b)) (Ln (Mul ?a ?b)))
            (rewrite (Sub (Ln ?a) (Ln ?b)) (Ln (Div ?a ?b)))
            (rewrite (Mul (Const ?c) (Ln ?x)) (Ln (Pow ?x (Const ?c))))
            
            ; Exponential properties: exp(a) * exp(b) -> exp(a + b)
            (rewrite (Mul (Exp ?a) (Exp ?b)) (Exp (Add ?a ?b)))
            (rewrite (Div (Exp ?a) (Exp ?b)) (Exp (Sub ?a ?b)))
            (rewrite (Pow (Exp ?x) (Const ?c)) (Exp (Mul (Const ?c) ?x)))
            
            ; Trigonometric identities
            (rewrite (Add (Pow (Sin ?x) (Const 2.0)) (Pow (Cos ?x) (Const 2.0))) (Const 1.0))
            (rewrite (Add (Mul (Sin ?x) (Sin ?x)) (Mul (Cos ?x) (Cos ?x))) (Const 1.0))
            
            ; Associativity and commutativity (limited to avoid explosion)
            (rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)))
            (rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))
            (rewrite (Add ?a ?b) (Add ?b ?a))
            (rewrite (Mul ?a ?b) (Mul ?b ?a))
            
            ; Power laws
            (rewrite (Mul (Pow ?x ?a) (Pow ?x ?b)) (Pow ?x (Add ?a ?b)))
            (rewrite (Div (Pow ?x ?a) (Pow ?x ?b)) (Pow ?x (Sub ?a ?b)))
            (rewrite (Pow (Pow ?x ?a) ?b) (Pow ?x (Mul ?a ?b)))
            
            ; ENHANCED constant folding for common values
            (rewrite (Add (Const ?a) (Const ?b)) (Const (+ ?a ?b)))
            (rewrite (Mul (Const ?a) (Const ?b)) (Const (* ?a ?b)))
            (rewrite (Sub (Const ?a) (Const ?b)) (Const (- ?a ?b)))
            (rewrite (Div (Const ?a) (Const ?b)) (Const (/ ?a ?b)))
            
            ; POLYNOMIAL simplification rules
            ; Collect like terms: x + x -> 2*x
            (rewrite (Add ?x ?x) (Mul (Const 2.0) ?x))
            (rewrite (Add (Mul (Const ?a) ?x) (Mul (Const ?b) ?x)) (Mul (Const (+ ?a ?b)) ?x))
            (rewrite (Add (Mul (Const ?a) ?x) ?x) (Mul (Const (+ ?a 1.0)) ?x))
            (rewrite (Add ?x (Mul (Const ?a) ?x)) (Mul (Const (+ 1.0 ?a)) ?x))
            
            ; RATIONAL function simplification
            ; Division by same base: x/x -> 1
            (rewrite (Div ?x ?x) (Const 1.0))
            ; Multiplication and division: (a/b) * (c/d) -> (a*c)/(b*d)
            (rewrite (Mul (Div ?a ?b) (Div ?c ?d)) (Div (Mul ?a ?c) (Mul ?b ?d)))
            ; Division of division: (a/b)/c -> a/(b*c)
            (rewrite (Div (Div ?a ?b) ?c) (Div ?a (Mul ?b ?c)))
            
            ; ADVANCED algebraic identities
            ; Difference of squares: (a+b)*(a-b) -> a²-b²
            (rewrite (Mul (Add ?a ?b) (Sub ?a ?b)) (Sub (Mul ?a ?a) (Mul ?b ?b)))
            ; Perfect square: (a+b)² -> a² + 2ab + b² (simplified to avoid complexity)
            
            ; NUMERICAL stability improvements
            ; Avoid very small constants that might be zero
            
            ; LOGARITHMIC and EXPONENTIAL advanced rules (simplified)
            ; ln(x^n) -> n*ln(x)
            (rewrite (Ln (Pow ?x ?n)) (Mul ?n (Ln ?x)))
            ; exp(n*ln(x)) -> x^n
            (rewrite (Exp (Mul ?n (Ln ?x))) (Pow ?x ?n))
            ; ln(sqrt(x)) -> 0.5*ln(x)
            (rewrite (Ln (Sqrt ?x)) (Mul (Const 0.5) (Ln ?x)))
            
            ; TRIGONOMETRIC advanced identities (simplified)
            ; sin(-x) -> -sin(x)
            (rewrite (Sin (Neg ?x)) (Neg (Sin ?x)))
            ; cos(-x) -> cos(x)
            (rewrite (Cos (Neg ?x)) (Cos ?x))
            ; sin(0) -> 0, cos(0) -> 1 (only for exact zero)
            (rewrite (Sin (Const 0.0)) (Const 0.0))
            (rewrite (Cos (Const 0.0)) (Const 1.0))
        ",
        )?;

        Ok(Self {
            egraph,
            expr_counter: 0,
        })
    }

    /// Optimize an expression using egglog's equality saturation with limited iterations
    pub fn optimize(&mut self, expr: &Expr) -> Result<Expr, egglog::Error> {
        // Convert our Expr to egglog format
        let egglog_expr = Self::expr_to_egglog(expr)?;

        // Add the expression to the egraph
        let expr_id = format!("expr_{}", self.expr_counter);
        self.expr_counter += 1;

        self.egraph
            .parse_and_run_program(None, &format!("(let {expr_id} {egglog_expr})"))?;

        // Run equality saturation with LIMITED iterations to prevent explosion
        self.egraph.parse_and_run_program(None, "(run 5)")?;

        // Try to extract a simplified expression
        // For now, we'll use a simple approach: just extract the expression
        let extraction_program = format!("(extract {expr_id})");

        let result = self
            .egraph
            .parse_and_run_program(None, &extraction_program)?;

        // Parse the extracted result back to our Expr type
        if let Some(extracted) = result.into_iter().next() {
            let extracted_str = extracted.to_string();
            println!("DEBUG: Extracted from egglog: {extracted_str}");

            if let Some(optimized_expr) = Self::egglog_to_expr(&extracted_str) {
                return Ok(optimized_expr);
            }
        }

        // Fallback to original expression if extraction fails
        Ok(expr.clone())
    }

    /// Convert our Expr to egglog string representation
    fn expr_to_egglog(expr: &Expr) -> Result<String, egglog::Error> {
        match expr {
            Expr::Const(c) => Ok(format!("(Const {c:.1})")),
            Expr::Var(name) => Ok(format!("(Var \"{name}\")")),
            Expr::Add(left, right) => {
                let left_str = Self::expr_to_egglog(left)?;
                let right_str = Self::expr_to_egglog(right)?;
                Ok(format!("(Add {left_str} {right_str})"))
            }
            Expr::Sub(left, right) => {
                let left_str = Self::expr_to_egglog(left)?;
                let right_str = Self::expr_to_egglog(right)?;
                Ok(format!("(Sub {left_str} {right_str})"))
            }
            Expr::Mul(left, right) => {
                let left_str = Self::expr_to_egglog(left)?;
                let right_str = Self::expr_to_egglog(right)?;
                Ok(format!("(Mul {left_str} {right_str})"))
            }
            Expr::Div(left, right) => {
                let left_str = Self::expr_to_egglog(left)?;
                let right_str = Self::expr_to_egglog(right)?;
                Ok(format!("(Div {left_str} {right_str})"))
            }
            Expr::Pow(base, exp) => {
                let base_str = Self::expr_to_egglog(base)?;
                let exp_str = Self::expr_to_egglog(exp)?;
                Ok(format!("(Pow {base_str} {exp_str})"))
            }
            Expr::Ln(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Ln {inner_str})"))
            }
            Expr::Exp(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Exp {inner_str})"))
            }
            Expr::Sqrt(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Sqrt {inner_str})"))
            }
            Expr::Sin(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Sin {inner_str})"))
            }
            Expr::Cos(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Cos {inner_str})"))
            }
            Expr::Neg(inner) => {
                let inner_str = Self::expr_to_egglog(inner)?;
                Ok(format!("(Neg {inner_str})"))
            }
        }
    }

    /// Convert egglog string representation back to our Expr type
    /// Returns None if parsing fails (we'll fallback to original expression)
    fn egglog_to_expr(egglog_str: &str) -> Option<Expr> {
        // This is a simplified parser for the egglog output
        // In a full implementation, you'd want a proper parser
        let trimmed = egglog_str.trim();

        if trimmed.starts_with("(Const ") && trimmed.ends_with(')') {
            let value_str = &trimmed[7..trimmed.len() - 1];
            if let Ok(value) = value_str.parse::<f64>() {
                return Some(Expr::Const(value));
            }
        }

        if trimmed.starts_with("(Var \"") && trimmed.ends_with("\")") {
            let var_name = &trimmed[6..trimmed.len() - 2];
            return Some(Expr::Var(var_name.to_string()));
        }

        // Handle binary operations
        if trimmed.starts_with("(Add ") && trimmed.ends_with(')') {
            if let Some((left, right)) = Self::parse_binary_args(&trimmed[5..trimmed.len() - 1]) {
                if let (Some(left_expr), Some(right_expr)) =
                    (Self::egglog_to_expr(&left), Self::egglog_to_expr(&right))
                {
                    return Some(Expr::Add(Box::new(left_expr), Box::new(right_expr)));
                }
            }
        }

        if trimmed.starts_with("(Mul ") && trimmed.ends_with(')') {
            if let Some((left, right)) = Self::parse_binary_args(&trimmed[5..trimmed.len() - 1]) {
                if let (Some(left_expr), Some(right_expr)) =
                    (Self::egglog_to_expr(&left), Self::egglog_to_expr(&right))
                {
                    return Some(Expr::Mul(Box::new(left_expr), Box::new(right_expr)));
                }
            }
        }

        if trimmed.starts_with("(Sub ") && trimmed.ends_with(')') {
            if let Some((left, right)) = Self::parse_binary_args(&trimmed[5..trimmed.len() - 1]) {
                if let (Some(left_expr), Some(right_expr)) =
                    (Self::egglog_to_expr(&left), Self::egglog_to_expr(&right))
                {
                    return Some(Expr::Sub(Box::new(left_expr), Box::new(right_expr)));
                }
            }
        }

        if trimmed.starts_with("(Div ") && trimmed.ends_with(')') {
            if let Some((left, right)) = Self::parse_binary_args(&trimmed[5..trimmed.len() - 1]) {
                if let (Some(left_expr), Some(right_expr)) =
                    (Self::egglog_to_expr(&left), Self::egglog_to_expr(&right))
                {
                    return Some(Expr::Div(Box::new(left_expr), Box::new(right_expr)));
                }
            }
        }

        if trimmed.starts_with("(Pow ") && trimmed.ends_with(')') {
            if let Some((left, right)) = Self::parse_binary_args(&trimmed[5..trimmed.len() - 1]) {
                if let (Some(left_expr), Some(right_expr)) =
                    (Self::egglog_to_expr(&left), Self::egglog_to_expr(&right))
                {
                    return Some(Expr::Pow(Box::new(left_expr), Box::new(right_expr)));
                }
            }
        }

        // Handle unary operations
        if trimmed.starts_with("(Ln ") && trimmed.ends_with(')') {
            let inner = &trimmed[4..trimmed.len() - 1];
            if let Some(inner_expr) = Self::egglog_to_expr(inner) {
                return Some(Expr::Ln(Box::new(inner_expr)));
            }
        }

        if trimmed.starts_with("(Exp ") && trimmed.ends_with(')') {
            let inner = &trimmed[5..trimmed.len() - 1];
            if let Some(inner_expr) = Self::egglog_to_expr(inner) {
                return Some(Expr::Exp(Box::new(inner_expr)));
            }
        }

        if trimmed.starts_with("(Sin ") && trimmed.ends_with(')') {
            let inner = &trimmed[5..trimmed.len() - 1];
            if let Some(inner_expr) = Self::egglog_to_expr(inner) {
                return Some(Expr::Sin(Box::new(inner_expr)));
            }
        }

        if trimmed.starts_with("(Cos ") && trimmed.ends_with(')') {
            let inner = &trimmed[5..trimmed.len() - 1];
            if let Some(inner_expr) = Self::egglog_to_expr(inner) {
                return Some(Expr::Cos(Box::new(inner_expr)));
            }
        }

        if trimmed.starts_with("(Neg ") && trimmed.ends_with(')') {
            let inner = &trimmed[5..trimmed.len() - 1];
            if let Some(inner_expr) = Self::egglog_to_expr(inner) {
                return Some(Expr::Neg(Box::new(inner_expr)));
            }
        }

        // For now, return None for complex expressions we can't parse
        None
    }

    /// Parse binary operation arguments from egglog format
    /// This is a simple parser that handles nested parentheses
    fn parse_binary_args(args_str: &str) -> Option<(String, String)> {
        let mut paren_count = 0;
        let mut split_pos = None;

        for (i, ch) in args_str.char_indices() {
            match ch {
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                ' ' if paren_count == 0 && split_pos.is_none() => {
                    split_pos = Some(i);
                }
                _ => {}
            }
        }

        if let Some(pos) = split_pos {
            let left = args_str[..pos].trim().to_string();
            let right = args_str[pos + 1..].trim().to_string();
            Some((left, right))
        } else {
            None
        }
    }
}

/// Extension trait to add egglog optimization to expressions
pub trait EgglogOptimize {
    /// Optimize this expression using egglog
    fn optimize_with_egglog(&self) -> Result<Self, egglog::Error>
    where
        Self: Sized;
}

impl EgglogOptimize for Expr {
    fn optimize_with_egglog(&self) -> Result<Self, egglog::Error> {
        let mut optimizer = EgglogOptimizer::new()?;
        optimizer.optimize(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egglog_optimizer_creation() {
        let optimizer = EgglogOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_expr_to_egglog_conversion() {
        // Test simple constant
        let expr = Expr::Const(42.0);
        let egglog_str = EgglogOptimizer::expr_to_egglog(&expr).unwrap();
        assert_eq!(egglog_str, "(Const 42.0)");

        // Test variable
        let expr = Expr::Var("x".to_string());
        let egglog_str = EgglogOptimizer::expr_to_egglog(&expr).unwrap();
        assert_eq!(egglog_str, "(Var \"x\")");

        // Test addition
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(1.0)),
        );
        let egglog_str = EgglogOptimizer::expr_to_egglog(&expr).unwrap();
        assert_eq!(egglog_str, "(Add (Var \"x\") (Const 1.0))");
    }

    #[test]
    fn test_basic_optimization() {
        let mut optimizer = EgglogOptimizer::new().unwrap();

        // Test x + 0 -> x optimization
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0)),
        );

        let optimized = optimizer.optimize(&expr).unwrap();
        // The optimizer should simplify x + 0 to x
        assert_eq!(optimized, Expr::Var("x".to_string()));
    }
}
