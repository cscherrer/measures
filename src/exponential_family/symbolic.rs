//! Symbolic optimization for exponential family distributions.
//!
//! This module provides traits and utilities for generating optimized functions
//! from symbolic mathematical expressions. It enables compile-time and runtime
//! optimization of log-density computations by:
//!
//! 1. Representing log-densities symbolically
//! 2. Simplifying expressions algebraically
//! 3. Extracting constant subexpressions
//! 4. Generating optimized executable code
//!
//! The symbolic optimization can provide significant performance improvements
//! over traditional caching by eliminating all runtime overhead.

use crate::exponential_family::traits::{ExponentialFamily, PrecomputeCache};
use rusymbols::Expression;
use std::collections::HashMap;

/// Trait for distributions that can be symbolically optimized.
pub trait SymbolicOptimizer<X, F>
where
    F: num_traits::Float + std::fmt::Debug,
{
    /// Create a symbolic representation of the log-density function.
    fn symbolic_log_density(&self) -> SymbolicLogDensity;
    
    /// Generate an optimized function from the symbolic representation.
    fn generate_optimized_function(&self) -> OptimizedFunction<X, F>;
}

/// A symbolic representation of a log-density function.
pub struct SymbolicLogDensity {
    /// The symbolic expression for the log-density
    pub expression: Expression,
    /// Parameter values (for code generation)
    pub parameters: HashMap<String, f64>,
    /// Variable names that remain symbolic (e.g., "x")
    pub variables: Vec<String>,
}

impl SymbolicLogDensity {
    /// Create a new symbolic log-density representation.
    pub fn new(
        expression: Expression, 
        parameters: HashMap<String, f64>, 
        variables: Vec<String>
    ) -> Self {
        Self {
            expression,
            parameters,
            variables,
        }
    }
    
    /// Evaluate the symbolic expression at given variable values.
    pub fn evaluate(&self, vars: &HashMap<&str, f64>) -> Option<f64> {
        self.expression.eval_args(vars)
    }
    
    /// Evaluate the symbolic expression for a single variable.
    pub fn evaluate_single(&self, var_name: &str, value: f64) -> Option<f64> {
        let mut vars = self.parameters.iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<HashMap<&str, f64>>();
        vars.insert(var_name, value);
        self.expression.eval_args(&vars)
    }
}

/// An optimized function generated from symbolic analysis.
pub struct OptimizedFunction<X, F> {
    /// The generated function as a closure
    pub function: Box<dyn Fn(&X) -> F>,
    /// Pre-computed constants used in the function
    pub constants: HashMap<String, f64>,
    /// The original symbolic expression (for documentation)
    pub source_expression: String,
}

impl<X, F> OptimizedFunction<X, F> 
where
    F: num_traits::Float,
{
    /// Create a new optimized function.
    pub fn new(
        function: Box<dyn Fn(&X) -> F>,
        constants: HashMap<String, f64>,
        source_expression: String,
    ) -> Self {
        Self {
            function,
            constants,
            source_expression,
        }
    }
    
    /// Call the optimized function.
    pub fn call(&self, x: &X) -> F {
        (self.function)(x)
    }
}

/// Extension trait to add symbolic optimization to existing distributions.
pub trait SymbolicExtension<X, F>: ExponentialFamily<X, F> + PrecomputeCache<X, F>
where
    X: Clone,
    F: num_traits::Float + std::fmt::Debug,
{
    /// Generate an optimized log-density function using symbolic analysis.
    fn symbolic_optimize(&self) -> OptimizedFunction<X, F>
    where
        Self: SymbolicOptimizer<X, F>,
    {
        self.generate_optimized_function()
    }
}

// Blanket implementation for all types that implement the required traits
impl<T, X, F> SymbolicExtension<X, F> for T
where
    T: ExponentialFamily<X, F> + PrecomputeCache<X, F>,
    X: Clone,
    F: num_traits::Float + std::fmt::Debug,
{
}

/// Utility functions for symbolic optimization.
pub mod utils {
    use super::*;
    
    /// Create a symbolic variable.
    pub fn symbolic_var(name: &str) -> Expression {
        Expression::new_var(name)
    }
    
    /// Create a symbolic constant.
    pub fn symbolic_const(value: f64) -> Expression {
        Expression::new_val(value)
    }
    
    /// Build a quadratic term: -(x - mu)² / (2 * sigma²)
    pub fn quadratic_term(
        x: &Expression, 
        mu: f64, 
        sigma: f64
    ) -> Expression {
        let mu_expr = Expression::new_val(mu);
        let sigma_squared = Expression::new_val(sigma * sigma);
        let two = Expression::new_val(2.0);
        
        let x_minus_mu = x.clone() - mu_expr;
        -(x_minus_mu.clone() * x_minus_mu) / (two * sigma_squared)
    }
    
    /// Build a linear term: coeff * x
    pub fn linear_term(x: &Expression, coeff: f64) -> Expression {
        Expression::new_val(coeff) * x.clone()
    }
    
    /// Extract numerical constants from an expression for code generation.
    pub fn extract_constants(_expr: &Expression) -> HashMap<String, f64> {
        // This is a simplified implementation - a full version would
        // traverse the expression tree and extract all constants
        HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbolic_log_density_creation() {
        let x = Expression::new_var("x");
        let mu = Expression::new_val(1.0);
        let expr = x - mu;
        
        let mut params = HashMap::new();
        params.insert("mu".to_string(), 1.0);
        
        let symbolic = SymbolicLogDensity::new(
            expr,
            params,
            vec!["x".to_string()]
        );
        
        // Test evaluation
        let result = symbolic.evaluate_single("x", 2.0);
        assert_eq!(result, Some(1.0)); // 2.0 - 1.0 = 1.0
    }
    
    #[test]
    fn test_quadratic_term() {
        let x = Expression::new_var("x");
        let quad = utils::quadratic_term(&x, 0.0, 1.0);
        
        let mut vars = HashMap::new();
        vars.insert("x", 2.0);
        
        let result = quad.eval_args(&vars);
        assert_eq!(result, Some(-2.0)); // -(2-0)²/(2*1²) = -4/2 = -2
    }
} 