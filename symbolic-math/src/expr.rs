//! Custom Symbolic IR for JIT Compilation
//!
//! This module provides a custom intermediate representation (IR) for symbolic
//! mathematical expressions that is specifically designed for JIT compilation
//! with Cranelift. Unlike generic symbolic libraries, this IR provides:
//!
//! - Full expression tree introspection
//! - Direct mapping to Cranelift CLIF IR
//! - Optimized representation for exponential family distributions
//! - Constant folding and algebraic simplification
//! - Type-safe expression construction

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};

/// Global cache for expression simplification results
static SIMPLIFICATION_CACHE: std::sync::LazyLock<Arc<Mutex<HashMap<String, Expr>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Global cache for evaluation results
static EVALUATION_CACHE: std::sync::LazyLock<Arc<Mutex<HashMap<(String, String), f64>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Cache statistics for monitoring performance
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits for simplification operations
    pub simplification_hits: usize,
    /// Number of cache misses for simplification operations
    pub simplification_misses: usize,
    /// Number of cache hits for evaluation operations
    pub evaluation_hits: usize,
    /// Number of cache misses for evaluation operations
    pub evaluation_misses: usize,
}

impl CacheStats {
    /// Get the hit rate for simplification cache
    #[must_use]
    pub fn simplification_hit_rate(&self) -> f64 {
        let total = self.simplification_hits + self.simplification_misses;
        if total == 0 {
            0.0
        } else {
            self.simplification_hits as f64 / total as f64
        }
    }

    /// Get the hit rate for evaluation cache
    #[must_use]
    pub fn evaluation_hit_rate(&self) -> f64 {
        let total = self.evaluation_hits + self.evaluation_misses;
        if total == 0 {
            0.0
        } else {
            self.evaluation_hits as f64 / total as f64
        }
    }
}

/// Global cache statistics
static CACHE_STATS: std::sync::LazyLock<Arc<Mutex<CacheStats>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(CacheStats::default())));

/// Get current cache statistics
pub fn get_cache_stats() -> CacheStats {
    CACHE_STATS.lock().unwrap().clone()
}

/// Clear all caches (useful for testing or memory management)
pub fn clear_caches() {
    SIMPLIFICATION_CACHE.lock().unwrap().clear();
    EVALUATION_CACHE.lock().unwrap().clear();
    *CACHE_STATS.lock().unwrap() = CacheStats::default();
}

/// A symbolic expression in our custom IR
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant value
    Const(f64),
    /// A variable (e.g., "x", "mu", "sigma")
    Var(String),
    /// Addition: a + b
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: a - b
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: a * b
    Mul(Box<Expr>, Box<Expr>),
    /// Division: a / b
    Div(Box<Expr>, Box<Expr>),
    /// Power: a^b
    Pow(Box<Expr>, Box<Expr>),
    /// Natural logarithm: ln(a)
    Ln(Box<Expr>),
    /// Exponential: exp(a)
    Exp(Box<Expr>),
    /// Square root: sqrt(a)
    Sqrt(Box<Expr>),
    /// Sine: sin(a)
    Sin(Box<Expr>),
    /// Cosine: cos(a)
    Cos(Box<Expr>),
    /// Negation: -a
    Neg(Box<Expr>),
}

impl Expr {
    /// Create a constant expression
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Expr::Const(value)
    }

    /// Create a variable expression
    pub fn variable(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    /// Create an addition expression
    #[must_use]
    pub fn add(left: Expr, right: Expr) -> Self {
        Expr::Add(Box::new(left), Box::new(right))
    }

    /// Create a subtraction expression
    #[must_use]
    pub fn sub(left: Expr, right: Expr) -> Self {
        Expr::Sub(Box::new(left), Box::new(right))
    }

    /// Create a multiplication expression
    #[must_use]
    pub fn mul(left: Expr, right: Expr) -> Self {
        Expr::Mul(Box::new(left), Box::new(right))
    }

    /// Create a division expression
    #[must_use]
    pub fn div(left: Expr, right: Expr) -> Self {
        Expr::Div(Box::new(left), Box::new(right))
    }

    /// Create a power expression
    #[must_use]
    pub fn pow(base: Expr, exponent: Expr) -> Self {
        Expr::Pow(Box::new(base), Box::new(exponent))
    }

    /// Create a natural logarithm expression
    #[must_use]
    pub fn ln(expr: Expr) -> Self {
        Expr::Ln(Box::new(expr))
    }

    /// Create an exponential expression
    #[must_use]
    pub fn exp(expr: Expr) -> Self {
        Expr::Exp(Box::new(expr))
    }

    /// Create a square root expression
    #[must_use]
    pub fn sqrt(expr: Expr) -> Self {
        Expr::Sqrt(Box::new(expr))
    }

    /// Create a negation expression
    #[must_use]
    pub fn neg(expr: Expr) -> Self {
        Expr::Neg(Box::new(expr))
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Result<f64, EvalError> {
        // Fast path for simple expressions to reduce overhead
        match self {
            // Inline constants - no function call overhead
            Expr::Const(value) => Ok(*value),

            // Optimized variable lookup - avoid closure allocation
            Expr::Var(name) => match vars.get(name) {
                Some(&value) => Ok(value),
                None => Err(EvalError::UndefinedVariable(name.clone())),
            },

            // Fast path for simple binary operations with constants
            Expr::Add(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Const(a), Expr::Const(b)) => Ok(a + b),
                (Expr::Const(0.0), expr) => expr.evaluate(vars),
                (expr, Expr::Const(0.0)) => expr.evaluate(vars),
                _ => Ok(left.evaluate(vars)? + right.evaluate(vars)?),
            },

            Expr::Mul(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Const(a), Expr::Const(b)) => Ok(a * b),
                (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Ok(0.0),
                (Expr::Const(1.0), expr) => expr.evaluate(vars),
                (expr, Expr::Const(1.0)) => expr.evaluate(vars),
                _ => Ok(left.evaluate(vars)? * right.evaluate(vars)?),
            },

            // Standard evaluation for other operations
            Expr::Sub(left, right) => Ok(left.evaluate(vars)? - right.evaluate(vars)?),
            Expr::Div(left, right) => {
                let denominator = right.evaluate(vars)?;
                if denominator == 0.0 {
                    Err(EvalError::DivisionByZero)
                } else {
                    Ok(left.evaluate(vars)? / denominator)
                }
            }
            Expr::Pow(base, exponent) => Ok(base.evaluate(vars)?.powf(exponent.evaluate(vars)?)),
            Expr::Ln(expr) => {
                let value = expr.evaluate(vars)?;
                if value <= 0.0 {
                    Err(EvalError::InvalidLogarithm(value))
                } else {
                    Ok(value.ln())
                }
            }
            Expr::Exp(expr) => Ok(expr.evaluate(vars)?.exp()),
            Expr::Sqrt(expr) => {
                let value = expr.evaluate(vars)?;
                if value < 0.0 {
                    Err(EvalError::InvalidSquareRoot(value))
                } else {
                    Ok(value.sqrt())
                }
            }
            Expr::Sin(expr) => Ok(expr.evaluate(vars)?.sin()),
            Expr::Cos(expr) => Ok(expr.evaluate(vars)?.cos()),
            Expr::Neg(expr) => Ok(-expr.evaluate(vars)?),
        }
    }

    /// Ultra-fast evaluation for single-variable expressions
    /// Avoids `HashMap` overhead by passing the variable value directly
    pub fn evaluate_single_var(&self, var_name: &str, value: f64) -> Result<f64, EvalError> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Var(name) => {
                if name == var_name {
                    Ok(value)
                } else {
                    Err(EvalError::UndefinedVariable(name.clone()))
                }
            }
            Expr::Add(left, right) => Ok(left.evaluate_single_var(var_name, value)?
                + right.evaluate_single_var(var_name, value)?),
            Expr::Sub(left, right) => Ok(left.evaluate_single_var(var_name, value)?
                - right.evaluate_single_var(var_name, value)?),
            Expr::Mul(left, right) => Ok(left.evaluate_single_var(var_name, value)?
                * right.evaluate_single_var(var_name, value)?),
            Expr::Div(left, right) => {
                let denominator = right.evaluate_single_var(var_name, value)?;
                if denominator == 0.0 {
                    Err(EvalError::DivisionByZero)
                } else {
                    Ok(left.evaluate_single_var(var_name, value)? / denominator)
                }
            }
            Expr::Pow(base, exponent) => Ok(base
                .evaluate_single_var(var_name, value)?
                .powf(exponent.evaluate_single_var(var_name, value)?)),
            Expr::Ln(expr) => {
                let val = expr.evaluate_single_var(var_name, value)?;
                if val <= 0.0 {
                    Err(EvalError::InvalidLogarithm(val))
                } else {
                    Ok(val.ln())
                }
            }
            Expr::Exp(expr) => Ok(expr.evaluate_single_var(var_name, value)?.exp()),
            Expr::Sqrt(expr) => {
                let val = expr.evaluate_single_var(var_name, value)?;
                if val < 0.0 {
                    Err(EvalError::InvalidSquareRoot(val))
                } else {
                    Ok(val.sqrt())
                }
            }
            Expr::Sin(expr) => Ok(expr.evaluate_single_var(var_name, value)?.sin()),
            Expr::Cos(expr) => Ok(expr.evaluate_single_var(var_name, value)?.cos()),
            Expr::Neg(expr) => Ok(-expr.evaluate_single_var(var_name, value)?),
        }
    }

    /// Optimized batch evaluation that reuses allocations
    pub fn evaluate_batch_optimized(
        &self,
        var_name: &str,
        values: &[f64],
    ) -> Result<Vec<f64>, EvalError> {
        // Pre-allocate result vector
        let mut results = Vec::with_capacity(values.len());

        // Check if this is a single-variable expression for ultra-fast path
        let vars = self.variables();
        if vars.len() == 1 && vars[0] == var_name {
            // Ultra-fast path: no HashMap needed
            for &value in values {
                results.push(self.evaluate_single_var(var_name, value)?);
            }
        } else {
            // Standard path with HashMap reuse
            let mut var_map = HashMap::with_capacity(vars.len());

            // Pre-populate with any constants or other variables (if needed)
            for var in &vars {
                if var != var_name {
                    var_map.insert(var.clone(), 0.0); // Will be overwritten if needed
                }
            }

            for &value in values {
                var_map.insert(var_name.to_string(), value);
                results.push(self.evaluate(&var_map)?);
            }
        }

        Ok(results)
    }

    /// Specialized evaluation for polynomial expressions: ax^n + bx^(n-1) + ... + c
    /// This is much faster than general evaluation for polynomial patterns
    #[must_use]
    pub fn evaluate_polynomial(&self, var_name: &str, value: f64) -> Option<f64> {
        match self {
            // Constant term
            Expr::Const(c) => Some(*c),

            // Variable term
            Expr::Var(name) if name == var_name => Some(value),

            // Linear term: ax
            Expr::Mul(coeff, var) => match (coeff.as_ref(), var.as_ref()) {
                (Expr::Const(a), Expr::Var(name)) if name == var_name => Some(a * value),
                (Expr::Var(name), Expr::Const(a)) if name == var_name => Some(a * value),
                _ => None,
            },

            // Power term: x^n or ax^n
            Expr::Pow(base, exp) => match (base.as_ref(), exp.as_ref()) {
                (Expr::Var(name), Expr::Const(n)) if name == var_name => Some(value.powf(*n)),
                _ => None,
            },

            // Addition: combine polynomial terms
            Expr::Add(left, right) => {
                if let (Some(left_val), Some(right_val)) = (
                    left.evaluate_polynomial(var_name, value),
                    right.evaluate_polynomial(var_name, value),
                ) {
                    Some(left_val + right_val)
                } else {
                    None
                }
            }

            // Subtraction: combine polynomial terms
            Expr::Sub(left, right) => {
                if let (Some(left_val), Some(right_val)) = (
                    left.evaluate_polynomial(var_name, value),
                    right.evaluate_polynomial(var_name, value),
                ) {
                    Some(left_val - right_val)
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Specialized evaluation for linear expressions: ax + b
    /// Returns (a, b) coefficients if this is a linear expression, None otherwise
    #[must_use]
    pub fn extract_linear_coefficients(&self, var_name: &str) -> Option<(f64, f64)> {
        match self {
            // Constant: 0x + b
            Expr::Const(b) => Some((0.0, *b)),

            // Variable: 1x + 0
            Expr::Var(name) if name == var_name => Some((1.0, 0.0)),

            // ax (coefficient * variable)
            Expr::Mul(coeff, var) => match (coeff.as_ref(), var.as_ref()) {
                (Expr::Const(a), Expr::Var(name)) if name == var_name => Some((*a, 0.0)),
                (Expr::Var(name), Expr::Const(a)) if name == var_name => Some((*a, 0.0)),
                _ => None,
            },

            // ax + b
            Expr::Add(left, right) => {
                if let (Some((a1, b1)), Some((a2, b2))) = (
                    left.extract_linear_coefficients(var_name),
                    right.extract_linear_coefficients(var_name),
                ) {
                    Some((a1 + a2, b1 + b2))
                } else {
                    None
                }
            }

            // ax - b
            Expr::Sub(left, right) => {
                if let (Some((a1, b1)), Some((a2, b2))) = (
                    left.extract_linear_coefficients(var_name),
                    right.extract_linear_coefficients(var_name),
                ) {
                    Some((a1 - a2, b1 - b2))
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Ultra-fast evaluation for linear expressions
    #[must_use]
    pub fn evaluate_linear(&self, var_name: &str, value: f64) -> Option<f64> {
        if let Some((a, b)) = self.extract_linear_coefficients(var_name) {
            Some(a * value + b)
        } else {
            None
        }
    }

    /// Smart evaluation that chooses the fastest method based on expression structure
    pub fn evaluate_smart(&self, var_name: &str, value: f64) -> Result<f64, EvalError> {
        // Try linear evaluation first (fastest)
        if let Some(result) = self.evaluate_linear(var_name, value) {
            return Ok(result);
        }

        // Try polynomial evaluation (fast for polynomial patterns)
        if let Some(result) = self.evaluate_polynomial(var_name, value) {
            return Ok(result);
        }

        // Fall back to single-variable evaluation (faster than HashMap)
        self.evaluate_single_var(var_name, value)
    }

    /// Simplify the expression using algebraic rules
    #[must_use]
    pub fn simplify(self) -> Self {
        match self {
            // Constant folding
            Expr::Add(left, right) => match (left.simplify(), right.simplify()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                (Expr::Const(0.0), expr) | (expr, Expr::Const(0.0)) => expr,
                // x + x = 2x
                (ref left, ref right) if left == right => {
                    Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(left.clone()))
                }
                (left, right) => Expr::Add(Box::new(left), Box::new(right)),
            },
            Expr::Sub(left, right) => match (left.simplify(), right.simplify()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                (expr, Expr::Const(0.0)) => expr,
                (Expr::Const(0.0), expr) => Expr::Neg(Box::new(expr)),
                // x - x = 0
                (ref left, ref right) if left == right => Expr::Const(0.0),
                (left, right) => Expr::Sub(Box::new(left), Box::new(right)),
            },
            Expr::Mul(left, right) => match (left.simplify(), right.simplify()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                (Expr::Const(1.0), expr) | (expr, Expr::Const(1.0)) => expr,
                (Expr::Const(-1.0), expr) | (expr, Expr::Const(-1.0)) => Expr::Neg(Box::new(expr)),
                // x * x = x^2
                (ref left, ref right) if left == right => {
                    Expr::Pow(Box::new(left.clone()), Box::new(Expr::Const(2.0)))
                }
                (left, right) => Expr::Mul(Box::new(left), Box::new(right)),
            },
            Expr::Div(left, right) => match (left.simplify(), right.simplify()) {
                (Expr::Const(a), Expr::Const(b)) if b != 0.0 => Expr::Const(a / b),
                (expr, Expr::Const(1.0)) => expr,
                (Expr::Const(0.0), _) => Expr::Const(0.0),
                // x / x = 1
                (ref left, ref right) if left == right => Expr::Const(1.0),
                (left, right) => Expr::Div(Box::new(left), Box::new(right)),
            },
            Expr::Pow(base, exponent) => match (base.simplify(), exponent.simplify()) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a.powf(b)),
                (_, Expr::Const(0.0)) => Expr::Const(1.0),
                (expr, Expr::Const(1.0)) => expr,
                (Expr::Const(1.0), _) => Expr::Const(1.0),
                // (x^a)^b = x^(a*b)
                (Expr::Pow(inner_base, inner_exp), outer_exp) => {
                    let combined_exp = Expr::Mul(inner_exp, Box::new(outer_exp));
                    Expr::Pow(inner_base, Box::new(combined_exp.simplify()))
                }
                (base, exponent) => Expr::Pow(Box::new(base), Box::new(exponent)),
            },
            Expr::Ln(expr) => match expr.simplify() {
                Expr::Const(a) if a > 0.0 => Expr::Const(a.ln()),
                Expr::Const(1.0) => Expr::Const(0.0),
                Expr::Exp(inner) => *inner,
                // ln(x^a) = a * ln(x)
                Expr::Pow(base, exp) => Expr::Mul(exp, Box::new(Expr::Ln(base))),
                expr => Expr::Ln(Box::new(expr)),
            },
            Expr::Exp(expr) => match expr.simplify() {
                Expr::Const(0.0) => Expr::Const(1.0),
                Expr::Const(a) => Expr::Const(a.exp()),
                Expr::Ln(inner) => *inner,
                // exp(a * ln(x)) = x^a
                Expr::Mul(coeff, ln_expr) => match ln_expr.as_ref() {
                    Expr::Ln(base) => Expr::Pow(base.clone(), coeff),
                    _ => Expr::Exp(Box::new(Expr::Mul(coeff, ln_expr))),
                },
                expr => Expr::Exp(Box::new(expr)),
            },
            Expr::Sqrt(expr) => match expr.simplify() {
                Expr::Const(a) if a >= 0.0 => Expr::Const(a.sqrt()),
                Expr::Const(0.0) => Expr::Const(0.0),
                Expr::Const(1.0) => Expr::Const(1.0),
                // sqrt(x^2) = |x| (simplified to x for positive domains)
                Expr::Pow(base, exp) => match exp.as_ref() {
                    Expr::Const(2.0) => *base,
                    _ => Expr::Sqrt(Box::new(Expr::Pow(base, exp))),
                },
                expr => Expr::Sqrt(Box::new(expr)),
            },
            Expr::Sin(expr) => match expr.simplify() {
                Expr::Const(0.0) => Expr::Const(0.0),
                Expr::Const(a) => Expr::Const(a.sin()),
                expr => Expr::Sin(Box::new(expr)),
            },
            Expr::Cos(expr) => match expr.simplify() {
                Expr::Const(0.0) => Expr::Const(1.0),
                Expr::Const(a) => Expr::Const(a.cos()),
                expr => Expr::Cos(Box::new(expr)),
            },
            Expr::Neg(expr) => match expr.simplify() {
                Expr::Const(a) => Expr::Const(-a),
                Expr::Neg(inner) => *inner,
                expr => Expr::Neg(Box::new(expr)),
            },
            other => other,
        }
    }

    /// Simplify the expression using algebraic rules with caching
    #[must_use]
    pub fn simplify_cached(self) -> Self {
        let expr_str = format!("{self}");

        // Check cache first
        if let Ok(cache) = SIMPLIFICATION_CACHE.lock()
            && let Some(cached_result) = cache.get(&expr_str)
        {
            if let Ok(mut stats) = CACHE_STATS.lock() {
                stats.simplification_hits += 1;
            }
            return cached_result.clone();
        }

        // Cache miss - compute simplification
        if let Ok(mut stats) = CACHE_STATS.lock() {
            stats.simplification_misses += 1;
        }

        let simplified = self.simplify();

        // Store in cache
        if let Ok(mut cache) = SIMPLIFICATION_CACHE.lock() {
            cache.insert(expr_str, simplified.clone());
        }

        simplified
    }

    /// Evaluate the expression with caching for repeated evaluations
    pub fn evaluate_cached(&self, vars: &HashMap<String, f64>) -> Result<f64, EvalError> {
        let expr_str = format!("{self}");
        let vars_str = format!("{vars:?}");
        let cache_key = (expr_str.clone(), vars_str);

        // Check cache first
        if let Ok(cache) = EVALUATION_CACHE.lock()
            && let Some(&cached_result) = cache.get(&cache_key)
        {
            if let Ok(mut stats) = CACHE_STATS.lock() {
                stats.evaluation_hits += 1;
            }
            return Ok(cached_result);
        }

        // Cache miss - compute evaluation
        if let Ok(mut stats) = CACHE_STATS.lock() {
            stats.evaluation_misses += 1;
        }

        let result = self.evaluate(vars)?;

        // Store in cache
        if let Ok(mut cache) = EVALUATION_CACHE.lock() {
            cache.insert(cache_key, result);
        }

        Ok(result)
    }

    /// Extract all variables used in this expression
    #[must_use]
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Expr::Var(name) => vars.push(name.clone()),
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Expr::Ln(expr)
            | Expr::Exp(expr)
            | Expr::Sqrt(expr)
            | Expr::Sin(expr)
            | Expr::Cos(expr)
            | Expr::Neg(expr) => expr.collect_variables(vars),
            Expr::Const(_) => {}
        }
    }

    /// Count the number of operations in this expression
    #[must_use]
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) => 0,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => 1 + left.complexity() + right.complexity(),
            Expr::Ln(expr)
            | Expr::Exp(expr)
            | Expr::Sqrt(expr)
            | Expr::Sin(expr)
            | Expr::Cos(expr)
            | Expr::Neg(expr) => 1 + expr.complexity(),
        }
    }

    /// Evaluate the expression for multiple data points (vectorized)
    /// This is more efficient than calling evaluate multiple times
    pub fn evaluate_batch(&self, var_name: &str, values: &[f64]) -> Result<Vec<f64>, EvalError> {
        let mut results = Vec::with_capacity(values.len());
        let mut vars = HashMap::new();

        for &value in values {
            vars.insert(var_name.to_string(), value);
            results.push(self.evaluate(&vars)?);
        }

        Ok(results)
    }

    /// Evaluate the expression for multiple variable sets (batch evaluation)
    pub fn evaluate_batch_vars(
        &self,
        var_sets: &[HashMap<String, f64>],
    ) -> Result<Vec<f64>, EvalError> {
        let mut results = Vec::with_capacity(var_sets.len());

        for vars in var_sets {
            results.push(self.evaluate(vars)?);
        }

        Ok(results)
    }

    /// Evaluate the expression for a grid of two variables
    pub fn evaluate_grid(
        &self,
        x_name: &str,
        x_values: &[f64],
        y_name: &str,
        y_values: &[f64],
    ) -> Result<Vec<Vec<f64>>, EvalError> {
        let mut results = Vec::with_capacity(y_values.len());
        let mut vars = HashMap::new();

        for &y in y_values {
            let mut row = Vec::with_capacity(x_values.len());
            vars.insert(y_name.to_string(), y);

            for &x in x_values {
                vars.insert(x_name.to_string(), x);
                row.push(self.evaluate(&vars)?);
            }

            results.push(row);
        }

        Ok(results)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(value) => write!(f, "{value}"),
            Expr::Var(name) => write!(f, "{name}"),
            Expr::Add(left, right) => write!(f, "({left} + {right})"),
            Expr::Sub(left, right) => write!(f, "({left} - {right})"),
            Expr::Mul(left, right) => write!(f, "({left} * {right})"),
            Expr::Div(left, right) => write!(f, "({left} / {right})"),
            Expr::Pow(base, exponent) => write!(f, "({base}^{exponent})"),
            Expr::Ln(expr) => write!(f, "ln({expr})"),
            Expr::Exp(expr) => write!(f, "exp({expr})"),
            Expr::Sqrt(expr) => write!(f, "sqrt({expr})"),
            Expr::Sin(expr) => write!(f, "sin({expr})"),
            Expr::Cos(expr) => write!(f, "cos({expr})"),
            Expr::Neg(expr) => write!(f, "(-{expr})"),
        }
    }
}

impl Expr {
    /// Format as LaTeX for mathematical documents
    #[must_use]
    pub fn to_latex(&self) -> String {
        match self {
            Expr::Const(value) => {
                if value.fract() == 0.0 && value.abs() < 1e10 {
                    format!("{}", *value as i64)
                } else {
                    format!("{value}")
                }
            }
            Expr::Var(name) => name.clone(),
            Expr::Add(left, right) => format!("{} + {}", left.to_latex(), right.to_latex()),
            Expr::Sub(left, right) => format!("{} - {}", left.to_latex(), right.to_latex()),
            Expr::Mul(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Const(c), _) if *c == 1.0 => right.to_latex(),
                (Expr::Const(c), _) if *c == -1.0 => format!("-{}", right.to_latex()),
                (Expr::Const(_), Expr::Var(_)) => {
                    format!("{}{}", left.to_latex(), right.to_latex())
                }
                _ => format!("{} \\cdot {}", left.to_latex(), right.to_latex()),
            },
            Expr::Div(left, right) => {
                format!("\\frac{{{}}}{{{}}}", left.to_latex(), right.to_latex())
            }
            Expr::Pow(base, exponent) => format!("{}^{{{}}}", base.to_latex(), exponent.to_latex()),
            Expr::Ln(expr) => format!("\\ln({})", expr.to_latex()),
            Expr::Exp(expr) => format!("e^{{{}}}", expr.to_latex()),
            Expr::Sqrt(expr) => format!("\\sqrt{{{}}}", expr.to_latex()),
            Expr::Sin(expr) => format!("\\sin({})", expr.to_latex()),
            Expr::Cos(expr) => format!("\\cos({})", expr.to_latex()),
            Expr::Neg(expr) => format!("-({})", expr.to_latex()),
        }
    }

    /// Format as Python code for numerical evaluation
    /// TODO: `to_julia`
    #[must_use]
    pub fn to_python(&self) -> String {
        match self {
            Expr::Const(value) => format!("{value}"),
            Expr::Var(name) => name.clone(),
            Expr::Add(left, right) => format!("({} + {})", left.to_python(), right.to_python()),
            Expr::Sub(left, right) => format!("({} - {})", left.to_python(), right.to_python()),
            Expr::Mul(left, right) => format!("({} * {})", left.to_python(), right.to_python()),
            Expr::Div(left, right) => format!("({} / {})", left.to_python(), right.to_python()),
            Expr::Pow(base, exponent) => {
                format!("({} ** {})", base.to_python(), exponent.to_python())
            }
            Expr::Ln(expr) => format!("math.log({})", expr.to_python()),
            Expr::Exp(expr) => format!("math.exp({})", expr.to_python()),
            Expr::Sqrt(expr) => format!("math.sqrt({})", expr.to_python()),
            Expr::Sin(expr) => format!("math.sin({})", expr.to_python()),
            Expr::Cos(expr) => format!("math.cos({})", expr.to_python()),
            Expr::Neg(expr) => format!("-({})", expr.to_python()),
        }
    }
}

// Implement standard operator traits to avoid confusion with method names
impl Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Expr::Neg(Box::new(self))
    }
}

// Enhanced conversion traits for ergonomic usage
impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Expr::Const(value)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        Expr::Const(f64::from(value))
    }
}

impl From<&str> for Expr {
    fn from(name: &str) -> Self {
        Expr::Var(name.to_string())
    }
}

impl From<String> for Expr {
    fn from(name: String) -> Self {
        Expr::Var(name)
    }
}

// Allow mixing Expr with numeric types
impl Add<f64> for Expr {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Expr::from(rhs)
    }
}

impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        Expr::from(self) + rhs
    }
}

impl Sub<f64> for Expr {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Expr::from(rhs)
    }
}

impl Sub<Expr> for f64 {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Self::Output {
        Expr::from(self) - rhs
    }
}

impl Mul<f64> for Expr {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Expr::from(rhs)
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::from(self) * rhs
    }
}

impl Div<f64> for Expr {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self / Expr::from(rhs)
    }
}

impl Div<Expr> for f64 {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Self::Output {
        Expr::from(self) / rhs
    }
}

// Enhanced mathematical functions as methods
impl Expr {
    /// Raise this expression to a power
    #[must_use]
    pub fn powf(self, exponent: impl Into<Expr>) -> Self {
        Expr::Pow(Box::new(self), Box::new(exponent.into()))
    }

    /// Take the natural logarithm of this expression (instance method)
    #[must_use]
    pub fn natural_log(self) -> Self {
        Expr::Ln(Box::new(self))
    }

    /// Take the exponential of this expression (instance method)
    #[must_use]
    pub fn exponential(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    /// Take the square root of this expression (instance method)
    #[must_use]
    pub fn square_root(self) -> Self {
        Expr::Sqrt(Box::new(self))
    }

    /// Take the sine of this expression
    #[must_use]
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Take the cosine of this expression
    #[must_use]
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Square this expression (x^2)
    #[must_use]
    pub fn square(self) -> Self {
        self.clone() * self
    }

    /// Cube this expression (x^3)
    #[must_use]
    pub fn cube(self) -> Self {
        let x = self.clone();
        x.clone() * x.clone() * x
    }

    /// Absolute value using sqrt(x^2)
    #[must_use]
    pub fn abs(self) -> Self {
        self.square().square_root()
    }
}

/// Errors that can occur during expression evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// Variable not found in the environment
    UndefinedVariable(String),
    /// Division by zero
    DivisionByZero,
    /// Invalid logarithm (non-positive argument)
    InvalidLogarithm(f64),
    /// Invalid square root (negative argument)
    InvalidSquareRoot(f64),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UndefinedVariable(name) => write!(f, "Undefined variable: {name}"),
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::InvalidLogarithm(value) => write!(f, "Invalid logarithm: ln({value})"),
            EvalError::InvalidSquareRoot(value) => write!(f, "Invalid square root: sqrt({value})"),
        }
    }
}

impl std::error::Error for EvalError {}

/// A pool of precomputed constants with dependency tracking
#[derive(Debug, Clone)]
pub struct ConstantPool {
    /// Constants with their computed values
    pub constants: HashMap<String, f64>,
    /// Dependency graph: each constant depends on these others
    pub dependencies: HashMap<String, Vec<String>>,
    /// Evaluation order (topologically sorted)
    pub evaluation_order: Vec<String>,
    /// Original parameter expressions that generated these constants
    pub expressions: HashMap<String, String>,
}

impl ConstantPool {
    /// Create an empty constant pool
    #[must_use]
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
            dependencies: HashMap::new(),
            evaluation_order: Vec::new(),
            expressions: HashMap::new(),
        }
    }

    /// Add a constant with its dependencies
    pub fn add_constant(
        &mut self,
        name: String,
        value: f64,
        expression: String,
        deps: Vec<String>,
    ) {
        self.constants.insert(name.clone(), value);
        self.expressions.insert(name.clone(), expression);
        self.dependencies.insert(name.clone(), deps);
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

/// A symbolic log-density function using our custom IR
#[derive(Debug, Clone)]
pub struct SymbolicLogDensity {
    /// The expression tree
    pub expression: Expr,
    /// Parameter values that can be substituted
    pub parameters: HashMap<String, f64>,
    /// Variables that remain symbolic (e.g., "x")
    pub variables: Vec<String>,
}

impl SymbolicLogDensity {
    /// Create a new symbolic log-density
    #[must_use]
    pub fn new(expression: Expr, parameters: HashMap<String, f64>) -> Self {
        let variables = expression.variables();
        Self {
            expression,
            parameters,
            variables,
        }
    }

    /// Evaluate the expression at a given point
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Result<f64, EvalError> {
        let mut env = self.parameters.clone();
        env.extend(vars.iter().map(|(k, v)| (k.clone(), *v)));
        self.expression.evaluate(&env)
    }

    /// Evaluate for a single variable (common case)
    pub fn evaluate_single(&self, var_name: &str, value: f64) -> Result<f64, EvalError> {
        let mut env = self.parameters.clone();
        env.insert(var_name.to_string(), value);
        self.expression.evaluate(&env)
    }

    /// Simplify the expression
    #[must_use]
    pub fn simplify(mut self) -> Self {
        self.expression = self.expression.simplify();
        self
    }

    /// Substitute parameter values into the expression
    #[must_use]
    pub fn substitute_parameters(mut self) -> Self {
        self.expression = self.substitute_expr(self.expression.clone());
        self
    }

    fn substitute_expr(&self, expr: Expr) -> Expr {
        match expr {
            Expr::Var(name) => {
                if let Some(&value) = self.parameters.get(&name) {
                    Expr::Const(value)
                } else {
                    Expr::Var(name)
                }
            }
            Expr::Add(left, right) => Expr::Add(
                Box::new(self.substitute_expr(*left)),
                Box::new(self.substitute_expr(*right)),
            ),
            Expr::Sub(left, right) => Expr::Sub(
                Box::new(self.substitute_expr(*left)),
                Box::new(self.substitute_expr(*right)),
            ),
            Expr::Mul(left, right) => Expr::Mul(
                Box::new(self.substitute_expr(*left)),
                Box::new(self.substitute_expr(*right)),
            ),
            Expr::Div(left, right) => Expr::Div(
                Box::new(self.substitute_expr(*left)),
                Box::new(self.substitute_expr(*right)),
            ),
            Expr::Pow(base, exponent) => Expr::Pow(
                Box::new(self.substitute_expr(*base)),
                Box::new(self.substitute_expr(*exponent)),
            ),
            Expr::Ln(expr) => Expr::Ln(Box::new(self.substitute_expr(*expr))),
            Expr::Exp(expr) => Expr::Exp(Box::new(self.substitute_expr(*expr))),
            Expr::Sqrt(expr) => Expr::Sqrt(Box::new(self.substitute_expr(*expr))),
            Expr::Sin(expr) => Expr::Sin(Box::new(self.substitute_expr(*expr))),
            Expr::Cos(expr) => Expr::Cos(Box::new(self.substitute_expr(*expr))),
            Expr::Neg(expr) => Expr::Neg(Box::new(self.substitute_expr(*expr))),
            other => other,
        }
    }
}

/// Helper functions for building common expressions
pub mod builders {
    use super::Expr;

    /// Build a quadratic expression: -0.5 * (x - mu)^2 / sigma^2
    #[must_use]
    pub fn quadratic_term(x: &str, mu: f64, sigma: f64) -> Expr {
        let x_var = Expr::variable(x);
        let mu_const = Expr::constant(mu);
        let sigma_squared = Expr::constant(sigma * sigma);
        let half = Expr::constant(0.5);

        let diff = Expr::sub(x_var, mu_const);
        let diff_squared = Expr::pow(diff, Expr::constant(2.0));
        let normalized = Expr::div(diff_squared, sigma_squared);
        Expr::neg(Expr::mul(half, normalized))
    }

    /// Build a linear term: coeff * x
    #[must_use]
    pub fn linear_term(x: &str, coeff: f64) -> Expr {
        Expr::mul(Expr::constant(coeff), Expr::variable(x))
    }

    /// Build a log normalization constant: -0.5 * ln(2π) - ln(sigma)
    #[must_use]
    pub fn log_normal_constant(sigma: f64) -> Expr {
        let half = Expr::constant(0.5);
        let two_pi = Expr::constant(2.0 * std::f64::consts::PI);
        let sigma_const = Expr::constant(sigma);

        let log_two_pi = Expr::ln(two_pi);
        let log_sigma = Expr::ln(sigma_const);

        Expr::sub(Expr::neg(Expr::mul(half, log_two_pi)), log_sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_evaluation() {
        let expr = Expr::add(
            Expr::mul(Expr::constant(2.0), Expr::variable("x")),
            Expr::constant(3.0),
        );

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);

        assert_eq!(expr.evaluate(&vars).unwrap(), 13.0);
    }

    #[test]
    fn test_expression_simplification() {
        let expr = Expr::add(Expr::constant(2.0), Expr::constant(3.0));
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(5.0));
    }

    #[test]
    fn test_quadratic_term() {
        let expr = builders::quadratic_term("x", 0.0, 1.0);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);

        // Should be -0.5 * (1 - 0)^2 / 1^2 = -0.5
        assert_eq!(expr.evaluate(&vars).unwrap(), -0.5);
    }

    #[test]
    fn test_symbolic_log_density() {
        let expr = builders::quadratic_term("x", 2.0, 1.5);
        let mut params = HashMap::new();
        params.insert("mu".to_string(), 2.0);
        params.insert("sigma".to_string(), 1.5);

        let symbolic = SymbolicLogDensity::new(expr, params);
        let result = symbolic.evaluate_single("x", 2.0).unwrap();

        // At x = mu, the quadratic term should be 0
        assert!((result - 0.0).abs() < 1e-10);
    }
}
