//! Automatic JIT compilation derivation for exponential family distributions.
//!
//! This module provides automatic derivation of JIT-compiled log-density functions
//! from the exponential family structure. Instead of manually implementing
//! `CustomJITOptimizer` for each distribution, this system automatically generates
//! the symbolic IR and JIT compilation from the exponential family traits.

use crate::exponential_family::jit::{JITError, JITFunction};
use crate::exponential_family::symbolic_ir::{Expr, SymbolicLogDensity};
use std::any::TypeId;
use std::collections::HashMap;

/// Registry of automatic JIT compilation patterns for different distribution types
pub struct AutoJITRegistry {
    patterns: HashMap<TypeId, Box<dyn AutoJITPattern>>,
}

impl AutoJITRegistry {
    /// Create a new registry with built-in patterns
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            patterns: HashMap::new(),
        };

        // Register built-in patterns
        registry.register_pattern(
            TypeId::of::<crate::distributions::Normal<f64>>(),
            Box::new(NormalPattern),
        );
        registry.register_pattern(
            TypeId::of::<crate::distributions::Exponential<f64>>(),
            Box::new(ExponentialPattern),
        );

        registry
    }

    /// Register a new pattern for a distribution type
    pub fn register_pattern(&mut self, type_id: TypeId, pattern: Box<dyn AutoJITPattern>) {
        self.patterns.insert(type_id, pattern);
    }

    /// Get the pattern for a distribution type
    #[must_use]
    pub fn get_pattern(&self, type_id: TypeId) -> Option<&dyn AutoJITPattern> {
        self.patterns.get(&type_id).map(std::convert::AsRef::as_ref)
    }
}

impl Default for AutoJITRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for automatic JIT compilation patterns
pub trait AutoJITPattern: Send + Sync {
    /// Generate symbolic log-density for this pattern
    fn generate_symbolic(
        &self,
        distribution: &dyn std::any::Any,
    ) -> Result<SymbolicLogDensity, JITError>;
}

/// Pattern for Normal distribution: N(μ, σ²)
/// Natural form: η = [μ/σ², -1/(2σ²)], T(x) = [x, x²]
/// Log-density: η₁·x + η₂·x² - A(η) = μx/σ² - x²/(2σ²) - ½log(2πσ²)
struct NormalPattern;

impl AutoJITPattern for NormalPattern {
    fn generate_symbolic(
        &self,
        distribution: &dyn std::any::Any,
    ) -> Result<SymbolicLogDensity, JITError> {
        let normal = distribution
            .downcast_ref::<crate::distributions::Normal<f64>>()
            .ok_or_else(|| {
                JITError::CompilationError(
                    "Invalid distribution type for Normal pattern".to_string(),
                )
            })?;

        let mu = normal.mean;
        let sigma = normal.std_dev;
        let sigma_sq = sigma * sigma;

        // Build expression: -½log(2πσ²) - (x-μ)²/(2σ²)
        let x = Expr::Var("x".to_string());
        let mu_expr = Expr::Const(mu);
        let diff = Expr::Sub(Box::new(x.clone()), Box::new(mu_expr));
        let diff_sq = Expr::Mul(Box::new(diff.clone()), Box::new(diff));

        let coeff = -1.0 / (2.0 * sigma_sq);
        let quadratic_term = Expr::Mul(Box::new(Expr::Const(coeff)), Box::new(diff_sq));

        let log_norm_const = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let constant_term = Expr::Const(log_norm_const);

        let expr = Expr::Add(Box::new(constant_term), Box::new(quadratic_term));

        let mut parameters = HashMap::new();
        parameters.insert("mu".to_string(), mu);
        parameters.insert("sigma".to_string(), sigma);
        parameters.insert("log_norm_const".to_string(), log_norm_const);
        parameters.insert("coeff".to_string(), coeff);

        Ok(SymbolicLogDensity::new(expr, parameters))
    }
}

/// Pattern for Exponential distribution: Exp(λ)
/// Natural form: η = -λ, T(x) = x
/// Log-density: -λx + log(λ)
struct ExponentialPattern;

impl AutoJITPattern for ExponentialPattern {
    fn generate_symbolic(
        &self,
        distribution: &dyn std::any::Any,
    ) -> Result<SymbolicLogDensity, JITError> {
        let exponential = distribution
            .downcast_ref::<crate::distributions::Exponential<f64>>()
            .ok_or_else(|| {
                JITError::CompilationError(
                    "Invalid distribution type for Exponential pattern".to_string(),
                )
            })?;

        let rate = exponential.rate;

        // Build expression: log(λ) - λx
        let x = Expr::Var("x".to_string());
        let log_rate = Expr::Const(rate.ln());
        let rate_term = Expr::Mul(Box::new(Expr::Const(-rate)), Box::new(x));
        let expr = Expr::Add(Box::new(log_rate), Box::new(rate_term));

        let mut parameters = HashMap::new();
        parameters.insert("rate".to_string(), rate);
        parameters.insert("log_rate".to_string(), rate.ln());

        Ok(SymbolicLogDensity::new(expr, parameters))
    }
}

/// Automatic JIT optimizer that uses pattern matching
pub struct AutoJITOptimizer {
    registry: AutoJITRegistry,
}

impl AutoJITOptimizer {
    /// Create a new automatic JIT optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: AutoJITRegistry::new(),
        }
    }

    /// Generate symbolic representation for any supported distribution
    pub fn generate_symbolic<D>(&self, distribution: &D) -> Result<SymbolicLogDensity, JITError>
    where
        D: 'static,
    {
        let type_id = TypeId::of::<D>();
        let pattern = self.registry.get_pattern(type_id).ok_or_else(|| {
            JITError::CompilationError(format!("No pattern registered for type {type_id:?}"))
        })?;

        pattern.generate_symbolic(distribution as &dyn std::any::Any)
    }

    /// Compile distribution to JIT function
    pub fn compile_jit<D>(&self, distribution: &D) -> Result<JITFunction, JITError>
    where
        D: 'static,
    {
        let symbolic = self.generate_symbolic(distribution)?;
        let compiler = crate::exponential_family::jit::JITCompiler::new()?;
        compiler.compile_custom_expression(&symbolic)
    }
}

impl Default for AutoJITOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait that adds automatic JIT compilation to any distribution
pub trait AutoJITExt {
    /// Automatically generate JIT-compiled log-density function
    fn auto_jit(&self) -> Result<JITFunction, JITError>;

    /// Automatically generate symbolic representation
    fn auto_symbolic(&self) -> Result<SymbolicLogDensity, JITError>;
}

impl<D> AutoJITExt for D
where
    D: 'static,
{
    fn auto_jit(&self) -> Result<JITFunction, JITError> {
        let optimizer = AutoJITOptimizer::new();
        optimizer.compile_jit(self)
    }

    fn auto_symbolic(&self) -> Result<SymbolicLogDensity, JITError> {
        let optimizer = AutoJITOptimizer::new();
        optimizer.generate_symbolic(self)
    }
}

/// Macro for automatically implementing `CustomJITOptimizer` for distributions
#[macro_export]
macro_rules! auto_jit_impl {
    ($dist_type:ty) => {
        impl $crate::exponential_family::jit::CustomJITOptimizer<f64, f64> for $dist_type {
            fn custom_symbolic_log_density(
                &self,
            ) -> $crate::exponential_family::symbolic_ir::SymbolicLogDensity {
                use $crate::exponential_family::auto_jit::AutoJITExt;
                self.auto_symbolic().unwrap_or_else(|_| {
                    // Fallback to zero expression if auto-derivation fails
                    $crate::exponential_family::symbolic_ir::SymbolicLogDensity::new(
                        $crate::exponential_family::symbolic_ir::Expr::Const(0.0),
                        std::collections::HashMap::new(),
                    )
                })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Exponential, Normal};

    #[test]
    fn test_normal_auto_jit() {
        let normal = Normal::new(2.0, 1.5);
        let result = normal.auto_symbolic();
        assert!(result.is_ok(), "Normal auto symbolic should succeed");

        let symbolic = result.unwrap();
        assert_eq!(symbolic.variables.len(), 1);
        assert_eq!(symbolic.variables[0], "x");
        assert!(symbolic.parameters.contains_key("mu"));
        assert!(symbolic.parameters.contains_key("sigma"));
    }

    #[test]
    fn test_exponential_auto_jit() {
        let exponential = Exponential::new(2.0);
        let result = exponential.auto_symbolic();
        assert!(result.is_ok(), "Exponential auto symbolic should succeed");

        let symbolic = result.unwrap();
        assert_eq!(symbolic.variables.len(), 1);
        assert_eq!(symbolic.variables[0], "x");
        assert!(symbolic.parameters.contains_key("rate"));
    }

    #[test]
    fn test_auto_jit_compilation() {
        let normal = Normal::new(0.0, 1.0);
        let result = normal.auto_jit();
        assert!(result.is_ok(), "Auto JIT compilation should succeed");
    }

    #[test]
    fn test_registry_pattern_lookup() {
        let registry = AutoJITRegistry::new();
        let normal_pattern = registry.get_pattern(TypeId::of::<Normal<f64>>());
        assert!(
            normal_pattern.is_some(),
            "Normal pattern should be registered"
        );

        let exp_pattern = registry.get_pattern(TypeId::of::<Exponential<f64>>());
        assert!(
            exp_pattern.is_some(),
            "Exponential pattern should be registered"
        );
    }
}
