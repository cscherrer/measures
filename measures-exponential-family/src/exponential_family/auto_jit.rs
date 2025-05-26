//! Automatic JIT compilation for exponential family distributions
//!
//! This module provides automatic JIT compilation that analyzes usage patterns
//! and decides when to compile distributions to native code for optimal performance.

use crate::exponential_family::jit::CustomSymbolicLogDensity;
use crate::exponential_family::jit::{JITError, JITFunction};
use num_traits::Float;
use std::any::TypeId;
use std::collections::HashMap;
use symbolic_math::{Expr, expr::SymbolicLogDensity};

/// Registry of automatic JIT compilation patterns for different distribution types
pub struct AutoJITRegistry {
    patterns: HashMap<TypeId, Box<dyn AutoJITPattern>>,
}

impl AutoJITRegistry {
    /// Create a new registry with built-in patterns
    #[must_use]
    pub fn new() -> Self {
        let registry = Self {
            patterns: HashMap::new(),
        };

        // Built-in patterns are registered by the distributions themselves
        // to avoid circular dependencies
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

        // Convert symbolic-math SymbolicLogDensity to CustomSymbolicLogDensity
        let custom_symbolic = CustomSymbolicLogDensity {
            expression: symbolic.expression,
            variables: symbolic.variables,
            parameters: symbolic.parameters,
        };

        compiler.compile_custom_expression(&custom_symbolic)
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

/// Macro to automatically implement JIT compilation for exponential family distributions
#[macro_export]
macro_rules! auto_jit_impl {
    ($dist_type:ty) => {
        // This macro is now a no-op since AutoJITExt is automatically implemented
        // for all types that are 'static. The actual JIT compilation is handled
        // by the pattern registry in AutoJITOptimizer.
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AutoJITRegistry::new();
        assert_eq!(registry.patterns.len(), 0, "Registry should start empty");
    }

    #[test]
    fn test_auto_jit_optimizer_creation() {
        let optimizer = AutoJITOptimizer::new();
        assert_eq!(
            optimizer.registry.patterns.len(),
            0,
            "Optimizer should start with empty registry"
        );
    }
}
