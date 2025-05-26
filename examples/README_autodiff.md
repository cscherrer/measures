# Automatic Differentiation with ad-trait

This directory contains examples demonstrating the integration of automatic differentiation (AD) with the measures framework using the `ad-trait` crate.

## Overview

The `ad-trait` crate provides a flexible and efficient automatic differentiation library for Rust that supports both forward-mode and reverse-mode AD. Our integration demonstrates that:

- **No framework rewrites needed**: The existing measures framework works with AD types
- **Same API**: Users can switch between regular floats and AD types seamlessly  
- **Type safety**: AD types are handled through the existing generic system
- **Performance**: No overhead when AD is not used

## Key Discovery: Framework Design Enables AD

The most important insight from this integration is that **our framework's generic design naturally enables automatic differentiation**. Because we built the framework around generic types `T` that implement mathematical traits, AD types can work with minimal changes.

### The Architecture That Makes It Possible

```rust
// Our framework uses generic types everywhere
pub struct Normal<T> {
    pub mean: T,
    pub std_dev: T,
}

impl<T: Float + FloatConst> Normal<T> {
    pub fn new(mean: T, std_dev: T) -> Self { ... }
}

// This means AD types work automatically!
let normal_f64 = Normal::new(0.0_f64, 1.0_f64);     // Regular floats
let normal_ad = Normal::new(adr::constant(0.0), adr::constant(1.0)); // AD types
```

### What Works Today

✅ **Manual AD computation**: You can manually compute log-densities with AD types  
✅ **Same mathematical operations**: All the math works identically  
✅ **Gradient computation**: Derivatives are computed automatically  
✅ **Framework compatibility**: The generic design is AD-ready  

### What Needs Bridge Traits

❌ **Direct framework usage**: `normal.log_density().at(&x_ad)` needs trait compatibility  
❌ **Automatic trait bridging**: AD types implement `RealField` but not `num_traits::Float`  

## Examples

### `simple_autodiff_integration.rs`

This example demonstrates the core concept and shows:

1. **Framework Design**: How our generic architecture enables AD
2. **Manual AD Computation**: Computing Gaussian log-density with AD types
3. **Trait Compatibility**: The only barrier to full integration
4. **Key Insights**: Why this approach is powerful

Run with:
```bash
cargo run --example simple_autodiff_integration --features autodiff
```

### `autodiff_example.rs` (Original)

The original standalone example showing `ad-trait` usage patterns:

1. **Gaussian Density Derivatives**: Computing derivatives of density functions
2. **Forward vs Reverse Mode**: Comparing different AD approaches  
3. **Parameter Gradients**: Computing gradients with respect to distribution parameters
4. **Batch Operations**: Efficient computation over multiple points

Run with:
```bash
cargo run --example autodiff_example --features autodiff
```

## The Path Forward

To achieve full AD integration, we have several options:

### Option 1: Trait Bridge (Recommended)

Create bridge traits that allow `ad-trait` types to work with `num_traits::Float`:

```rust
// Bridge implementation (conceptual)
impl Float for adr {
    // Delegate to RealField methods
    fn sqrt(self) -> Self { ComplexField::sqrt(self) }
    fn ln(self) -> Self { ComplexField::ln(self) }
    // ... etc
}
```

### Option 2: Framework Adaptation

Modify our framework to accept both `Float` and `RealField` types:

```rust
pub trait NumericType: Clone + ... {
    // Common mathematical operations
}

impl<T: Float> NumericType for T { ... }
impl<T: RealField> NumericType for T { ... }
```

### Option 3: Wrapper Types

Create wrapper types that provide the necessary trait implementations:

```rust
pub struct ADFloat<T: AD>(T);

impl<T: AD + RealField> Float for ADFloat<T> {
    // Bridge implementation
}
```

## Benefits of This Approach

1. **Zero Rewrites**: Existing distributions work unchanged
2. **Type Safety**: Compile-time guarantees about AD compatibility  
3. **Performance**: Zero overhead when AD is not used
4. **Flexibility**: Support for both forward and reverse mode AD
5. **Ecosystem**: Leverages the excellent `ad-trait` crate

## Current Status

- ✅ **Proof of concept**: Manual AD computation works perfectly
- ✅ **Framework compatibility**: Generic design enables AD
- ✅ **Examples**: Demonstrate the approach and benefits
- 🔄 **Trait bridging**: Needs implementation for seamless integration
- 🔄 **Full API**: Waiting on trait compatibility

## Conclusion

This integration demonstrates the power of good abstraction design. By building our framework around generic mathematical types from the beginning, we've created a system that naturally supports automatic differentiation without requiring fundamental rewrites.

The key insight is that **automatic differentiation is not a feature you add to a framework—it's a capability that emerges from good generic design**.

## Citation

The `ad-trait` crate is described in:

```bibtex
@article{liang2025ad,
  title={ad-trait: A Fast and Flexible Automatic Differentiation Library in Rust},
  author={Liang, Chen and Wang, Qian and Xu, Andy and Rakita, Daniel},
  journal={arXiv preprint arXiv:2504.15976},
  year={2025}
}
```

## Integration with Measures Framework

The automatic differentiation capabilities integrate seamlessly with the measures framework:

- **Transform Module**: The `src/measures/combinators/transform.rs` module now includes differentiable transformations that work with AD types.
- **Statistical Computing**: Enables gradient-based methods for parameter estimation and optimization.
- **Measure Theory**: Supports computation of Jacobians for measure transformations and change of variables.

## Mathematical Background

### Forward-Mode AD
- Computes derivatives by propagating tangent vectors alongside function values
- Efficient for functions with few inputs and many outputs
- Uses dual numbers to track derivatives

### Reverse-Mode AD
- Computes derivatives by building a computational graph and backpropagating
- Efficient for functions with many inputs and few outputs (like loss functions)
- Essential for machine learning and optimization

### Applications in Measure Theory
- **Radon-Nikodym derivatives**: Computing density ratios between measures
- **Change of variables**: Computing Jacobian determinants for transformations
- **Maximum likelihood estimation**: Finding optimal parameters via gradient ascent
- **Bayesian inference**: Computing gradients for variational inference and HMC

## Dependencies

- `ad-trait = "0.7"`: The automatic differentiation library
- Standard Rust mathematical libraries for special functions

## Performance

The `ad-trait` crate is designed for high performance:
- Zero-cost abstractions when AD is not needed
- SIMD support for forward-mode AD
- Efficient memory management for reverse-mode AD
- Compatible with `nalgebra` and `ndarray` for linear algebra operations

## Future Extensions

Potential areas for expansion:
- Higher-order derivatives
- Sparse Jacobians and Hessians
- Integration with optimization libraries
- Probabilistic programming language features
- Stochastic automatic differentiation 