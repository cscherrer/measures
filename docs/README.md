# Measures Documentation

This directory contains detailed documentation for the measures crate.

## Contents

- [Capabilities Summary](capabilities_summary.md) - High-level overview of all framework capabilities
- [General Density Computation](general_density_computation.md) - Computing densities with respect to any base measure
- [Performance Optimization](performance_optimization.md) - JIT compilation and zero-overhead optimization techniques

## Getting Started

The `measures` crate provides a type-safe framework for working with measure theory
and probability distributions. At its core, it separates measures from densities
to enable flexible computation of probabilities and likelihood functions.

### Basic Example

```rust
use measures::{Normal, LogDensityBuilder};

// Create a standard normal distribution
let normal = Normal::new(0.0, 1.0);

// Compute log-density at x = 0.5
let log_density: f64 = normal.log_density().at(&0.5);

// Compute log-density with respect to another measure
let other_normal = Normal::new(1.0, 2.0);
let relative_log_density: f64 = normal.log_density().wrt(other_normal).at(&0.5);
```

### General Density Computation

The framework supports computing densities with respect to any base measure, not just the root measure:

```rust
use measures::{Normal, LogDensityBuilder};

let normal1 = Normal::new(0.0, 1.0);
let normal2 = Normal::new(1.0, 2.0);

// Compute density of normal1 with respect to normal2
let relative_density = normal1.log_density().wrt(normal2).at(&0.5);
```

### IID Collections

```rust
use measures::{IIDExtension, Normal};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

let samples = vec![0.5, -0.3, 1.2];
let iid_log_density: f64 = iid_normal.iid_log_density(&samples);
```

## Key Features

### 🎯 General Density Computation
- Compute densities with respect to any base measure
- Automatic optimization for measures with shared roots
- Applications in importance sampling, model comparison, and variational inference

### 🚀 Performance Optimization  
- Zero-overhead runtime code generation
- Compile-time macro optimization
- JIT compilation with Cranelift
- Comprehensive performance analysis and best practices

### 🔧 Type Safety
- Compile-time verification of measure compatibility
- Generic numeric types (f64, f32, dual numbers for autodiff)
- Zero-cost abstractions with static dispatch

### 📊 Exponential Families
- Unified interface for exponential family distributions
- Automatic IID handling with efficient batch computation
- Natural parameter and sufficient statistic access

For more examples, see the `examples/` directory in the repository. 