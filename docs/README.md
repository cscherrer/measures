# Measures Documentation

This directory contains detailed documentation for the measures crate.

## Contents

- [Measure Theory Concepts](measure_theory.md) - Mathematical background on measure theory
- [Exponential Family](exponential_family.md) - Details on exponential family distributions
- [Usage Guide](usage_guide.md) - Examples and patterns for using the library

## Getting Started

The `measures` crate provides a type-safe framework for working with measure theory
and probability distributions. At its core, it separates measures from densities
to enable flexible computation of probabilities and likelihood functions.

### Basic Example

```rust
use measures::{Normal};

// Create a standard normal distribution
let normal = Normal::new(0.0, 1.0);

// Compute density at x = 0
let density: f64 = normal.density(&0.0).into();

// Compute log-density (more efficient)
let log_density: f64 = normal.log_density(&0.0).into();
``` 