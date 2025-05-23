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
use measures::{Normal, LogDensityBuilder};

// Create a standard normal distribution
let normal = Normal::new(0.0, 1.0);

// Compute log-density at x = 0.5
let log_density: f64 = normal.log_density().at(&0.5);

// Compute log-density with respect to another measure
let other_normal = Normal::new(1.0, 2.0);
let relative_log_density: f64 = normal.log_density().wrt(other_normal).at(&0.5);
```

### IID Collections

```rust
use measures::{IIDExtension, Normal};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();

let samples = vec![0.5, -0.3, 1.2];
let iid_log_density: f64 = iid_normal.iid_log_density(&samples);
```

For more examples, see the `examples/` directory in the repository. 