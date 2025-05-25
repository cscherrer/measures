# IID Exponential Family Implementation

## Mathematical Foundation

This implementation realizes the theoretical principle that **IID collections from exponential families are themselves exponential families** with a specific mathematical structure.

### Single Variable Exponential Family

A probability distribution belongs to the exponential family if it can be written as:

```
f(x|θ) = h(x) exp(η(θ)ᵀT(x) - A(η(θ)))
```

where:
- `h(x)` is the base measure
- `η(θ)` is the natural parameter  
- `T(x)` is the sufficient statistic
- `A(η)` is the log-partition (cumulant) function

### IID Collection Structure

For `X₁, X₂, ..., Xₙ` IID from this exponential family, the joint density is:

```
f(x₁,...,xₙ|θ) = ∏ᵢh(xᵢ) exp(η(θ)ᵀ∑ᵢT(xᵢ) - nA(η(θ)))
```

This maintains exponential family form with:
- **Base measure**: `∏ᵢh(xᵢ)` (product of individual base measures)
- **Sufficient statistic**: `∑ᵢT(xᵢ)` (sum of individual sufficient statistics)  
- **Natural parameter**: `η(θ)` (same as original)
- **Log-partition function**: `nA(η(θ))` (scaled by sample size)

## Implementation Overview

### Core Components

1. **IID Wrapper** (`src/exponential_family/iid.rs`)
   - `IID<D>` struct wraps any exponential family distribution
   - Implements `Measure<Vec<X>>` for vector observations
   - Provides both standard and manual API methods

2. **Extension Trait** 
   - `IIDExtension` adds `.iid()` method to all exponential families
   - Blanket implementation for any `ExponentialFamily<X, F>`

3. **Root Measure Handling**
   - `IIDRootMeasure` trait maps individual root measures to vector versions
   - `LebesgueMeasure<X>` → `LebesgueMeasure<Vec<X>>`
   - `CountingMeasure<X>` → `CountingMeasure<Vec<X>>`

### Key Features

#### Consistent API Design
The IID implementation provides **two equivalent APIs**:

```rust
// Standard API (recommended) - consistent with individual distributions
let log_density: f64 = iid_normal.log_density().at(&samples);

// Manual API (backward compatibility)
let log_density: f64 = iid_normal.iid_log_density(&samples);
```

Both APIs produce identical results and use the same efficient computation under the hood.

#### Correct Mathematical Implementation
- Log-density computation follows `log p(x₁,...,xₙ) = ∑ᵢ log p(xᵢ)`
- Handles empty samples correctly (log-density = 0)
- Proper scaling with sample size

## Usage Examples

### Basic Usage

```rust
use measures::{IIDExtension, LogDensityBuilder, Normal};

let normal = Normal::new(0.0, 1.0);
let iid_normal = normal.iid();
let samples = vec![0.0, 1.0, -1.0];

// Standard API (recommended)
let log_density = iid_normal.log_density().at(&samples);

// Manual API (backward compatibility)  
let log_density = iid_normal.iid_log_density(&samples);
```

### Statistical Inference

```rust
// Maximum likelihood estimation
let candidates = vec![
    Normal::new(0.0, 1.0),
    Normal::new(1.0, 1.5),
    Normal::new(2.0, 0.8),
];

let observed_data = vec![0.5, 1.2, 0.8, 1.1];

let best_model = candidates
    .iter()
    .max_by(|a, b| {
        let log_likelihood_a = a.clone().iid().log_density().at(&observed_data);
        let log_likelihood_b = b.clone().iid().log_density().at(&observed_data);
        log_likelihood_a.partial_cmp(&log_likelihood_b).unwrap()
    })
    .unwrap();
```

## API Design Philosophy

### Consistency with Individual Distributions

The standard API maintains consistency across the library:

```rust
// Individual observation
let x = 1.5;
let log_density = normal.log_density().at(&x);

// IID observations  
let samples = vec![1.5, 2.0, 1.8];
let log_density = iid_normal.log_density().at(&samples);
```

### Backward Compatibility

The manual API is preserved for existing code:

```rust
// Still works
let log_density = iid_normal.iid_log_density(&samples);
```

## Mathematical Properties Verified

### ✓ Exponential Family Structure
IID maintains all exponential family properties with proper parameter scaling

### ✓ Additivity  
For independent samples: `log p(x,y) = log p(x) + log p(y)`

### ✓ Sample Size Scaling  
For repeated patterns: `log p(2×pattern) = 2 × log p(pattern)`

### ✓ Sufficient Statistics
All parameter information contained in `∑ᵢT(xᵢ)` for IID samples

### ✓ Empty Sample Handling
Empty vector has log-density 0 (probability 1)

### ✓ Consistency
Multiple computation methods yield identical results (within numerical precision)

## Applications Enabled

### Statistical Inference
- **Maximum Likelihood Estimation**: Find parameters maximizing joint likelihood
- **Model Selection**: Compare models via likelihood ratios
- **Hypothesis Testing**: Likelihood ratio tests for nested models

### Bayesian Analysis
- **Conjugate Priors**: Exponential family structure enables analytical updates
- **Posterior Computation**: Efficient likelihood evaluation for MCMC
- **Model Evidence**: Marginal likelihood computation

### Computational Efficiency
- **Batch Processing**: Single computation for multiple density evaluations
- **Parallel Computation**: Independent samples enable parallelization  
- **Memory Efficiency**: Avoid redundant parameter computations

## Technical Design

### Trait Hierarchy
```
ExponentialFamily<X, F>
    ↓ IIDExtension
IID<D> : Measure<Vec<X>>
    ↓ HasLogDensity (automatic)
LogDensity<Vec<X>, IID<D>>
    ↓ EvaluateAt
f64 (or F: Float)
```

### Type Relationships
- Individual: `D: ExponentialFamily<X, F>`
- IID: `IID<D>: Measure<Vec<X>>`
- Root measures: `D::RootMeasure` → `IIDRootMeasure::IIDRoot`

### Implementation Strategy

The standard API works through:

1. **Automatic `HasLogDensity` Implementation**: IID distributions automatically get `HasLogDensity<Vec<X>, F>` via the blanket implementation for exponential families
2. **Overridden `exp_fam_log_density`**: The IID implementation overrides this method to handle sample size scaling correctly
3. **Efficient Computation**: Uses the specialized `compute_iid_exp_fam_log_density` function under the hood

### Future Extensions
The current implementation provides the foundation for:
1. **Full ExponentialFamily Integration**: Complete trait implementation for IID measures
2. **Automatic Caching**: Integration with generic exponential family cache
3. **SIMD Optimization**: Vectorized operations for large samples
4. **Conjugate Prior Support**: Bayesian inference capabilities

## Conclusion

The IID implementation successfully bridges individual and joint distributions while maintaining:
- **Mathematical Correctness**: Proper exponential family structure
- **API Consistency**: Standard interface matching individual distributions  
- **Performance**: Efficient computation using exponential family properties
- **Flexibility**: Support for both standard and manual APIs

This provides a solid foundation for statistical computing applications requiring IID sample handling. 