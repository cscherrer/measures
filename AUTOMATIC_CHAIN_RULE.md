# Automatic Chain Rule Implementation

## Mathematical Foundation

The automatic chain rule implementation addresses the complete exponential family log-density formula:

```
log f(x|θ) = η(θ)ᵀT(x) - A(η(θ)) + log h(x)
```

Where `log h(x)` represents the log-density of the base measure with respect to the root measure. Previous implementations omitted this term, requiring manual computation in distribution-specific code.

## Architecture

### Enhanced Default Implementation

The `ExponentialFamily` trait now provides a complete default implementation:

**Before:**
```rust
fn exp_fam_log_density(&self, x: &X) -> F {
    let natural_params = self.to_natural();
    let sufficient_stats = self.sufficient_statistic(x);
    let log_partition = self.log_partition();
    
    // Only exponential family part - MISSING base measure density!
    natural_params.dot(&sufficient_stats) - log_partition
}
```

**After:**
```rust
fn exp_fam_log_density(&self, x: &X) -> F {
    let natural_params = self.to_natural();
    let sufficient_stats = self.sufficient_statistic(x);
    let log_partition = self.log_partition();

    // Standard exponential family part: η·T(x) - A(η)
    let exp_fam_part = natural_params.dot(&sufficient_stats) - log_partition;
    
    // Chain rule part: log-density of base measure with respect to root measure
    let base_measure = self.base_measure();
    let chain_rule_part = base_measure.log_density_wrt_root(x);
    
    // Complete log-density: exponential family + chain rule
    exp_fam_part + chain_rule_part
}
```

### Base Measure Implementation

#### FactorialMeasure for Discrete Distributions

A new `FactorialMeasure` type handles discrete distributions requiring factorial normalization:

```rust
/// Factorial measure for discrete distributions.
/// Represents dν = (1/k!) dμ where μ is the counting measure.
pub struct FactorialMeasure<F: Float> {
    counting: CountingMeasure<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> HasLogDensity<u64, F> for FactorialMeasure<F> {
    fn log_density_wrt_root(&self, x: &u64) -> F {
        // Compute -log(k!) = -sum(log(i) for i in 1..=k)
        if *x == 0 {
            F::zero() // log(0!) = log(1) = 0
        } else {
            let mut neg_log_factorial = F::zero();
            for i in 1..=*x {
                neg_log_factorial = neg_log_factorial - F::from(i).unwrap().ln();
            }
            neg_log_factorial
        }
    }
}
```

Note: This implementation demonstrates the mathematical concept. The actual implementation uses optimized factorial computation.

## Implementation Details

### Poisson Distribution Integration

The Poisson distribution now uses `FactorialMeasure` as its base measure:

**Before:** Required manual override with factorial computation
**After:** Uses `FactorialMeasure` as base measure, no override needed

```rust
impl<F: Float + FloatConst> ExponentialFamily<u64, F> for Poisson<F> {
    type BaseMeasure = FactorialMeasure<F>;  // Proper factorial base measure
    
    fn base_measure(&self) -> Self::BaseMeasure {
        FactorialMeasure::new()
    }
    
    // No override needed - default implementation handles complete formula
}
```

### Normal Distribution Compatibility

For continuous distributions using Lebesgue measure as both root and base measure:

```rust
impl<T: Float + FloatConst> ExponentialFamily<T, T> for Normal<T> {
    type BaseMeasure = LebesgueMeasure<T>;  // Same as root measure
    
    fn base_measure(&self) -> Self::BaseMeasure {
        LebesgueMeasure::new()
    }
    
    // Chain rule part = 0 since base_measure == root_measure
}
```

When `base_measure == root_measure`, the chain rule term equals zero, maintaining backward compatibility.

## Mathematical Verification

### Computational Validation

The implementation produces identical results to manual computation:

```rust
// Automatic computation using default implementation
let auto_result = poisson.exp_fam_log_density(&k);

// Manual computation for verification
let natural_params = poisson.to_natural();
let sufficient_stats = poisson.sufficient_statistic(&k);
let log_partition = poisson.log_partition();
let factorial_term = -log_factorial(k as f64);

let manual_result = natural_params[0] * sufficient_stats[0] - log_partition + factorial_term;

assert!((auto_result - manual_result).abs() < 1e-10);
```

### Reference Implementation Comparison

Results match external reference implementations (rv crate) within numerical precision:

- **Poisson validation**: All test points pass with error < 1e-10
- **Normal validation**: Exact agreement with reference values
- **Edge cases**: Correct handling of k=0, large k values

## Design Benefits

### Elimination of Manual Overrides

Previously, each discrete distribution required manual implementation of factorial terms. The automatic chain rule eliminates this requirement:

- **Before**: Each distribution implements custom log-density logic
- **After**: Define base measure, get correct mathematics automatically

### Mathematical Rigor

The implementation ensures complete adherence to measure theory:

- Implements the full exponential family formula including base measure densities
- Handles chain rule automatically via base measure log-density computation
- Works for any base measure implementing `HasLogDensity`

### Backward Compatibility

Existing continuous distributions continue to work unchanged:
- When `base_measure == root_measure`, chain rule contribution is zero
- No API changes required for existing code
- Performance characteristics preserved

## Future Extensions

### Distribution Support

The framework enables straightforward implementation of additional discrete distributions:

- **Binomial**: Can use `FactorialMeasure` for combinatorial factors
- **Geometric**: Uses `CountingMeasure` directly
- **Negative Binomial**: Combination of factorial and beta function base measures

### Custom Base Measures

The extension framework allows domain-specific base measures:

- **Beta functions**: For distributions requiring Gamma function normalization
- **Spherical measures**: For distributions on manifolds
- **Custom normalizations**: Application-specific measure definitions

## Implementation Verification

### Test Coverage

Comprehensive testing validates the automatic chain rule:

- **Unit tests**: Individual component verification
- **Integration tests**: Complete distribution implementations
- **Reference comparison**: Agreement with established libraries
- **Edge case handling**: Boundary conditions and special values

### Performance Impact

The automatic chain rule maintains performance characteristics:

- **Zero overhead**: When `base_measure == root_measure`
- **Minimal overhead**: Single additional computation for factorial measures
- **Optimization preservation**: Compiler inlining and specialization maintained

## Conclusion

The automatic chain rule implementation provides mathematically complete exponential family support while eliminating implementation burden for new distributions. The approach maintains architectural integrity and performance while ensuring mathematical correctness through systematic application of measure theory principles. 