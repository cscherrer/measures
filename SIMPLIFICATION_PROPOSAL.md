# Simplification Proposal for Measures Codebase

## Overview

After reviewing the core, exponential_family, and iid modules, several opportunities for simplification have been identified that will reduce complexity without sacrificing performance or composability.

## 1. Core Density Module Simplification

### Current Issues
- 8+ overlapping traits for log-density computation
- Complex builder pattern that may be over-engineered
- Redundant functionality across multiple traits

### Proposed Changes

```rust
// Consolidate to just these core traits:
pub trait HasLogDensity<T, F> {
    fn log_density_wrt_root(&self, x: &T) -> F;
}

pub trait LogDensityBuilder<T>: Measure<T> {
    fn log_density(&self) -> LogDensity<T, Self>;
}

// Remove these traits:
// - LogDensityTrait (functionality merged into LogDensity struct)
// - GeneralLogDensity (use HasLogDensity directly)
// - HasLogDensityWrt (use LogDensity builder pattern)
// - LogDensityEval (redundant with HasLogDensity)
// - EvaluateAt (merge into LogDensity implementation)
```

### Benefits
- Reduces API surface by ~60%
- Clearer separation of concerns
- Maintains all functionality through simpler interfaces

## 2. Remove Redundant Helper Functions

### Current Issues
```rust
// This function just calls distribution.exp_fam_log_density(x)
pub fn compute_exp_fam_log_density<X, F, D>(distribution: &D, x: &X) -> F
```

### Proposed Changes
- Remove `compute_exp_fam_log_density` entirely
- Keep `compute_iid_exp_fam_log_density` as it provides real value
- Use direct method calls instead of wrapper functions

## 3. Simplify IID Implementation

### Current Issues
- Manual `IIDRootMeasure` implementations for each primitive measure
- Complex trait bounds and wrapper types

### Proposed Changes

```rust
// Simpler IID root measure handling
impl<M: Measure<X>, X: Clone> Measure<Vec<X>> for IID<M> {
    type RootMeasure = IID<M::RootMeasure>;
    
    fn root_measure(&self) -> Self::RootMeasure {
        IID::new(self.distribution.root_measure())
    }
}

// Remove IIDRootMeasure trait entirely
// Simplify IIDBaseMeasure to just wrap the base measure directly
```

### Benefits
- Eliminates ~50 lines of boilerplate code
- More composable design
- Easier to extend to new measure types

## 4. Keep Type-Level Programming (REVISED)

### Analysis
The type-level booleans (`True`, `False`, `TypeLevelBool`) are **essential** for trait dispatch disambiguation and should be kept. They solve a real problem that const generics cannot solve:

```rust
// This works and is necessary:
impl<T: Clone, F, M> HasLogDensity<T, F> for M
where
    M: Measure<T, IsExponentialFamily = True> + ExponentialFamily<T, F>,
    // ...

// This would NOT work with const generics:
// impl<T, const IS_EXP_FAM: bool> HasLogDensity<T, F> for M
// where M: Measure<T> + MeasureMarker<IS_EXPONENTIAL_FAMILY = IS_EXP_FAM>,
//       IS_EXP_FAM == true,  // ← Not valid Rust syntax
```

### What to Remove
- Unused `LogDensityMethod` types (`Specialized`, `ExponentialFamily`, `Default`)
- These are defined but never used in the codebase

## 5. Exponential Family Trait Simplification

### Current Issues
- `natural_and_log_partition()` method that just calls separate methods
- Redundant `ExponentialFamilyMeasure` marker trait

### Proposed Changes

```rust
pub trait ExponentialFamily<X: Clone, F: Float>: Clone {
    type NaturalParam: Clone;
    type SufficientStat;
    type BaseMeasure: Measure<X> + Clone;

    fn from_natural(param: Self::NaturalParam) -> Self;
    fn natural_params(&self) -> Self::NaturalParam;
    fn log_partition(&self) -> F;
    fn sufficient_statistic(&self, x: &X) -> Self::SufficientStat;
    fn base_measure(&self) -> Self::BaseMeasure;
    
    // Keep the optimized method but make it optional
    fn natural_params_and_log_partition(&self) -> (Self::NaturalParam, F) {
        (self.natural_params(), self.log_partition())
    }
}

// Remove ExponentialFamilyMeasure trait
```

## 6. DotProduct Trait Enhancement

### Current Issue
API design forces users to use `[T; 1]` instead of `T` for scalar parameters.

### Proposed Solution

```rust
// Add blanket implementation for scalar types
impl<T: Float> DotProduct<T> for T {
    type Output = T;
    fn dot(&self, other: &T) -> T {
        *self * *other
    }
}

// This allows natural parameter types to be just T instead of [T; 1]
```

## Implementation Strategy

### Phase 1: Core Simplification
1. Consolidate density traits
2. Remove redundant helper functions
3. Remove unused LogDensityMethod types (keep TypeLevelBool)

### Phase 2: IID Simplification
1. Simplify IID root measure handling
2. Remove IIDRootMeasure trait
3. Update all IID implementations

### Phase 3: API Enhancement
1. Add scalar DotProduct implementation
2. Update exponential family trait
3. Update all distribution implementations

## Expected Benefits

- **Reduced complexity**: ~30% fewer traits and types (revised from 40%)
- **Better ergonomics**: Simpler APIs for common use cases
- **Maintained performance**: All optimizations preserved
- **Enhanced composability**: Cleaner abstractions enable easier extension
- **Reduced maintenance burden**: Less code to maintain and document
- **Preserved trait dispatch**: Critical type-level programming kept intact

## Compatibility

All changes can be made in a backward-compatible way by:
1. Deprecating old traits/functions rather than removing immediately
2. Providing migration guides
3. Using feature flags for new simplified APIs

This proposal maintains all current functionality while significantly reducing complexity and improving the developer experience, while preserving the essential trait dispatch mechanisms. 