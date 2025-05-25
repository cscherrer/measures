//! Comprehensive JIT Comparison Benchmark
//!
//! This benchmark compares different JIT approaches in their intended use cases:
//! 1. Standard evaluation (baseline)
//! 2. Zero-overhead optimization (compile-time)
//! 3. Static Inline JIT (runtime closures with embedded constants)
//! 4. Auto-JIT (symbolic derivation + Cranelift compilation)
//!
//! Run with: cargo bench --bench `jit_comparison_bench` --features jit

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use measures::{LogDensityBuilder, Normal};
use std::hint::black_box;

#[cfg(feature = "jit")]
use measures::exponential_family::jit::{StaticInlineJITOptimizer, ZeroOverheadOptimizer};

#[cfg(feature = "jit")]
use measures::exponential_family::AutoJITExt;

fn benchmark_single_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_evaluation");
    
    let normal = Normal::new(2.0, 1.5);
    let test_x = 2.5;
    
    // 1. Standard evaluation (baseline)
    group.bench_function("standard", |b| {
        let log_density = normal.log_density();
        b.iter(|| black_box(log_density.at(&black_box(test_x))));
    });
    
    // 2. Zero-overhead optimization (compile-time)
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead", |b| {
        let optimized_fn = normal.clone().zero_overhead_optimize();
        b.iter(|| black_box(optimized_fn(&black_box(test_x))));
    });
    
    // 3. Static Inline JIT (runtime closures)
    #[cfg(feature = "jit")]
    {
        if let Ok(static_jit_fn) = normal.compile_static_inline_jit() {
            group.bench_function("static_inline_jit", |b| {
                b.iter(|| black_box(static_jit_fn.call(black_box(test_x))));
            });
        }
    }
    
    // 4. Auto-JIT (symbolic + Cranelift)
    #[cfg(feature = "jit")]
    {
        if let Ok(auto_jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit", |b| {
                b.iter(|| black_box(auto_jit_fn.call(black_box(test_x))));
            });
        }
    }
    
    group.finish();
}

fn benchmark_batch_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_evaluation");
    
    let normal = Normal::new(0.0, 1.0);
    let test_values: Vec<f64> = (0..100).map(|i| (f64::from(i) - 50.0) / 10.0).collect();
    
    // Standard evaluation
    group.bench_function("standard", |b| {
        let log_density = normal.log_density();
        b.iter(|| {
            for &x in &test_values {
                black_box(log_density.at(&black_box(x)));
            }
        });
    });
    
    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead", |b| {
        let optimized_fn = normal.clone().zero_overhead_optimize();
        b.iter(|| {
            for &x in &test_values {
                black_box(optimized_fn(&black_box(x)));
            }
        });
    });
    
    // Static Inline JIT
    #[cfg(feature = "jit")]
    {
        if let Ok(static_jit_fn) = normal.compile_static_inline_jit() {
            group.bench_function("static_inline_jit", |b| {
                b.iter(|| {
                    for &x in &test_values {
                        black_box(static_jit_fn.call(black_box(x)));
                    }
                });
            });
        }
    }
    
    // Auto-JIT
    #[cfg(feature = "jit")]
    {
        if let Ok(auto_jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit", |b| {
                b.iter(|| {
                    for &x in &test_values {
                        black_box(auto_jit_fn.call(black_box(x)));
                    }
                });
            });
        }
    }
    
    group.finish();
}

fn benchmark_compilation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_overhead");
    
    let normal = Normal::new(1.0, 2.0);
    
    // Static Inline JIT compilation (should be very fast)
    #[cfg(feature = "jit")]
    group.bench_function("static_inline_jit_compilation", |b| {
        b.iter(|| {
            let normal_clone = normal.clone();
            let _jit_fn = normal_clone
                .compile_static_inline_jit()
                .expect("Static inline JIT should succeed");
        });
    });
    
    // Auto-JIT compilation (includes symbolic derivation + Cranelift)
    #[cfg(feature = "jit")]
    group.bench_function("auto_jit_compilation", |b| {
        b.iter(|| {
            let normal_clone = black_box(normal.clone());
            let _jit_fn = normal_clone
                .auto_jit()
                .expect("Auto-JIT should succeed");
        });
    });
    
    // Zero-overhead "compilation" (just closure creation)
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead_compilation", |b| {
        b.iter(|| {
            let _optimized_fn = black_box(normal.clone()).zero_overhead_optimize();
        });
    });
    
    group.finish();
}

fn benchmark_amortization_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("amortization_analysis");
    
    let normal = Normal::new(0.0, 1.0);
    let test_x = 1.5;
    
    // Test different numbers of evaluations to find break-even points
    let call_counts = vec![1, 10, 100, 1000, 10000];
    
    for &count in &call_counts {
        // Auto-JIT with compilation overhead included
        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("auto_jit_with_compilation", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    // Include compilation time
                    let jit_fn = normal.auto_jit().expect("Auto-JIT should succeed");
                    // Then evaluate multiple times
                    for _ in 0..count {
                        black_box(jit_fn.call(black_box(test_x)));
                    }
                });
            },
        );
        
        // Static Inline JIT with compilation overhead included
        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("static_inline_jit_with_compilation", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    // Include compilation time
                    let jit_fn = normal.compile_static_inline_jit().expect("Static inline JIT should succeed");
                    // Then evaluate multiple times
                    for _ in 0..count {
                        black_box(jit_fn.call(black_box(test_x)));
                    }
                });
            },
        );
        
        // Standard evaluation for comparison
        group.bench_with_input(
            BenchmarkId::new("standard", count),
            &count,
            |b, &count| {
                let log_density = normal.log_density();
                b.iter(|| {
                    for _ in 0..count {
                        black_box(log_density.at(&black_box(test_x)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_accuracy_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_preservation");
    
    let normal = Normal::new(0.0, 1.0);
    let test_values: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 100.0).collect();
    
    // Standard evaluation with accuracy verification
    group.bench_function("standard_with_verification", |b| {
        let log_density = normal.log_density();
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += log_density.at(&black_box(x));
            }
            black_box(sum)
        });
    });
    
    // Static Inline JIT with accuracy verification
    #[cfg(feature = "jit")]
    {
        if let Ok(static_jit_fn) = normal.compile_static_inline_jit() {
            group.bench_function("static_inline_jit_with_verification", |b| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in &test_values {
                        sum += static_jit_fn.call(black_box(x));
                    }
                    black_box(sum)
                });
            });
        }
    }
    
    // Auto-JIT with accuracy verification
    #[cfg(feature = "jit")]
    {
        if let Ok(auto_jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit_with_verification", |b| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in &test_values {
                        sum += auto_jit_fn.call(black_box(x));
                    }
                    black_box(sum)
                });
            });
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_evaluation,
    benchmark_batch_evaluation,
    benchmark_compilation_overhead,
    benchmark_amortization_analysis,
    benchmark_accuracy_preservation
);

criterion_main!(benches); 