//! Simple benchmarks for JIT compilation performance
//!
//! This benchmark suite focuses on the core performance comparison between
//! standard evaluation and JIT compilation for the automatic derivation system.
//!
//! Run with: cargo bench --bench `simple_jit_benchmarks` --features jit

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use measures::{Exponential, LogDensityBuilder, Normal};
use std::hint::black_box;

#[cfg(feature = "jit")]
use measures::exponential_family::AutoJITExt;

#[cfg(feature = "jit")]
use measures::exponential_family::jit::generate_zero_overhead_exp_fam;

fn benchmark_normal_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_performance");

    let normal = Normal::new(2.0, 1.5);
    let test_x = 2.5;

    // Standard evaluation
    group.bench_function("standard", |b| {
        let log_density = normal.log_density();
        b.iter(|| black_box(log_density.at(&black_box(test_x))));
    });

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead", |b| {
        let optimized_fn = generate_zero_overhead_exp_fam(normal.clone());
        b.iter(|| black_box(optimized_fn(&black_box(test_x))));
    });

    // Pre-compiled JIT function (avoid compilation overhead in benchmark)
    #[cfg(feature = "jit")]
    {
        if let Ok(jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit", |b| {
                b.iter(|| black_box(jit_fn.call(black_box(test_x))));
            });
        }
    }

    group.finish();
}

fn benchmark_exponential_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_performance");

    let exponential = Exponential::new(2.0);
    let test_x = 1.5;

    // Standard evaluation
    group.bench_function("standard", |b| {
        let log_density = exponential.log_density();
        b.iter(|| black_box(log_density.at(&black_box(test_x))));
    });

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead", |b| {
        let optimized_fn = generate_zero_overhead_exp_fam(exponential.clone());
        b.iter(|| black_box(optimized_fn(&black_box(test_x))));
    });

    // Pre-compiled JIT function
    #[cfg(feature = "jit")]
    {
        if let Ok(jit_fn) = exponential.auto_jit() {
            group.bench_function("auto_jit", |b| {
                b.iter(|| black_box(jit_fn.call(black_box(test_x))));
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
    group.bench_function("standard_batch", |b| {
        let log_density = normal.log_density();
        b.iter(|| {
            for &x in &test_values {
                black_box(log_density.at(&black_box(x)));
            }
        });
    });

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead_batch", |b| {
        let optimized_fn = generate_zero_overhead_exp_fam(normal.clone());
        b.iter(|| {
            for &x in &test_values {
                black_box(optimized_fn(&black_box(x)));
            }
        });
    });

    // Pre-compiled JIT function
    #[cfg(feature = "jit")]
    {
        if let Ok(jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit_batch", |b| {
                b.iter(|| {
                    for &x in &test_values {
                        black_box(jit_fn.call(black_box(x)));
                    }
                });
            });
        }
    }

    group.finish();
}

fn benchmark_parameter_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_variations");

    let test_x = 1.0;
    let parameters = vec![
        (0.0, 1.0),  // Standard normal
        (2.0, 0.5),  // High mean, low variance
        (-1.0, 3.0), // Negative mean, high variance
    ];

    for (i, &(mu, sigma)) in parameters.iter().enumerate() {
        let normal = Normal::new(mu, sigma);

        // Standard evaluation
        group.bench_with_input(BenchmarkId::new("standard", i), &normal, |b, normal| {
            let log_density = normal.log_density();
            b.iter(|| black_box(log_density.at(&black_box(test_x))));
        });

        // Auto-JIT compilation (if available)
        #[cfg(feature = "jit")]
        {
            if let Ok(jit_fn) = normal.auto_jit() {
                group.bench_with_input(BenchmarkId::new("auto_jit", i), &jit_fn, |b, jit_fn| {
                    b.iter(|| black_box(jit_fn.call(black_box(test_x))));
                });
            }
        }
    }

    group.finish();
}

fn benchmark_accuracy_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_preservation");

    let normal = Normal::new(0.0, 1.0);
    let test_values: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 100.0).collect();

    // Standard evaluation with accuracy check
    group.bench_function("standard_with_accuracy", |b| {
        let log_density = normal.log_density();
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += log_density.at(&black_box(x));
            }
            black_box(sum)
        });
    });

    // JIT evaluation with accuracy check
    #[cfg(feature = "jit")]
    {
        if let Ok(jit_fn) = normal.auto_jit() {
            group.bench_function("auto_jit_with_accuracy", |b| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in &test_values {
                        sum += jit_fn.call(black_box(x));
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
    benchmark_normal_performance,
    benchmark_exponential_performance,
    benchmark_batch_evaluation,
    benchmark_parameter_variations,
    benchmark_accuracy_preservation
);

criterion_main!(benches);
