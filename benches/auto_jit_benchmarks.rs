//! Comprehensive benchmarks for automatic JIT compilation derivation
//!
//! This benchmark suite compares performance across different optimization levels:
//! 1. Standard trait-based evaluation
//! 2. Zero-overhead compile-time optimization
//! 3. Automatic JIT compilation
//!
//! Run with: cargo bench --features jit

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use measures::{Exponential, LogDensityBuilder, Normal};

#[cfg(feature = "jit")]
use measures::exponential_family::{AutoJITExt, CustomJITOptimizer};

#[cfg(feature = "jit")]
use measures::exponential_family::jit::generate_zero_overhead_exp_fam;

fn benchmark_normal_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_distribution");

    let normal = Normal::new(2.0, 1.5);
    let test_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 5.0, 10.0, -5.0];

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
        let optimized_fn = generate_zero_overhead_exp_fam(normal.clone());
        b.iter(|| {
            for &x in &test_values {
                black_box(optimized_fn(&black_box(x)));
            }
        });
    });

    // Automatic JIT compilation
    #[cfg(feature = "jit")]
    group.bench_function("auto_jit", |b| {
        let jit_fn = normal.auto_jit().expect("JIT compilation should succeed");
        b.iter(|| {
            for &x in &test_values {
                black_box(jit_fn.call(black_box(x)));
            }
        });
    });

    // Manual JIT implementation (for comparison)
    #[cfg(feature = "jit")]
    group.bench_function("manual_jit", |b| {
        let symbolic = normal.custom_symbolic_log_density();
        let compiler = measures::exponential_family::jit::JITCompiler::new().unwrap();
        let jit_fn = compiler
            .compile_custom_expression(&symbolic)
            .expect("Manual JIT should succeed");
        b.iter(|| {
            for &x in &test_values {
                black_box(jit_fn.call(black_box(x)));
            }
        });
    });

    group.finish();
}

fn benchmark_exponential_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_distribution");

    let exponential = Exponential::new(2.0);
    let test_values = vec![0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];

    // Standard evaluation
    group.bench_function("standard", |b| {
        let log_density = exponential.log_density();
        b.iter(|| {
            for &x in &test_values {
                black_box(log_density.at(&black_box(x)));
            }
        });
    });

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("zero_overhead", |b| {
        let optimized_fn = generate_zero_overhead_exp_fam(exponential.clone());
        b.iter(|| {
            for &x in &test_values {
                black_box(optimized_fn(&black_box(x)));
            }
        });
    });

    // Automatic JIT compilation
    #[cfg(feature = "jit")]
    group.bench_function("auto_jit", |b| {
        let jit_fn = exponential
            .auto_jit()
            .expect("JIT compilation should succeed");
        b.iter(|| {
            for &x in &test_values {
                black_box(jit_fn.call(black_box(x)));
            }
        });
    });

    group.finish();
}

fn benchmark_single_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_evaluation");

    let normal = Normal::new(0.0, 1.0);
    let test_x = 1.5;

    // Standard evaluation
    group.bench_function("normal_standard", |b| {
        let log_density = normal.log_density();
        b.iter(|| black_box(log_density.at(&black_box(test_x))));
    });

    // Zero-overhead optimization
    #[cfg(feature = "jit")]
    group.bench_function("normal_zero_overhead", |b| {
        let optimized_fn = generate_zero_overhead_exp_fam(normal.clone());
        b.iter(|| black_box(optimized_fn(&black_box(test_x))));
    });

    // Automatic JIT compilation
    #[cfg(feature = "jit")]
    group.bench_function("normal_auto_jit", |b| {
        let jit_fn = normal.auto_jit().expect("JIT compilation should succeed");
        b.iter(|| black_box(jit_fn.call(black_box(test_x))));
    });

    group.finish();
}

fn benchmark_compilation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_overhead");

    let normal = Normal::new(1.0, 2.0);

    // Measure JIT compilation time
    #[cfg(feature = "jit")]
    group.bench_function("auto_jit_compilation", |b| {
        b.iter(|| {
            let _jit_fn = black_box(normal.clone())
                .auto_jit()
                .expect("JIT compilation should succeed");
        });
    });

    // Measure symbolic generation time
    #[cfg(feature = "jit")]
    group.bench_function("symbolic_generation", |b| {
        b.iter(|| {
            let _symbolic = black_box(normal.clone())
                .auto_symbolic()
                .expect("Symbolic generation should succeed");
        });
    });

    group.finish();
}

fn benchmark_parameter_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_variations");

    let test_x = 1.0;
    let parameters = vec![
        (0.0, 1.0),  // Standard normal
        (2.0, 0.5),  // High mean, low variance
        (-1.0, 3.0), // Negative mean, high variance
        (10.0, 0.1), // High mean, very low variance
    ];

    for (i, &(mu, sigma)) in parameters.iter().enumerate() {
        let normal = Normal::new(mu, sigma);

        // Standard evaluation
        group.bench_with_input(BenchmarkId::new("standard", i), &normal, |b, normal| {
            let log_density = normal.log_density();
            b.iter(|| black_box(log_density.at(&black_box(test_x))));
        });

        // Automatic JIT compilation
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("auto_jit", i), &normal, |b, normal| {
            let jit_fn = normal.auto_jit().expect("JIT compilation should succeed");
            b.iter(|| black_box(jit_fn.call(black_box(test_x))));
        });
    }

    group.finish();
}

fn benchmark_accuracy_vs_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_vs_performance");

    let normal = Normal::new(0.0, 1.0);
    let test_values: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 100.0).collect();

    // Measure accuracy and performance together
    group.bench_function("standard_accuracy", |b| {
        let log_density = normal.log_density();
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += log_density.at(&black_box(x));
            }
            black_box(sum)
        });
    });

    #[cfg(feature = "jit")]
    group.bench_function("auto_jit_accuracy", |b| {
        let jit_fn = normal.auto_jit().expect("JIT compilation should succeed");
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += jit_fn.call(black_box(x));
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Benchmark memory allocation patterns
    group.bench_function("create_standard_evaluator", |b| {
        b.iter(|| {
            let normal = black_box(Normal::new(0.0, 1.0));
            let _log_density = black_box(normal.log_density());
        });
    });

    #[cfg(feature = "jit")]
    group.bench_function("create_jit_function", |b| {
        b.iter(|| {
            let normal = black_box(Normal::new(0.0, 1.0));
            let _jit_fn = black_box(normal.auto_jit().expect("JIT compilation should succeed"));
        });
    });

    group.finish();
}

fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    let normal = Normal::new(0.0, 1.0);
    let sizes = vec![10, 100, 1000, 10000];

    for &size in &sizes {
        let test_values: Vec<f64> = (0..size).map(|i| f64::from(i) / f64::from(size)).collect();

        group.bench_with_input(
            BenchmarkId::new("standard", size),
            &test_values,
            |b, values| {
                let log_density = normal.log_density();
                b.iter(|| {
                    for &x in values {
                        black_box(log_density.at(&black_box(x)));
                    }
                });
            },
        );

        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("auto_jit", size),
            &test_values,
            |b, values| {
                let jit_fn = normal.auto_jit().expect("JIT compilation should succeed");
                b.iter(|| {
                    for &x in values {
                        black_box(jit_fn.call(black_box(x)));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_normal_distribution,
    benchmark_exponential_distribution,
    benchmark_single_evaluation,
    benchmark_compilation_overhead,
    benchmark_parameter_variations,
    benchmark_accuracy_vs_performance,
    benchmark_memory_usage,
    benchmark_scalability
);

criterion_main!(benches);
