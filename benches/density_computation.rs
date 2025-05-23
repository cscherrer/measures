use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, black_box, criterion_group,
    criterion_main,
};
use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};
use pprof::criterion::{Output, PProfProfiler};
use rv::dist::{Gaussian, Poisson as RvPoisson};
use rv::prelude::*;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Benchmark single density evaluations with rv comparisons
/// Reduced from 5 inputs to 2 representative ones
fn bench_single_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_density_evaluation");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Use only 2 representative inputs instead of 5
    let inputs: Vec<f64> = vec![0.5, 2.0];
    let k_inputs: Vec<u64> = vec![1, 10];

    for (i, &x) in inputs.iter().enumerate() {
        let normal = black_box(Normal::new(0.0_f64, 1.0_f64));
        let rv_normal = black_box(Gaussian::new(0.0, 1.0).unwrap());

        group.bench_function(format!("measures_normal_exp_fam_{i}"), |b| {
            b.iter(|| {
                let result = black_box(&normal).exp_fam_log_density(black_box(&x));
                black_box(result)
            });
        });

        group.bench_function(format!("rv_normal_ln_f_{i}"), |b| {
            b.iter(|| {
                let result = black_box(&rv_normal).ln_f(black_box(&x));
                black_box(result)
            });
        });

        group.bench_function(format!("measures_normal_log_density_{i}"), |b| {
            b.iter(|| {
                let ld = black_box(&normal).log_density();
                let result: f64 = ld.at(black_box(&x));
                black_box(result)
            });
        });
    }

    for (i, &k) in k_inputs.iter().enumerate() {
        let poisson = black_box(Poisson::new(2.5_f64));
        let rv_poisson = black_box(RvPoisson::new(2.5).unwrap());

        group.bench_function(format!("measures_poisson_exp_fam_{i}"), |b| {
            b.iter(|| {
                let result = black_box(&poisson).exp_fam_log_density(black_box(&k));
                black_box(result)
            });
        });

        group.bench_function(format!("rv_poisson_ln_f_{i}"), |b| {
            b.iter(|| {
                let result = black_box(&rv_poisson).ln_f(black_box(&(k as usize)));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark batch evaluations - reduced to 2 sizes and most important patterns
fn bench_batch_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_density_evaluation");

    // Test with only 2 batch sizes instead of 3
    for size in &[100, 1000] {
        let points: Vec<f64> = (0..*size).map(|i| f64::from(i) * 0.1).collect();
        let normal = Normal::new(0.0_f64, 1.0_f64);

        group.bench_with_input(
            BenchmarkId::new("normal_repeated_clone", size),
            size,
            |b, _| {
                b.iter(|| {
                    for &x in &points {
                        let ld = normal.log_density();
                        let result: f64 = ld.at(black_box(&x));
                        black_box(result);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("normal_reused_log_density", size),
            size,
            |b, _| {
                b.iter(|| {
                    let ld = normal.log_density();
                    for &x in &points {
                        let result: f64 = ld.at(black_box(&x));
                        black_box(result);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("normal_direct_computation", size),
            size,
            |b, _| {
                b.iter(|| {
                    for &x in &points {
                        let result = normal.log_density_wrt_root(black_box(&x));
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Consolidated factorial optimization benchmark (combines previous factorial tests)
/// This replaces both bench_poisson_factorial and bench_factorial_optimization
fn bench_factorial_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial_optimization");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

    let poisson = black_box(Poisson::new(2.5_f64));
    let rv_poisson = black_box(RvPoisson::new(2.5).unwrap());

    // Test key k values that demonstrate O(1) vs O(k) scaling
    for k in &[1, 10, 50, 200, 1000] {
        // Our O(1) optimized approach
        group.bench_with_input(
            BenchmarkId::new("measures_o1_stirling", k),
            k,
            |b, &k| {
                b.iter(|| {
                    let result = black_box(&poisson).exp_fam_log_density(black_box(&k));
                    black_box(result)
                });
            },
        );

        // rv reference implementation
        group.bench_with_input(BenchmarkId::new("rv_poisson_reference", k), k, |b, &k| {
            b.iter(|| {
                let result = black_box(&rv_poisson).ln_f(black_box(&(k as usize)));
                black_box(result)
            });
        });

        // O(k) naive approach (only for smaller k to avoid excessive runtime)
        if k <= &100 {
            group.bench_with_input(
                BenchmarkId::new("naive_ok_factorial", k),
                k,
                |b, &k| {
                    b.iter(|| {
                        let mut log_factorial = 0.0_f64;
                        for i in 1..=black_box(k) {
                            log_factorial += (i as f64).ln();
                        }
                        let lambda = 2.5_f64;
                        let result = (k as f64) * lambda.ln() - lambda - log_factorial;
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Consolidated component benchmark (combines exp_fam_components and component_creation)
fn bench_exponential_family_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_family_components");

    let normal = Normal::new(1.0_f64, 2.0_f64);
    let poisson = Poisson::new(2.5_f64);
    let x = 0.5_f64;
    let k = 5u64;

    // Core exponential family operations
    group.bench_function("normal_to_natural", |b| {
        b.iter(|| {
            let result = normal.to_natural();
            black_box(result)
        });
    });

    group.bench_function("normal_sufficient_statistic", |b| {
        b.iter(|| {
            let result = normal.sufficient_statistic(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("normal_log_partition", |b| {
        b.iter(|| {
            let result = normal.log_partition();
            black_box(result)
        });
    });

    group.bench_function("poisson_to_natural", |b| {
        b.iter(|| {
            let result = poisson.to_natural();
            black_box(result)
        });
    });

    group.bench_function("poisson_sufficient_statistic", |b| {
        b.iter(|| {
            let result = poisson.sufficient_statistic(black_box(&k));
            black_box(result)
        });
    });

    // Compare component-wise vs integrated computation
    group.bench_function("normal_components_separate", |b| {
        b.iter(|| {
            let nat_params = normal.to_natural();
            let suff_stats = normal.sufficient_statistic(black_box(&x));
            let log_partition = normal.log_partition();
            let base_measure = normal.base_measure();
            let base_density: f64 = base_measure.log_density_wrt_root(black_box(&x));

            let result = nat_params[0] * suff_stats[0] + nat_params[1] * suff_stats[1]
                - log_partition
                + base_density;
            black_box(result)
        });
    });

    group.bench_function("normal_components_integrated", |b| {
        b.iter(|| {
            let result = normal.exp_fam_log_density(black_box(&x));
            black_box(result)
        });
    });

    group.finish();
}

/// Essential allocation pattern tests
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    let normal = Normal::new(0.0_f64, 1.0_f64);
    let x = 0.5_f64;

    // Core allocation comparison
    group.bench_function("minimal_allocations", |b| {
        b.iter(|| {
            let result = normal.log_density_wrt_root(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("heavy_allocations", |b| {
        b.iter(|| {
            let ld = normal.log_density();
            let result: f64 = ld.at(black_box(&x));
            black_box(result)
        });
    });

    // Batch allocation pattern (simplified)
    let points: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();

    group.bench_function("batch_clone_per_eval", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &points {
                let ld = normal.log_density();
                sum += ld.at(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.bench_function("batch_single_clone", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            let ld = normal.log_density();
            for &x in &points {
                sum += ld.at(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Basic creation and cloning benchmarks
fn bench_measure_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_creation");

    group.bench_function("normal_creation", |b| {
        b.iter(|| {
            let normal = Normal::new(black_box(0.0_f64), black_box(1.0_f64));
            black_box(normal)
        });
    });

    let normal = Normal::new(0.0_f64, 1.0_f64);
    group.bench_function("normal_clone", |b| {
        b.iter(|| {
            let cloned = normal.clone();
            black_box(cloned)
        });
    });

    group.bench_function("poisson_creation", |b| {
        b.iter(|| {
            let poisson = Poisson::new(black_box(2.5_f64));
            black_box(poisson)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_single_evaluations, bench_batch_evaluations, bench_factorial_optimization,
              bench_exponential_family_components, bench_allocation_patterns, bench_measure_creation
}

criterion_main!(benches);
