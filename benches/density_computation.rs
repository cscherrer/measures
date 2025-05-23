use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};
use pprof::criterion::{Output, PProfProfiler};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Benchmark single density evaluations
fn bench_single_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_density_evaluation");

    // Normal distribution benchmarks
    let normal = Normal::new(0.0_f64, 1.0_f64);
    let x = 0.5_f64;

    group.bench_function("normal_log_density", |b| {
        b.iter(|| {
            let ld = normal.log_density();
            let result: f64 = ld.at(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("normal_exp_fam_log_density", |b| {
        b.iter(|| {
            let result = normal.exp_fam_log_density(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("normal_log_density_wrt_root", |b| {
        b.iter(|| {
            let result = normal.log_density_wrt_root(black_box(&x));
            black_box(result)
        });
    });

    // Poisson distribution benchmarks
    let poisson = Poisson::new(2.5_f64);
    let k = 3u64;

    group.bench_function("poisson_log_density", |b| {
        b.iter(|| {
            let ld = poisson.log_density();
            let result: f64 = ld.at(black_box(&k));
            black_box(result)
        });
    });

    group.bench_function("poisson_exp_fam_log_density", |b| {
        b.iter(|| {
            let result = poisson.exp_fam_log_density(black_box(&k));
            black_box(result)
        });
    });

    group.bench_function("poisson_log_density_wrt_root", |b| {
        b.iter(|| {
            let result = poisson.log_density_wrt_root(black_box(&k));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark batch evaluations to see how cloning affects performance
fn bench_batch_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_density_evaluation");

    // Test with different batch sizes
    for size in &[10, 100, 1000] {
        let points: Vec<f64> = (0..*size).map(|i| f64::from(i) * 0.1).collect();
        let normal = Normal::new(0.0_f64, 1.0_f64);

        group.bench_with_input(
            BenchmarkId::new("normal_repeated_clone", size),
            size,
            |b, _| {
                b.iter(|| {
                    for &x in &points {
                        // This creates a new LogDensity each time (lots of cloning)
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
                    // Reuse the LogDensity object
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
                    // Direct computation without LogDensity wrapper
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

/// Benchmark Poisson performance (now O(1) optimized)
fn bench_poisson_factorial(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_performance");

    let poisson = Poisson::new(2.5_f64);

    // Test our O(1) factorial computation across different k values
    // This validates that performance is indeed O(1) regardless of k
    for k in &[0, 1, 5, 10, 20, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("poisson_optimized", k), k, |b, &k| {
            b.iter(|| {
                let result = poisson.exp_fam_log_density(black_box(&k));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark exponential family components
fn bench_exp_fam_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_family_components");

    let normal = Normal::new(1.0_f64, 2.0_f64);
    let x = 0.5_f64;

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

    group.bench_function("normal_base_measure", |b| {
        b.iter(|| {
            let result = normal.base_measure();
            black_box(result)
        });
    });

    // Poisson components
    let poisson = Poisson::new(2.5_f64);
    let k = 5u64;

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

    group.bench_function("poisson_base_measure", |b| {
        b.iter(|| {
            let result = poisson.base_measure();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark measure creation and cloning
fn bench_measure_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_creation_and_cloning");

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

    let poisson = Poisson::new(2.5_f64);
    group.bench_function("poisson_clone", |b| {
        b.iter(|| {
            let cloned = poisson.clone();
            black_box(cloned)
        });
    });

    group.finish();
}

/// Benchmark memory allocation patterns (run with dhat for detailed analysis)
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    // Test Normal distribution cloning patterns
    let normal = Normal::new(0.0_f64, 1.0_f64);
    let x = 0.5_f64;

    group.bench_function("normal_minimal_allocations", |b| {
        b.iter(|| {
            // Minimal allocation path - direct computation
            let result = normal.log_density_wrt_root(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("normal_medium_allocations", |b| {
        b.iter(|| {
            // Medium allocation path - exp_fam method (some cloning)
            let result = normal.exp_fam_log_density(black_box(&x));
            black_box(result)
        });
    });

    group.bench_function("normal_heavy_allocations", |b| {
        b.iter(|| {
            // Heavy allocation path - full LogDensity construction
            let ld = normal.log_density();
            let result: f64 = ld.at(black_box(&x));
            black_box(result)
        });
    });

    // Test Poisson factorial computation memory patterns
    let poisson = Poisson::new(2.5_f64);
    let k = 10u64; // Larger k to stress test factorial computation

    group.bench_function("poisson_factorial_allocations", |b| {
        b.iter(|| {
            let result = poisson.exp_fam_log_density(black_box(&k));
            black_box(result)
        });
    });

    // Test batch processing allocation patterns
    let points: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();

    group.bench_function("batch_clone_per_evaluation", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &points {
                // Clone on every evaluation (worst case)
                let ld = normal.log_density();
                sum += ld.at(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.bench_function("batch_single_clone", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            let ld = normal.log_density(); // Clone once
            for &x in &points {
                sum += ld.at(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.bench_function("batch_no_wrapper", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &points {
                // No LogDensity wrapper at all
                sum += normal.log_density_wrt_root(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Benchmark exponential family component creation (cloning hotspots)
fn bench_component_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("component_creation");

    let normal = Normal::new(1.0_f64, 2.0_f64);
    let poisson = Poisson::new(2.5_f64);
    let x = 0.5_f64;
    let k = 5u64;

    // Normal component creation
    group.bench_function("normal_natural_params_creation", |b| {
        b.iter(|| {
            let result = normal.to_natural();
            black_box(result)
        });
    });

    group.bench_function("normal_base_measure_creation", |b| {
        b.iter(|| {
            let result = normal.base_measure();
            black_box(result)
        });
    });

    group.bench_function("normal_sufficient_stats_creation", |b| {
        b.iter(|| {
            let result = normal.sufficient_statistic(black_box(&x));
            black_box(result)
        });
    });

    // Poisson component creation (includes FactorialMeasure)
    group.bench_function("poisson_base_measure_creation", |b| {
        b.iter(|| {
            let result = poisson.base_measure();
            black_box(result)
        });
    });

    group.bench_function("factorial_measure_log_density", |b| {
        b.iter(|| {
            let factorial_measure = poisson.base_measure();
            let result = factorial_measure.log_density_wrt_root(black_box(&k));
            black_box(result)
        });
    });

    // Compare component-wise vs all-at-once computation
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

/// Benchmark O(k) vs O(1) factorial computation approaches
fn bench_factorial_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial_optimization");

    let poisson = Poisson::new(2.5_f64);

    // Test performance across different k values
    for k in &[1, 5, 10, 20, 50, 100, 200, 500] {
        group.bench_with_input(BenchmarkId::new("o1_stirling_factorial", k), k, |b, &k| {
            b.iter(|| {
                // New O(1) approach using Stirling's approximation
                let result = poisson.exp_fam_log_density(black_box(&k));
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("ok_loop_factorial", k), k, |b, &k| {
            b.iter(|| {
                // Old O(k) approach using loop (for comparison)
                let mut log_factorial = 0.0_f64;
                for i in 1..=black_box(k) {
                    log_factorial += (i as f64).ln();
                }
                let result =
                    poisson.to_natural()[0] * (k as f64) - poisson.log_partition() - log_factorial;
                black_box(result)
            });
        });
    }

    // Specific accuracy test for the crossover point
    group.bench_function("stirling_accuracy_k12", |b| {
        b.iter(|| {
            // Test accuracy at the crossover point (k=12)
            let k = 12u64;
            let result = poisson.exp_fam_log_density(black_box(&k));
            black_box(result)
        });
    });

    // Test very large k values where O(1) really shines
    group.bench_function("stirling_large_k1000", |b| {
        b.iter(|| {
            let k = 1000u64;
            let result = poisson.exp_fam_log_density(black_box(&k));
            black_box(result)
        });
    });

    group.bench_function("stirling_very_large_k10000", |b| {
        b.iter(|| {
            let k = 10000u64;
            let result = poisson.exp_fam_log_density(black_box(&k));
            black_box(result)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_single_evaluations, bench_batch_evaluations, bench_poisson_factorial,
              bench_exp_fam_components, bench_measure_creation, bench_allocation_patterns, bench_component_creation, bench_factorial_optimization
}

criterion_main!(benches);
