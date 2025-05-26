use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::ExponentialFamily;
use measures::{HasLogDensity, LogDensityBuilder, Normal};
use rand::{Rng, rng};
use rv::dist::{Gaussian, Poisson as RvPoisson};
use rv::prelude::*;
use std::hint::black_box;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Macro for compile-time optimization when parameters are known at compile time
macro_rules! optimized_normal {
    ($mu:expr, $sigma:expr) => {{
        let mu = $mu;
        let sigma = $sigma;
        let sigma_sq = sigma * sigma;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

        move |x: f64| -> f64 {
            let diff = x - mu;
            log_norm_constant - diff * diff * inv_two_sigma_sq
        }
    }};
}

/// Zero-overhead runtime code generation for Normal distribution
fn generate_zero_overhead_normal(mu: f64, sigma: f64) -> impl Fn(f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

    move |x: f64| -> f64 {
        let diff = x - mu;
        log_norm_constant - diff * diff * inv_two_sigma_sq
    }
}

/// Runtime specialization with const generics
pub struct SpecializedNormal<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32>;

impl<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32> Default
    for SpecializedNormal<MU_TIMES_1000, SIGMA_TIMES_1000>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MU_TIMES_1000: i32, const SIGMA_TIMES_1000: i32>
    SpecializedNormal<MU_TIMES_1000, SIGMA_TIMES_1000>
{
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    #[inline(always)]
    #[must_use]
    pub fn log_density(&self, x: f64) -> f64 {
        let mu = f64::from(MU_TIMES_1000) / 1000.0;
        let sigma = f64::from(SIGMA_TIMES_1000) / 1000.0;
        let sigma_sq = sigma * sigma;
        let log_norm_constant = -0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln();
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma_sq);

        let diff = x - mu;
        log_norm_constant - diff * diff * inv_two_sigma_sq
    }
}

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
/// This replaces both `bench_poisson_factorial` and `bench_factorial_optimization`
fn bench_factorial_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial_optimization");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

    // Test key k values that demonstrate O(1) vs O(k) scaling
    for k in &[1, 10, 50, 200, 1000] {
        // Create fresh measures for each k to prevent optimization (just like original)
        let poisson = black_box(Poisson::new(2.5_f64 + (*k as f64) * 0.001));
        let rv_poisson = black_box(RvPoisson::new(2.5 + (*k as f64) * 0.001).unwrap());

        // Our O(1) optimized approach
        group.bench_with_input(BenchmarkId::new("measures_o1_stirling", k), k, |b, &k| {
            b.iter(|| {
                let result = black_box(&poisson).exp_fam_log_density(black_box(&k));
                black_box(result)
            });
        });

        // rv reference implementation
        group.bench_with_input(BenchmarkId::new("rv_poisson_reference", k), k, |b, &k| {
            b.iter(|| {
                let result = black_box(&rv_poisson).ln_f(black_box(&(k as usize)));
                black_box(result)
            });
        });

        // O(k) naive approach (only for smaller k to avoid excessive runtime)
        if k <= &100 {
            group.bench_with_input(BenchmarkId::new("naive_ok_factorial", k), k, |b, &k| {
                b.iter(|| {
                    let mut log_factorial = 0.0_f64;
                    for i in 1..=black_box(k) {
                        log_factorial += (i as f64).ln();
                    }
                    let lambda = 2.5_f64 + (k as f64) * 0.001;
                    let result = (k as f64) * lambda.ln() - lambda - log_factorial;
                    black_box(result)
                });
            });
        }
    }

    group.finish();
}

/// Consolidated component benchmark (combines `exp_fam_components` and `component_creation`)
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

/// Benchmark Normal distribution optimization techniques
fn bench_normal_optimization_techniques(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_optimization_techniques");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Generate random test values to prevent compiler optimization
    let mut rng = rng();
    let test_values: Vec<f64> = (0..1000).map(|_| rng.random_range(-3.0..3.0)).collect();

    let normal = Normal::new(0.0_f64, 1.0_f64);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();

    // Standard exponential family evaluation
    group.bench_function("standard_exp_fam", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += black_box(&normal).exp_fam_log_density(black_box(&x));
            }
            black_box(sum)
        });
    });

    // Standard log density builder
    group.bench_function("standard_log_density", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                let ld = black_box(&normal).log_density();
                sum += ld.at(black_box(&x));
            }
            black_box(sum)
        });
    });

    // Zero-overhead runtime code generation
    let zero_overhead_fn = generate_zero_overhead_normal(0.0, 1.0);
    group.bench_function("zero_overhead_runtime", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += black_box(&zero_overhead_fn)(black_box(x));
            }
            black_box(sum)
        });
    });

    // Compile-time macro optimization
    let macro_fn = optimized_normal!(0.0, 1.0);
    group.bench_function("compile_time_macro", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += black_box(&macro_fn)(black_box(x));
            }
            black_box(sum)
        });
    });

    // Const generic specialization
    let specialized: SpecializedNormal<0, 1000> = SpecializedNormal::new();
    group.bench_function("const_generic_specialized", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += black_box(&specialized).log_density(black_box(x));
            }
            black_box(sum)
        });
    });

    // Reference implementation (rv crate)
    group.bench_function("rv_reference", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &test_values {
                sum += black_box(&rv_normal).ln_f(black_box(&x));
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_single_evaluations, bench_batch_evaluations, bench_factorial_optimization,
              bench_exponential_family_components, bench_allocation_patterns, bench_measure_creation,
              bench_normal_optimization_techniques
}

criterion_main!(benches);
