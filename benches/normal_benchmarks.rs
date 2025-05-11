use criterion::{Criterion, black_box, criterion_group, criterion_main};
use measures::{HasDensity, Normal};
use rv::dist::Gaussian;
use rv::traits::ContinuousDistr;

fn bench_density(c: &mut Criterion) {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();

    let mut group = c.benchmark_group("density");

    group.bench_function("our_normal", |b| {
        b.iter(|| {
            let density: f64 = our_normal.density(&black_box(0.0)).into();
            black_box(density)
        });
    });

    group.bench_function("rv_normal", |b| {
        b.iter(|| black_box(rv_normal.pdf(&black_box(0.0))));
    });

    group.finish();
}

fn bench_log_density(c: &mut Criterion) {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();

    let mut group = c.benchmark_group("log_density");

    group.bench_function("our_normal", |b| {
        b.iter(|| {
            let log_density: f64 = our_normal.log_density(&black_box(0.0)).into();
            black_box(log_density)
        });
    });

    group.bench_function("rv_normal", |b| {
        b.iter(|| black_box(rv_normal.ln_pdf(&black_box(0.0))));
    });

    group.finish();
}

fn bench_density_batch(c: &mut Criterion) {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();
    let xs: Vec<f64> = (-10..=10).map(|x| f64::from(x) / 10.0).collect();

    let mut group = c.benchmark_group("density_batch");

    group.bench_function("our_normal", |b| {
        b.iter(|| {
            for &x in &xs {
                let density: f64 = our_normal.density(&black_box(x)).into();
                black_box(density);
            }
        });
    });

    group.bench_function("rv_normal", |b| {
        b.iter(|| {
            for &x in &xs {
                black_box(rv_normal.pdf(&black_box(x)));
            }
        });
    });

    group.finish();
}

fn bench_log_density_batch(c: &mut Criterion) {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();
    let xs: Vec<f64> = (-10..=10).map(|x| f64::from(x) / 10.0).collect();

    let mut group = c.benchmark_group("log_density_batch");

    group.bench_function("our_normal", |b| {
        b.iter(|| {
            for &x in &xs {
                let log_density: f64 = our_normal.log_density(&black_box(x)).into();
                black_box(log_density);
            }
        });
    });

    group.bench_function("rv_normal", |b| {
        b.iter(|| {
            for &x in &xs {
                black_box(rv_normal.ln_pdf(&black_box(x)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_density,
    bench_log_density,
    bench_density_batch,
    bench_log_density_batch
);
criterion_main!(benches);
