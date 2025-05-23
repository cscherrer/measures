use measures::Normal;
use measures::core::HasLogDensity;
use measures::distributions::discrete::poisson::Poisson;
use measures::exponential_family::ExponentialFamily;
use std::hint::black_box;

// Standalone functions for easy assembly analysis

#[inline(never)]
#[must_use]
pub fn measures_poisson_optimized(poisson: &Poisson<f64>, k: &u64) -> f64 {
    poisson.exp_fam_log_density(k)
}

#[inline(never)]
#[must_use]
pub fn measures_poisson_direct(poisson: &Poisson<f64>, k: &u64) -> f64 {
    poisson.log_density_wrt_root(k)
}

#[inline(never)]
#[must_use]
pub fn measures_normal_optimized(normal: &Normal<f64>, x: &f64) -> f64 {
    normal.exp_fam_log_density(x)
}

#[inline(never)]
#[must_use]
pub fn measures_normal_direct(normal: &Normal<f64>, x: &f64) -> f64 {
    normal.log_density_wrt_root(x)
}

fn main() {
    let poisson = Poisson::new(2.5_f64);
    let normal = Normal::new(0.0_f64, 1.0_f64);
    let k = 10u64;
    let x = 0.5_f64;

    // Force evaluation to prevent optimization
    let result1 = black_box(measures_poisson_optimized(
        black_box(&poisson),
        black_box(&k),
    ));
    let result2 = black_box(measures_poisson_direct(black_box(&poisson), black_box(&k)));
    let result3 = black_box(measures_normal_optimized(black_box(&normal), black_box(&x)));
    let result4 = black_box(measures_normal_direct(black_box(&normal), black_box(&x)));

    println!("Poisson optimized: {result1}");
    println!("Poisson direct: {result2}");
    println!("Normal optimized: {result3}");
    println!("Normal direct: {result4}");
}
