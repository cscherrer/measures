//! Exponential Family vs Non-Exponential Family Example
//!
//! Demonstrates that the trait system correctly handles both exponential family
//! distributions (Normal) and non-exponential family distributions (Cauchy).

use measures::{Cauchy, LogDensityBuilder, Normal};

fn main() {
    let normal = Normal::new(0.0, 1.0); // Exponential family
    let cauchy = Cauchy::new(0.0, 1.0); // NOT exponential family
    let test_point = 0.5;

    // Basic log-density computation
    let normal_log_density: f64 = normal.log_density().at(&test_point);
    let cauchy_log_density: f64 = cauchy.log_density().at(&test_point);

    println!("Log-densities at x={test_point}:");
    println!("  Normal: {normal_log_density:.6}");
    println!("  Cauchy: {cauchy_log_density:.6}");

    // Relative density computation
    let normal_wrt_cauchy: f64 = normal.log_density().wrt(cauchy.clone()).at(&test_point);
    let cauchy_wrt_normal: f64 = cauchy.log_density().wrt(normal.clone()).at(&test_point);

    println!("Relative densities:");
    println!("  log(Normal/Cauchy): {normal_wrt_cauchy:.6}");
    println!("  log(Cauchy/Normal): {cauchy_wrt_normal:.6}");
    println!(
        "  Sum (should be ~0): {:.10}",
        normal_wrt_cauchy + cauchy_wrt_normal
    );

    // Type-level dispatch verification
    use measures::core::{MeasureMarker, types::TypeLevelBool};

    let normal_is_exp_fam = <Normal<f64> as MeasureMarker>::IsExponentialFamily::VALUE;
    let cauchy_is_exp_fam = <Cauchy<f64> as MeasureMarker>::IsExponentialFamily::VALUE;

    println!("Type-level markers:");
    println!("  Normal IsExponentialFamily: {normal_is_exp_fam}");
    println!("  Cauchy IsExponentialFamily: {cauchy_is_exp_fam}");
}
