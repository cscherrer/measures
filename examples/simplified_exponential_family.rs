use measures::exponential_family::{compute_exp_fam_log_density, compute_iid_exp_fam_log_density};
use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== Simplified Exponential Family Computation ===\n");

    let normal = Normal::new(1.0, 2.0);
    let x = 0.5;

    println!("Normal distribution: μ=1.0, σ=2.0");
    println!("Single point: x={x}\n");

    // Method 1: Direct exponential family computation
    println!("=== Single Point Computation ===");
    let direct_log_density: f64 = compute_exp_fam_log_density(&normal, &x);
    println!("Central function result: {direct_log_density:.6}");

    // Verify with standard method
    let standard_log_density: f64 = normal.log_density().at(&x);
    println!("Standard method result: {standard_log_density:.6}");
    println!(
        "✓ Results match: {}",
        (direct_log_density - standard_log_density).abs() < 1e-10
    );

    // Method 2: IID computation
    println!("\n=== IID Computation ===");
    let samples = vec![0.5, 1.2, 0.8, 1.5, 0.9];
    println!("Samples: {samples:?}");

    // Using central IID function
    let iid_log_density: f64 = compute_iid_exp_fam_log_density(&normal, &samples);
    println!("Central IID function: {iid_log_density:.6}");

    // Using IID wrapper
    let iid_normal = normal.clone().iid();
    let wrapper_log_density: f64 = iid_normal.iid_log_density(&samples);
    println!("IID wrapper result: {wrapper_log_density:.6}");

    // Verify with manual summation
    let manual_sum: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();
    println!("Manual summation: {manual_sum:.6}");

    println!(
        "✓ All methods match: {}",
        (iid_log_density - manual_sum).abs() < 1e-10
            && (wrapper_log_density - manual_sum).abs() < 1e-10
    );

    println!("\n=== API Consistency ===");
    println!("Single point: normal.log_density().at(&x)");
    println!("IID samples:  iid_normal.iid_log_density(&samples)");
    println!("Or directly:  compute_iid_exp_fam_log_density(&normal, &samples)");

    println!("\n=== Summary ===");
    println!("✓ Single source of computation for exponential families");
    println!("✓ Centralized functions reduce code duplication");
    println!("✓ Efficient IID computation using exponential family structure");
    println!("✓ Consistent results across all methods");
    println!("✓ Clear API separation: log_density() vs iid_log_density()");
}
