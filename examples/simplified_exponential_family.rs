use measures::exponential_family::{compute_exp_fam_log_density, compute_iid_exp_fam_log_density};
use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== Simplified Exponential Family Computation ===\n");

    let normal = Normal::new(1.0, 2.0);
    let x = 0.5;
    
    println!("Normal distribution: μ=1.0, σ=2.0");
    println!("Single point: x={}\n", x);

    // Method 1: Direct exponential family computation
    println!("=== Single Point Computation ===");
    let direct_log_density: f64 = compute_exp_fam_log_density(&normal, &x);
    println!("Central function result: {:.6}", direct_log_density);

    // Verify with standard method
    let standard_log_density: f64 = normal.log_density().at(&x);
    println!("Standard method result: {:.6}", standard_log_density);
    println!("✓ Results match: {}", (direct_log_density - standard_log_density).abs() < 1e-10);

    // Method 2: IID computation
    println!("\n=== IID Computation ===");
    let samples = vec![0.5, 1.2, 0.8, 1.5, 0.9];
    println!("Samples: {:?}", samples);

    // Using central IID function
    let iid_log_density: f64 = compute_iid_exp_fam_log_density(&normal, &samples);
    println!("Central IID function: {:.6}", iid_log_density);

    // Using IID wrapper
    let iid_normal = normal.clone().iid();
    let wrapper_log_density: f64 = iid_normal.log_density(&samples);
    println!("IID wrapper result: {:.6}", wrapper_log_density);

    // Verify with manual summation
    let manual_sum: f64 = samples
        .iter()
        .map(|&x| normal.log_density().at(&x))
        .sum();
    println!("Manual summation: {:.6}", manual_sum);

    println!("✓ All methods match: {}", 
        (iid_log_density - manual_sum).abs() < 1e-10 &&
        (wrapper_log_density - manual_sum).abs() < 1e-10
    );

    println!("\n=== Summary ===");
    println!("✓ Single source of computation for exponential families");
    println!("✓ Centralized functions reduce code duplication");
    println!("✓ Efficient IID computation using exponential family structure");
    println!("✓ Consistent results across all methods");
} 