//! Simplified Exponential Family Computation
//!
//! Demonstrates the central exponential family computation functions and
//! compares them with the standard API methods.

use measures::exponential_family::{compute_exp_fam_log_density, compute_iid_exp_fam_log_density};
use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    let normal = Normal::new(1.0, 2.0);
    let x = 0.5;
    let samples = vec![0.5, 1.2, 0.8, 1.5, 0.9];

    // Single point computation
    let direct_log_density: f64 = compute_exp_fam_log_density(&normal, &x);
    let standard_log_density: f64 = normal.log_density().at(&x);
    
    println!("Single point (x={}): direct={:.6}, standard={:.6}", 
             x, direct_log_density, standard_log_density);
    assert!((direct_log_density - standard_log_density).abs() < 1e-10);

    // IID computation
    let iid_log_density: f64 = compute_iid_exp_fam_log_density(&normal, &samples);
    let iid_normal = normal.clone().iid();
    let wrapper_log_density: f64 = iid_normal.iid_log_density(&samples);
    let standard_api_result: f64 = iid_normal.log_density().at(&samples);
    let manual_sum: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();

    println!("IID computation:");
    println!("  Central function: {:.6}", iid_log_density);
    println!("  IID wrapper:      {:.6}", wrapper_log_density);
    println!("  Standard API:     {:.6}", standard_api_result);
    println!("  Manual sum:       {:.6}", manual_sum);

    // Verify all methods produce identical results
    assert!((iid_log_density - manual_sum).abs() < 1e-10);
    assert!((wrapper_log_density - manual_sum).abs() < 1e-10);
    assert!((standard_api_result - manual_sum).abs() < 1e-10);
    
    println!("✓ All computation methods verified");
}
