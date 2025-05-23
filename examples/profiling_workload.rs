//! Profiling workload example
//!
//! This example simulates real-world usage patterns for performance analysis.
//!
//! Run with: cargo run --example `profiling_workload` --profile profiling

use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::exponential_family::ExponentialFamily;
use measures::{Normal, distributions::discrete::poisson::Poisson};

fn main() {
    println!("🎯 Profiling Workload Simulation");
    println!("=================================");

    // Monte Carlo simulation
    monte_carlo_simulation();

    // ML optimization simulation
    ml_optimization_simulation();

    // Statistical inference simulation
    statistical_inference_simulation();

    // Poisson factorial stress test
    poisson_factorial_stress_test();

    println!("✅ Profiling workload complete");
}

/// Simulate Monte Carlo sampling workload
fn monte_carlo_simulation() {
    println!("\n🎰 Monte Carlo Simulation (100k samples)");

    let normal = Normal::new(0.0_f64, 1.0_f64);
    let ld = normal.log_density();

    let mut sum = 0.0;
    for i in 0..100_000 {
        let x = f64::from(i) * 0.0001 - 5.0; // Range from -5 to 5
        sum += ld.at(&x);
    }

    println!("  📊 Total log-density sum: {sum:.3}");
}

/// Simulate ML optimization workload
fn ml_optimization_simulation() {
    println!("\n🎯 ML Optimization Simulation");

    // Multiple distributions for parameter optimization
    let distributions = [
        Normal::new(0.0, 1.0),
        Normal::new(0.5, 1.5),
        Normal::new(-0.5, 0.8),
        Normal::new(1.0, 2.0),
        Normal::new(-1.0, 0.5),
    ];

    // Simulated data points
    let data_points: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01 - 5.0).collect();

    let mut best_likelihood = f64::NEG_INFINITY;
    let mut best_dist_idx = 0;

    for (dist_idx, dist) in distributions.iter().enumerate() {
        let mut likelihood = 0.0;
        for &x in &data_points {
            likelihood += dist.log_density_wrt_root(&x);
        }

        if likelihood > best_likelihood {
            best_likelihood = likelihood;
            best_dist_idx = dist_idx;
        }
    }

    println!("  📈 Best distribution: {best_dist_idx} with likelihood {best_likelihood:.3}");
}

/// Simulate statistical inference workload
fn statistical_inference_simulation() {
    println!("\n📈 Statistical Inference Simulation");

    let normal1 = Normal::new(0.0_f64, 1.0_f64);
    let normal2 = Normal::new(0.1_f64, 1.1_f64);
    let ld_relative = normal1.log_density().wrt(normal2);

    let mut evidence_sum = 0.0;
    for i in 0..10_000 {
        let x = f64::from(i) * 0.001 - 5.0;
        evidence_sum += ld_relative.at(&x);
    }

    println!("  📊 Evidence sum: {evidence_sum:.3}");
}

/// Stress test Poisson factorial computation
fn poisson_factorial_stress_test() {
    println!("\n🧮 Poisson Factorial Stress Test");

    let poisson = Poisson::new(10.0_f64);

    let mut total = 0.0;
    for k in 0..100 {
        for _ in 0..100 {
            total += poisson.exp_fam_log_density(&k);
        }
    }

    println!("  📊 Total Poisson log-density: {total:.3}");
}
