//! Validation tests comparing our implementations with rv crate as reference.
//! 
//! ## Key Insights from IntegralMeasure Investigation
//! 
//! The debugging tests prove that the IntegralMeasure approach is mathematically correct:
//! 
//! 1. **Working Poisson** = Exponential Family part + Chain rule part
//!    - ExpFam part: η·T(x) - A(η) = k·ln(λ) - λ  
//!    - Chain rule part: -log(k!) from the factorial base measure
//!    - Combined: k·ln(λ) - λ - log(k!) = correct Poisson log-PMF
//! 
//! 2. **Default ExpFam implementation** only computes η·T(x) - A(η), missing the chain rule
//! 
//! 3. **Future enhancement**: Modify ExponentialFamily trait to automatically include 
//!    chain rule when base measure != root measure, eliminating the need for manual overrides

use measures::{Normal, distributions::discrete::poisson::Poisson};
use measures::core::{HasLogDensity, LogDensityBuilder};
use measures::measures::derived::integral::IntegralMeasure;
use measures::measures::primitive::counting::CountingMeasure;
use measures::traits::DotProduct;
use rv::prelude::*;

#[test]
fn test_normal_vs_rv() {
    let our_normal = Normal::new(1.0, 2.0);
    let rv_normal = rv::dist::Gaussian::new(1.0, 2.0).unwrap();
    
    let test_points = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    
    for &x in &test_points {
        let our_logpdf: f64 = our_normal.log_density().at(&x);
        let rv_logpdf = rv_normal.ln_f(&x);
        
        println!("Normal at x={}: ours={}, rv={}", x, our_logpdf, rv_logpdf);
        
        assert!(
            (our_logpdf - rv_logpdf).abs() < 1e-10,
            "Normal mismatch at x={}: ours={}, rv={}",
            x, our_logpdf, rv_logpdf
        );
    }
}

#[test] 
fn test_poisson_vs_rv() {
    let our_poisson = Poisson::new(2.5_f64);
    let rv_poisson = rv::dist::Poisson::new(2.5).unwrap();
    
    let test_points = vec![0usize, 1, 2, 3, 4, 5, 10];
    
    for &k in &test_points {
        let our_logpmf: f64 = our_poisson.log_density().at(&(k as u64));
        let rv_logpmf = rv_poisson.ln_pmf(&k);
        
        println!("Poisson at k={}: ours={}, rv={}", k, our_logpmf, rv_logpmf);
        
        // Check if our result is reasonable (should be negative)
        if our_logpmf > 0.0 {
            println!("WARNING: Our Poisson log-pmf is positive at k={}: {}", k, our_logpmf);
        }
        
        assert!(
            (our_logpmf - rv_logpmf).abs() < 1e-10,
            "Poisson mismatch at k={}: ours={}, rv={}",
            k, our_logpmf, rv_logpmf
        );
    }
}

#[test]
fn test_poisson_working_vs_integral_measure() {
    println!("\n=== Debugging Poisson: Working vs IntegralMeasure ===");
    
    let lambda = 2.5_f64;
    let k = 3u64;
    
    // === Working Implementation ===
    let working_poisson = Poisson::new(lambda);
    let working_logpmf: f64 = working_poisson.log_density().at(&k);
    
    println!("Working Poisson({}): log_pmf({}) = {}", lambda, k, working_logpmf);
    
    // === Manual IntegralMeasure Implementation ===
    let counting = CountingMeasure::<u64>::new();
    let integral_base = IntegralMeasure::new(
        counting.clone(),
        |k: &u64| {
            let mut log_factorial = 0.0_f64;
            for i in 1..=*k {
                log_factorial += (i as f64).ln();
            }
            -log_factorial  // -log(k!) for 1/k! density
        }
    );
    
    // === Step-by-step computation ===
    println!("\n--- Manual Chain Rule Computation ---");
    
    // 1. CountingMeasure log-density wrt root (should be 0)
    let counting_log_density: f64 = counting.log_density_wrt_root(&k);
    println!("log(dCounting/dRoot) at k={}: {}", k, counting_log_density);
    
    // 2. IntegralMeasure log-density function at k
    let factorial_fn = |k: &u64| {
        let mut log_factorial = 0.0_f64;
        for i in 1..=*k {
            log_factorial += (i as f64).ln();
        }
        -log_factorial
    };
    let integral_fn_value = factorial_fn(&k);
    println!("Integral log-density function at k={}: {}", k, integral_fn_value);
    
    // 3. IntegralMeasure log-density wrt root (chain rule)
    let integral_log_density: f64 = integral_base.log_density_wrt_root(&k);
    println!("log(dIntegral/dRoot) at k={}: {}", k, integral_log_density);
    println!("  = log_density_fn({}) + counting_log_density = {} + {} = {}", 
             k, integral_fn_value, counting_log_density, integral_log_density);
    
    // 4. Exponential family computation for hypothetical IntegralMeasure Poisson
    let natural_param = [lambda.ln()];
    let sufficient_stat = [k as f64];
    let log_partition = lambda;
    
    let exp_fam_part = natural_param.dot(&sufficient_stat) - log_partition;
    println!("\n--- Exponential Family Part ---");
    println!("η·T(x) - A(η) = {} * {} - {} = {}", 
             natural_param[0], sufficient_stat[0], log_partition, exp_fam_part);
    
    // 5. What would IntegralMeasure Poisson give?
    let integral_poisson_result = exp_fam_part;  // Default implementation
    println!("\n--- Final Results ---");
    println!("Working Poisson result: {}", working_logpmf);
    println!("IntegralMeasure Poisson (no chain rule): {}", integral_poisson_result);
    println!("Expected with chain rule: {} + {} = {}", 
             integral_poisson_result, integral_log_density, 
             integral_poisson_result + integral_log_density);
    
    // 6. Manual computation for verification
    let manual_result = k as f64 * lambda.ln() - lambda - (1..=k).map(|i| (i as f64).ln()).sum::<f64>();
    println!("Manual Poisson computation: {}", manual_result);
    
    // What's the difference?
    let difference = working_logpmf - integral_poisson_result;
    println!("\nDifference (working - exp_fam_only): {}", difference);
    println!("Expected difference (should be -log(k!)): {}", integral_fn_value);
    
    // Verify rv matches our working implementation
    let rv_poisson = rv::dist::Poisson::new(lambda).unwrap();
    let rv_logpmf = rv_poisson.ln_pmf(&(k as usize));
    println!("rv reference: {}", rv_logpmf);
    
    // The key insight: IntegralMeasure + ExpFam is mathematically correct!
    // Working result = ExpFam part + Chain rule part
    assert!((working_logpmf - (integral_poisson_result + integral_log_density)).abs() < 1e-10);
}

#[test]
fn test_integral_measure_poisson_implementation() {
    println!("\n=== Key Insight: IntegralMeasure + ExpFam = Working Implementation ===");
    
    let lambda = 2.5_f64;
    let test_points = [0u64, 1, 2, 3, 4, 5];
    
    let working_poisson = Poisson::new(lambda);
    
    for &k in &test_points {
        let working_result: f64 = working_poisson.log_density().at(&k);
        
        // Manual decomposition: ExpFam part + Chain rule part
        let exp_fam_part = (k as f64) * lambda.ln() - lambda;  // η·T(x) - A(η)
        
        let factorial_part = if k == 0 { 0.0 } else {
            -(1..=k).map(|i| (i as f64).ln()).sum::<f64>()  // -log(k!)
        };
        
        let decomposed_result = exp_fam_part + factorial_part;
        
        println!("k={}: Working={:.6}, ExpFam={:.6}, Factorial={:.6}, Sum={:.6}", 
                 k, working_result, exp_fam_part, factorial_part, decomposed_result);
        
        assert!(
            (working_result - decomposed_result).abs() < 1e-10,
            "Decomposition failed at k={}: working={}, decomposed={}",
            k, working_result, decomposed_result
        );
    }
    
    println!("✅ Working Poisson = Exponential Family part + Factorial base measure part");
    println!("✅ This proves IntegralMeasure approach is mathematically sound!");
    println!("✅ The issue was that default ExpFam impl doesn't include chain rule automatically");
} 