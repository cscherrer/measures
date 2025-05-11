use measures::{HasDensity, LebesgueMeasure, Normal};
use rv::dist::Gaussian;
use rv::traits::ContinuousDistr;

#[test]
fn test_density_comparison() {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();
    let lebesgue = LebesgueMeasure::new();

    // Test points around the mean
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let our_density: f64 = our_normal.density(&x).wrt(&lebesgue).into();
        let rv_density = rv_normal.pdf(&x);
        
        // Compare densities (should be very close)
        assert!((our_density - rv_density).abs() < 1e-10, 
            "Density mismatch at x={}: our={}, rv={}", x, our_density, rv_density);
    }
}

#[test]
fn test_log_density_comparison() {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();
    let lebesgue = LebesgueMeasure::new();

    // Test points around the mean
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let our_log_density: f64 = our_normal.log_density(&x).wrt(&lebesgue).into();
        let rv_log_density = rv_normal.ln_pdf(&x);
        
        // Compare log densities (should be very close)
        assert!((our_log_density - rv_log_density).abs() < 1e-10,
            "Log density mismatch at x={}: our={}, rv={}", x, our_log_density, rv_log_density);
    }
}

#[test]
fn test_different_parameters() {
    let test_params = [
        (0.0, 1.0),   // Standard normal
        (1.0, 1.0),   // Shifted normal
        (0.0, 2.0),   // Wider normal
        (-1.0, 0.5),  // Shifted and narrower normal
    ];

    for (mu, sigma) in test_params {
        let our_normal = Normal::new(mu, sigma);
        let rv_normal = Gaussian::new(mu, sigma).unwrap();
        let lebesgue = LebesgueMeasure::new();

        // Test points around the mean
        let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

        for x in test_points {
            let our_density: f64 = our_normal.density(&x).wrt(&lebesgue).into();
            let rv_density = rv_normal.pdf(&x);
            
            assert!((our_density - rv_density).abs() < 1e-10,
                "Density mismatch at x={} for μ={}, σ={}: our={}, rv={}", 
                x, mu, sigma, our_density, rv_density);

            let our_log_density: f64 = our_normal.log_density(&x).wrt(&lebesgue).into();
            let rv_log_density = rv_normal.ln_pdf(&x);
            
            assert!((our_log_density - rv_log_density).abs() < 1e-10,
                "Log density mismatch at x={} for μ={}, σ={}: our={}, rv={}", 
                x, mu, sigma, our_log_density, rv_log_density);
        }
    }
} 