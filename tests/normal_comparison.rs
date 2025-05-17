use measures::{HasDensity, Normal};
use rv::dist::Gaussian;
use rv::traits::ContinuousDistr;

#[test]
fn test_density_comparison() {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();

    // Test points around the mean
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        // Get density as a Density object first, then convert to f64
        let density = our_normal.density(&x);
        let our_density: f64 = density.into();
        let rv_density = rv_normal.pdf(&x);

        // Compare densities (should be very close)
        assert!(
            (our_density - rv_density).abs() < 1e-10,
            "Density mismatch at x={x}: our={our_density}, rv={rv_density}"
        );
    }
}

#[test]
fn test_log_density_comparison() {
    let our_normal = Normal::new(0.0, 1.0);
    let rv_normal = Gaussian::new(0.0, 1.0).unwrap();

    // Test points around the mean
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        // Get log-density as a LogDensity object first, then convert to f64
        let log_density = our_normal.log_density(&x);
        let our_log_density: f64 = log_density.into();
        let rv_log_density = rv_normal.ln_pdf(&x);

        // Compare log densities (should be very close)
        assert!(
            (our_log_density - rv_log_density).abs() < 1e-10,
            "Log density mismatch at x={x}: our={our_log_density}, rv={rv_log_density}"
        );
    }
}

#[test]
fn test_different_parameters() {
    let test_params = [
        (0.0, 1.0),  // Standard normal
        (1.0, 1.0),  // Shifted normal
        (0.0, 2.0),  // Wider normal
        (-1.0, 0.5), // Shifted and narrower normal
    ];

    for (mu, sigma) in test_params {
        let our_normal = Normal::new(mu, sigma);
        let rv_normal = Gaussian::new(mu, sigma).unwrap();

        // Test points around the mean
        let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

        for x in test_points {
            // Get density as a Density object first, then convert to f64
            let density = our_normal.density(&x);
            let our_density: f64 = density.into();
            let rv_density = rv_normal.pdf(&x);

            assert!(
                (our_density - rv_density).abs() < 1e-10,
                "Density mismatch at x={x} for μ={mu}, σ={sigma}: our={our_density}, rv={rv_density}"
            );

            // Get log-density as a LogDensity object first, then convert to f64
            let log_density = our_normal.log_density(&x);
            let our_log_density: f64 = log_density.into();
            let rv_log_density = rv_normal.ln_pdf(&x);

            assert!(
                (our_log_density - rv_log_density).abs() < 1e-10,
                "Log density mismatch at x={x} for μ={mu}, σ={sigma}: our={our_log_density}, rv={rv_log_density}"
            );
        }
    }
}
