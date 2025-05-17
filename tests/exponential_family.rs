use measures::{HasDensity, Normal, exponential_family::ExpFam};

#[test]
fn test_normal_exp_fam_wrapper() {
    // Create a standard normal distribution
    let normal = Normal::new(0.0, 1.0);

    // Wrap it in the exponential family combinator
    let exp_fam_normal = ExpFam::new(normal.clone());

    // Test points
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        // Get log-density directly
        let direct_log_density: f64 = normal.log_density(&x).into();

        // Get log-density through the wrapper
        let wrapped_log_density: f64 = exp_fam_normal.log_density(&x).into();

        // They should be very close
        assert!(
            (direct_log_density - wrapped_log_density).abs() < 1e-10,
            "Log density mismatch at x={x}: direct={direct_log_density}, wrapped={wrapped_log_density}"
        );
    }
}

#[test]
fn test_specialized_exp_fam_computation() {
    // Create a normal distribution with non-standard parameters
    let normal = Normal::new(1.5, 2.0);

    // Get log-density directly
    let x = 0.5;
    let direct_log_density: f64 = normal.log_density(&x).into();

    // Compute using the specialized exponential family form
    let specialized_log_density = normal.log_density(&x).compute_exp_fam_form();

    // They should be very close
    assert!(
        (direct_log_density - specialized_log_density).abs() < 1e-10,
        "Log density mismatch: direct={direct_log_density}, specialized={specialized_log_density}"
    );
}
