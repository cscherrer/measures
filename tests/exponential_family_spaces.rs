use measures::{
    HasDensity,
    distributions::{
        discrete::poisson::Poisson,
        multivariate::multinormal::{Matrix, MultivariateNormal, Vector},
    },
    exponential_family::{ExpFam, ExponentialFamilyMeasure, InnerProduct},
};

#[test]
fn test_poisson_with_different_field_types() {
    // Create Poisson distributions with different field types
    let poisson_f32 = Poisson::<f32>::new(3.0);
    let poisson_f64 = Poisson::<f64>::new(3.0);

    // They should both work with the same value space (u64)
    let x = 2u64;

    let log_density_f32: f64 = poisson_f32.log_density(&x).into();
    let log_density_f64: f64 = poisson_f64.log_density(&x).into();

    // Results should be very close
    assert!((log_density_f32 - log_density_f64).abs() < 1e-5);

    // We can also wrap them in ExpFam for specialized computation
    let exp_fam_poisson = ExpFam::new(poisson_f64);
    let specialized_log_density: f64 = exp_fam_poisson.log_density(&x).into();

    assert!((specialized_log_density - log_density_f64).abs() < 1e-10);
}

#[test]
fn test_multivariate_normal_vector_space() {
    // Create a 2D multivariate normal distribution
    let mean = Vector::new(vec![1.0, 2.0]);
    let cov_data = vec![vec![1.0, 0.5], vec![0.5, 2.0]];
    let cov = Matrix::new(cov_data);

    let mvn = MultivariateNormal::new(mean, cov);

    // Test point in the vector space
    let x = Vector::new(vec![0.0, 0.0]);

    // Compute log density
    let log_density: f64 = mvn.log_density(&x).into();

    // We can wrap in ExpFam for specialized computation
    let exp_fam_mvn = ExpFam::new(mvn.clone());
    let specialized_log_density: f64 = exp_fam_mvn.log_density(&x).into();

    // Results should be very close
    assert!((specialized_log_density - log_density).abs() < 1e-10);
}

#[test]
fn test_array_inner_product() {
    // Test the array inner product implementation
    let a: [f64; 3] = [1.0, 2.0, 3.0];
    let b: [f64; 3] = [4.0, 5.0, 6.0];

    // Compute using our InnerProduct trait
    let inner_product = a.inner_product(&b);

    // Compute manually
    let manual = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    assert_eq!(inner_product, manual);
}
