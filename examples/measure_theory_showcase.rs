//! Comprehensive showcase of the measure-theoretic framework.
//!
//! This example demonstrates how the unified measure-theoretic approach
//! works for both exponential families and non-exponential families,
//! and how combinators can be used to build complex models.

use measures::distributions::continuous::cauchy::Cauchy;
use measures::distributions::discrete::poisson::Poisson;
use measures::mixture;
use measures::{LogDensityBuilder, MixtureExt, Normal, ProductMeasureExt};

fn main() {
    println!("=== Measure Theory Showcase ===\n");

    // 1. Basic exponential family vs non-exponential family
    demonstrate_exponential_vs_non_exponential();

    // 2. Product measures for independence
    demonstrate_product_measures();

    // 3. Pushforward measures for transformations
    demonstrate_pushforward_measures();

    // 4. Mixture measures for complex models
    demonstrate_mixture_measures();

    // 5. Complex hierarchical model
    demonstrate_hierarchical_model();
}

fn demonstrate_exponential_vs_non_exponential() {
    println!("1. Exponential Family vs Non-Exponential Family");
    println!("================================================");

    // Exponential family: Normal distribution
    let normal = Normal::new(0.0, 1.0);
    let normal_density: f64 = normal.log_density().at(&0.0);
    println!("Normal(0,1) log-density at x=0: {normal_density:.6}");

    // Non-exponential family: Cauchy distribution
    let cauchy = Cauchy::new(0.0, 1.0);
    let cauchy_density: f64 = cauchy.log_density().at(&0.0);
    println!("Cauchy(0,1) log-density at x=0: {cauchy_density:.6}");

    // Both work seamlessly in the same framework!
    let relative_density: f64 = normal.log_density().wrt(cauchy).at(&0.0);
    println!("Relative density Normal/Cauchy at x=0: {relative_density:.6}");

    println!();
}

fn demonstrate_product_measures() {
    println!("2. Product Measures for Independence");
    println!("===================================");

    // Independent normal and Poisson
    let normal = Normal::new(1.0, 2.0);
    let poisson = Poisson::new(3.0);

    // Clone before moving to product
    let joint = normal.clone().product(poisson.clone());
    let joint_density: f64 = joint.log_density().at(&(1.5, 2u64));

    println!("Joint Normal-Poisson density at (1.5, 2): {joint_density:.6}");

    // Verify independence: joint = marginal1 + marginal2
    let marginal1: f64 = normal.log_density().at(&1.5);
    let marginal2: f64 = poisson.log_density().at(&2u64);
    let sum_marginals = marginal1 + marginal2;

    println!("Sum of marginals: {sum_marginals:.6}");
    println!(
        "Difference (should be ~0): {:.10}",
        joint_density - sum_marginals
    );

    println!();
}

fn demonstrate_pushforward_measures() {
    println!("3. Pushforward Measures for Transformations");
    println!("===========================================");

    // For now, skip pushforward examples since they need trait implementation fixes
    println!("Pushforward measures temporarily disabled due to trait implementation issues.");
    println!("This will be fixed in the next iteration.");

    println!();
}

fn demonstrate_mixture_measures() {
    println!("4. Mixture Measures for Complex Models");
    println!("======================================");

    // Gaussian mixture model
    let component1 = Normal::new(-2.0, 1.0);
    let component2 = Normal::new(2.0, 1.0);
    let component3 = Normal::new(0.0, 0.5);

    let mixture = mixture![(0.3, component1), (0.5, component2), (0.2, component3)];

    // Evaluate at different points
    let points = [-3.0, -1.0, 0.0, 1.0, 3.0];
    println!("Gaussian mixture densities:");
    for &x in &points {
        let density: f64 = mixture.log_density().at(&x);
        println!("  x={x:4.1}: {density:.6}");
    }

    // Uniform mixture using fluent interface
    let normal1 = Normal::new(-1.0, 1.0);
    let normal2 = Normal::new(1.0, 1.0);
    let uniform_mix = normal1.uniform_mixture(normal2);

    let uniform_density: f64 = uniform_mix.log_density().at(&0.0);
    println!("Uniform mixture density at x=0: {uniform_density:.6}");

    println!();
}

fn demonstrate_hierarchical_model() {
    println!("5. Complex Hierarchical Model");
    println!("=============================");

    // Build a simpler model for now (without pushforward)
    // 1. Start with a mixture of normals
    let component1 = Normal::new(-1.0, 0.5);
    let component2 = Normal::new(1.0, 0.5);
    let base_mixture = component1.uniform_mixture(component2);

    // 2. Create a joint model with a Poisson component
    let poisson = Poisson::new(2.0);
    let joint_model = base_mixture.product(poisson);

    // 3. Evaluate the model
    let density: f64 = joint_model.log_density().at(&(0.5, 1u64));
    println!("Hierarchical model density: {density:.6}");

    // 4. Show that we can still compute relative densities
    let simple_normal = Normal::new(0.0, 1.0);
    let simple_poisson = Poisson::new(2.0);
    let simple_joint = simple_normal.product(simple_poisson);

    let relative: f64 = joint_model.log_density().wrt(simple_joint).at(&(0.5, 1u64));
    println!("Relative density vs simple model: {relative:.6}");

    println!();
    println!("=== Framework Benefits ===");
    println!("✓ Unified interface for all distributions");
    println!("✓ Exponential families and non-exponential families work seamlessly");
    println!("✓ Compositional: build complex models from simple parts");
    println!("✓ Type-safe measure theory operations");
    println!("✓ Efficient relative density computations");
    println!("✓ Pushforward measures (coming soon with trait fixes)");
}
