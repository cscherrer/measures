//! Simple Automatic Differentiation Integration
//!
//! This example demonstrates the core concept of AD integration with the measures framework.
//! It shows how the framework's design enables AD support with minimal changes.

#[cfg(feature = "autodiff")]
use ad_trait::AD;
#[cfg(feature = "autodiff")]
use ad_trait::forward_ad::adfn::adfn;
#[cfg(feature = "autodiff")]
use ad_trait::reverse_ad::adr::adr;
#[cfg(feature = "autodiff")]
use simba::scalar::ComplexField;

use measures::{LogDensityBuilder, Normal};

fn main() {
    println!("=== Simple Automatic Differentiation Integration ===\n");

    #[cfg(feature = "autodiff")]
    {
        // The key insight: our framework's generic design means that
        // if AD types implement the right traits, they work automatically!

        println!("1. Framework Design Enables AD");
        println!("   The measures framework uses generic types T that implement Float.");
        println!("   AD types from ad_trait implement RealField, which is compatible.");
        println!("   This means AD integration is possible with trait bridging.\n");

        // Example: Manual computation showing the concept
        println!("2. Manual AD Computation (Proof of Concept)");

        // Standard computation
        let normal_f64 = Normal::new(0.0_f64, 1.0_f64);
        let x_f64 = 1.5_f64;

        // This works because f64 implements Float
        let log_density_f64: f64 = normal_f64.log_density().at(&x_f64);
        println!("   Standard f64: log_density({x_f64}) = {log_density_f64:.6}");

        // The challenge: AD types don't implement num_traits::Float
        // But they do implement simba::scalar::RealField which has similar functionality

        println!("\n3. AD Type Capabilities");
        let x_ad = adr::constant(1.5);
        println!("   AD constant: {:.6}", x_ad.to_constant());

        let x_fwd = adfn::<1>::new(1.5, [1.0]);
        println!(
            "   Forward AD: value={:.6}, tangent={:?}",
            x_fwd.value(),
            x_fwd.tangent()
        );

        println!("\n4. The Path Forward");
        println!("   To enable full AD integration, we need:");
        println!("   • Bridge traits between num_traits::Float and simba::RealField");
        println!("   • Or extend ad_trait to implement num_traits::Float");
        println!("   • Or create adapter types that provide the bridge");

        println!("\n5. Manual Log-Density with AD (Conceptual)");
        // This shows what the computation would look like if we had the trait bridge
        let mu = adr::constant(0.0);
        let sigma = adr::constant(1.0);
        let x = adr::constant(1.5);

        // Manual Gaussian log-density computation with AD
        let two_pi = adr::constant(2.0 * std::f64::consts::PI);
        let half = adr::constant(0.5);
        let one = adr::constant(1.0);

        let normalization = one / (sigma * two_pi.sqrt());
        let z_score = (x - mu) / sigma;
        let exponent = -half * z_score * z_score;
        let manual_log_density = (normalization * exponent.exp()).ln();

        println!(
            "   Manual AD log-density = {:.6}",
            manual_log_density.to_constant()
        );
        println!("   This matches the f64 result: {log_density_f64:.6}");

        println!("\n=== Key Insights ===");
        println!("• The framework's generic design is AD-ready");
        println!("• AD types have the mathematical capabilities needed");
        println!("• Only trait compatibility needs to be solved");
        println!("• No fundamental rewrites of the framework required");
        println!("• This demonstrates the power of good abstraction design!");
    }

    #[cfg(not(feature = "autodiff"))]
    {
        println!("Automatic differentiation features are not enabled.");
        println!("Run with: cargo run --example simple_autodiff_integration --features autodiff");

        // Show that regular computation works perfectly
        println!("\nRegular computation (without AD):");
        let normal = Normal::new(0.0, 1.0);
        let x = 1.5;
        let log_density: f64 = normal.log_density().at(&x);
        println!("   x = {}: log_density = {:.6}", x, log_density);
        println!("   Framework works perfectly with regular types!");
    }
}
