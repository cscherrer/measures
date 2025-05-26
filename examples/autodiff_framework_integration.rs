//! Automatic Differentiation Integration with Measures Framework
//!
//! This example demonstrates how automatic differentiation works seamlessly
//! with the existing measures framework. No rewrites needed - just use AD types!

#[cfg(feature = "autodiff")]
use ad_trait::AD;
#[cfg(feature = "autodiff")]
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ReverseAD};
#[cfg(feature = "autodiff")]
use ad_trait::forward_ad::adfn::adfn;
#[cfg(feature = "autodiff")]
use ad_trait::function_engine::FunctionEngine;
#[cfg(feature = "autodiff")]
use ad_trait::reverse_ad::adr::adr;
#[cfg(feature = "autodiff")]
use num_traits::{Float, FloatConst};

use measures::{LogDensityBuilder, Normal};

fn main() {
    println!("=== Automatic Differentiation with Measures Framework ===\n");

    // NOTE: This example demonstrates the trait bridge issue mentioned in the roadmap.
    // The AD types (adr, adfn) don't implement the required traits (Float, FloatConst)
    // that our framework expects. This is the "trait bridge implementation" task
    // identified as high priority in the roadmap.
    
    println!("=== Automatic Differentiation Integration Status ===");
    println!("This example is currently disabled due to missing trait implementations.");
    println!("The AD types need to implement num_traits::Float and FloatConst traits.");
    println!("This is tracked as a high-priority item in the roadmap.\n");
    
    println!("=== What Should Work (After Trait Bridge Implementation) ===");
    println!("1. Normal distribution with reverse-mode AD");
    println!("2. Normal distribution with forward-mode AD");
    println!("3. Parameter derivatives");
    println!("4. Batch computation with AD");
    println!("5. Zero-cost abstractions when AD not used\n");

    #[cfg(not(feature = "autodiff"))]
    {
        println!("Automatic differentiation features are not enabled.");
        println!(
            "Run with: cargo run --example autodiff_framework_integration --features autodiff"
        );
    }

    // Show that regular computation still works
    println!("=== Regular Computation (Without AD) ===");
    let normal = Normal::new(0.0, 1.0);
    let x = 1.5;
    let log_density: f64 = normal.log_density().at(&x);
    println!("   x = {}: log_density = {:.6}", x, log_density);
    
    println!("\n=== Next Steps ===");
    println!("1. Implement trait bridge between num_traits::Float and AD types");
    println!("2. Implement trait bridge between FloatConst and AD types");
    println!("3. Test integration with existing distributions");
    println!("4. Add comprehensive AD examples");
    
    /* TODO: Uncomment when trait bridge is implemented
    
    #[cfg(feature = "autodiff")]
    {
        // Example 1: Normal distribution with reverse-mode AD
        println!("1. Normal Distribution with Reverse-Mode AD");
        let normal = Normal::new(0.0, 1.0);

        // Create AD variable
        let x = adr::constant(1.5);

        // Use the SAME API as with regular floats!
        let log_density: adr = normal.log_density().at(&x);

        println!("   x = 1.5");
        println!("   log_density = {:.6}", log_density.to_constant());
        println!("   This is an AD type that can compute gradients!\n");

        // Example 2: Forward-mode AD with the same distribution
        println!("2. Normal Distribution with Forward-Mode AD");
        let x_fwd = adfn::<1>::new(1.5, [1.0]); // x with tangent vector [1.0]

        // SAME API again!
        let log_density_fwd: adfn<1> = normal.log_density().at(&x_fwd);

        println!("   x = 1.5 (with tangent [1.0])");
        println!("   log_density = {:.6}", log_density_fwd.value());
        println!("   derivative = {:.6}", log_density_fwd.tangent()[0]);
        println!("   This computed the derivative automatically!\n");

        // Example 3: Different parameters with AD
        println!("3. Parameter Derivatives");

        // Create a function that computes log-density for different parameters
        #[derive(Clone)]
        struct NormalLogDensity<T> {
            x: T,
        }

        impl<T> DifferentiableFunctionTrait<T> for NormalLogDensity<T>
        where
            T: AD + Float + FloatConst,
        {
            const NAME: &'static str = "NormalLogDensity";

            fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
                let mu = inputs[0];
                let sigma = inputs[1];

                // Create Normal distribution with AD parameters
                let normal = Normal::new(mu, sigma);

                // Compute log-density - this works because Normal<T> implements
                // the same traits for AD types as for regular floats!
                vec![normal.log_density().at(&self.x)]
            }

            fn num_inputs(&self) -> usize {
                2
            } // mu, sigma
            fn num_outputs(&self) -> usize {
                1
            }
        }

        impl<T: AD> NormalLogDensity<T> {
            fn to_other_ad_type<T2: AD>(&self) -> NormalLogDensity<T2> {
                NormalLogDensity {
                    x: self.x.to_other_ad_type::<T2>(),
                }
            }
        }

        // Set up the function engine
        let x_val = 1.5;
        let function_standard = NormalLogDensity { x: x_val };
        let function_derivative = function_standard.to_other_ad_type::<adr>();
        let engine = FunctionEngine::new(function_standard, function_derivative, ReverseAD::new());

        // Compute gradients with respect to parameters
        let params = vec![0.0, 1.0]; // mu=0, sigma=1
        let (log_density, gradients) = engine.derivative(&params);

        println!("   Parameters: μ = {}, σ = {}", params[0], params[1]);
        println!("   x = {}", x_val);
        println!("   log_density = {:.6}", log_density[0]);
        println!("   ∂/∂μ = {:.6}", gradients[(0, 0)]);
        println!("   ∂/∂σ = {:.6}", gradients[(0, 1)]);
        println!("   Gradients computed automatically!\n");

        // Example 4: Batch computation with AD
        println!("4. Batch Computation with AD");
        let x_values = vec![0.0, 1.0, -1.0, 2.0];

        for &x_val in &x_values {
            let x_ad = adr::constant(x_val);
            let log_density: adr = normal.log_density().at(&x_ad);
            println!(
                "   x = {:.1}: log_density = {:.6}",
                x_val,
                log_density.to_constant()
            );
        }

        println!("\n=== Key Benefits ===");
        println!("• No framework rewrites needed");
        println!("• Same API for regular floats and AD types");
        println!("• Automatic gradient computation");
        println!("• Works with all existing distributions");
        println!("• Type-safe and zero-cost when AD not used");
    }
    */
}
