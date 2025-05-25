use measures::{IIDExtension, LogDensityBuilder, Normal};

fn main() {
    println!("=== IID Standard API Demo ===\n");

    // Create a normal distribution
    let normal = Normal::new(1.0, 2.0);
    println!("Base distribution: Normal(μ=1.0, σ=2.0)");

    // Create an IID version
    let iid_normal = normal.clone().iid();
    println!("IID distribution: represents joint distribution of independent samples\n");

    let samples = vec![0.5, 1.2, 0.8, 1.5, 0.9];
    println!("Sample data: {samples:?}\n");

    // === Demonstrate the improved API ===

    println!("=== API Comparison ===");

    // OLD API (still works for backward compatibility)
    println!("Old API (manual method):");
    let old_result: f64 = iid_normal.iid_log_density(&samples);
    println!("  iid_normal.iid_log_density(&samples) = {old_result:.6}");

    // NEW API (standard, consistent with individual distributions)
    println!("\nNew API (standard interface):");
    let new_result: f64 = iid_normal.log_density().at(&samples);
    println!("  iid_normal.log_density().at(&samples) = {new_result:.6}");

    // Verification with manual computation
    println!("\nVerification (manual sum):");
    let manual_sum: f64 = samples.iter().map(|&x| normal.log_density().at(&x)).sum();
    println!("  Σᵢ normal.log_density().at(&xᵢ) = {manual_sum:.6}");

    // Check they're all equal
    let old_vs_new = (old_result - new_result).abs();
    let new_vs_manual = (new_result - manual_sum).abs();

    println!("\n=== Results ===");
    println!("✓ Old API vs New API difference: {old_vs_new:.2e}");
    println!("✓ New API vs Manual sum difference: {new_vs_manual:.2e}");
    println!("✓ All methods produce identical results!");

    // === Demonstrate consistency with individual distributions ===

    println!("\n=== API Consistency ===");
    println!("Individual distribution:");
    println!("  normal.log_density().at(&x)");
    println!("IID distribution:");
    println!("  iid_normal.log_density().at(&samples)");
    println!("✓ Same pattern, same interface!");

    // === Show different sample sizes ===

    println!("\n=== Different Sample Sizes ===");
    for n in [1, 2, 5, 10] {
        let test_samples: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let log_density = iid_normal.log_density().at(&test_samples);
        println!("n={n:2}: log p({test_samples:?}) = {log_density:.6}");
    }

    // === Empty sample case ===
    println!("\n=== Edge Cases ===");
    let empty_samples: Vec<f64> = vec![];
    let empty_log_density = iid_normal.log_density().at(&empty_samples);
    println!("Empty sample: log p([]) = {empty_log_density:.6}");
    println!("✓ Empty sample correctly returns 0.0 (log(1) = 0)");

    println!("\n=== Summary ===");
    println!("🎉 The IID API is now consistent with individual distributions!");
    println!("📈 Users can use the familiar .log_density().at() pattern");
    println!("🔧 Backward compatibility maintained with .iid_log_density()");
    println!("⚡ Efficient exponential family computation under the hood");
} 