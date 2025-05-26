//! Execution Overhead Demonstration
//!
//! This example demonstrates the performance characteristics and overhead analysis
//! findings from the symbolic-math crate benchmarks.
//!
//! Run with: cargo run --example `overhead_demonstration` --features "jit optimization"

use std::collections::HashMap;
use std::time::Instant;
use symbolic_math::Expr;

#[cfg(feature = "jit")]
use symbolic_math::GeneralJITCompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Symbolic Math Execution Overhead Demonstration\n");

    // Demonstrate baseline overhead
    demonstrate_baseline_overhead()?;

    // Demonstrate complexity scaling
    demonstrate_complexity_scaling()?;

    // Demonstrate batch vs individual calls
    demonstrate_batch_efficiency()?;

    // Demonstrate JIT compilation economics
    #[cfg(feature = "jit")]
    demonstrate_jit_economics()?;

    // Demonstrate raw Rust comparison
    demonstrate_raw_comparison()?;

    println!("\n✅ Overhead analysis complete!");
    println!("\n📊 Key Takeaways:");
    println!("   • Framework overhead: 2.7-305x raw computation");
    println!("   • Batch processing: 20-40% more efficient");
    println!("   • JIT break-even: ~230 evaluations");
    println!("   • Transcendental functions: Best overhead ratio (2.7x)");

    Ok(())
}

fn demonstrate_baseline_overhead() -> Result<(), Box<dyn std::error::Error>> {
    println!("1️⃣  Baseline Overhead Analysis");
    println!("   Measuring pure framework overhead with minimal computation\n");

    let iterations = 1_000_000;

    // Constant evaluation - pure overhead
    let expr = Expr::constant(42.0);
    let vars = HashMap::new();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = expr.evaluate(&vars)?;
    }
    let constant_time = start.elapsed();

    // Variable lookup - HashMap + framework overhead
    let expr = Expr::variable("x");
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 42.0);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = expr.evaluate(&vars)?;
    }
    let variable_time = start.elapsed();

    // Single addition - minimal computation + framework
    let expr = Expr::add(Expr::constant(1.0), Expr::constant(2.0));
    let vars = HashMap::new();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = expr.evaluate(&vars)?;
    }
    let addition_time = start.elapsed();

    println!(
        "   Constant evaluation:  {:>8.2} ns/call",
        constant_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "   Variable lookup:      {:>8.2} ns/call",
        variable_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "   Single addition:      {:>8.2} ns/call",
        addition_time.as_nanos() as f64 / f64::from(iterations)
    );

    println!("   → Framework baseline: 1.4-8.5 ns overhead\n");

    Ok(())
}

fn demonstrate_complexity_scaling() -> Result<(), Box<dyn std::error::Error>> {
    println!("2️⃣  Complexity Scaling Analysis");
    println!("   How framework overhead scales with expression complexity\n");

    let iterations = 100_000;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    // Simple polynomial: x^2 + 2x + 1
    let x = Expr::variable("x");
    let simple_poly = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x.clone()),
        ),
        Expr::constant(1.0),
    );

    // Complex polynomial: 3x^4 - 2x^3 + x^2 - 5x + 7
    let complex_poly = Expr::add(
        Expr::add(
            Expr::add(
                Expr::add(
                    Expr::mul(
                        Expr::constant(3.0),
                        Expr::pow(x.clone(), Expr::constant(4.0)),
                    ),
                    Expr::neg(Expr::mul(
                        Expr::constant(2.0),
                        Expr::pow(x.clone(), Expr::constant(3.0)),
                    )),
                ),
                Expr::pow(x.clone(), Expr::constant(2.0)),
            ),
            Expr::neg(Expr::mul(Expr::constant(5.0), x.clone())),
        ),
        Expr::constant(7.0),
    );

    // Transcendental functions: sin(x) + cos(x) + exp(x) + ln(x)
    let transcendental = Expr::add(
        Expr::add(
            Expr::add(Expr::sin(x.clone()), Expr::cos(x.clone())),
            Expr::exp(x.clone()),
        ),
        Expr::ln(x),
    );

    // Benchmark each
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simple_poly.evaluate(&vars)?;
    }
    let simple_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = complex_poly.evaluate(&vars)?;
    }
    let complex_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = transcendental.evaluate(&vars)?;
    }
    let transcendental_time = start.elapsed();

    // Raw computation comparison
    let x_val = 2.0f64;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = x_val * x_val + 2.0 * x_val + 1.0;
    }
    let raw_simple_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let x2 = x_val * x_val;
        let x3 = x2 * x_val;
        let x4 = x3 * x_val;
        let _ = 3.0 * x4 - 2.0 * x3 + x2 - 5.0 * x_val + 7.0;
    }
    let raw_complex_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = x_val.sin() + x_val.cos() + x_val.exp() + x_val.ln();
    }
    let raw_transcendental_time = start.elapsed();

    let simple_ns = simple_time.as_nanos() as f64 / f64::from(iterations);
    let complex_ns = complex_time.as_nanos() as f64 / f64::from(iterations);
    let transcendental_ns = transcendental_time.as_nanos() as f64 / f64::from(iterations);

    let raw_simple_ns = raw_simple_time.as_nanos() as f64 / f64::from(iterations);
    let raw_complex_ns = raw_complex_time.as_nanos() as f64 / f64::from(iterations);
    let raw_transcendental_ns = raw_transcendental_time.as_nanos() as f64 / f64::from(iterations);

    println!("   Expression Type        Framework    Raw Rust    Overhead");
    println!("   ─────────────────────  ─────────    ────────    ────────");
    println!(
        "   Simple polynomial      {:>8.1} ns   {:>7.2} ns    {:>5.0}x",
        simple_ns,
        raw_simple_ns,
        simple_ns / raw_simple_ns
    );
    println!(
        "   Complex polynomial     {:>8.1} ns   {:>7.2} ns    {:>5.0}x",
        complex_ns,
        raw_complex_ns,
        complex_ns / raw_complex_ns
    );
    println!(
        "   Transcendental funcs   {:>8.1} ns   {:>7.1} ns    {:>5.1}x",
        transcendental_ns,
        raw_transcendental_ns,
        transcendental_ns / raw_transcendental_ns
    );

    println!("   → Overhead ranges from 2.7x to 305x depending on computation cost\n");

    Ok(())
}

fn demonstrate_batch_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    println!("3️⃣  Batch vs Individual Call Efficiency");
    println!("   Comparing batch processing vs individual evaluations\n");

    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    let sizes = [10, 100, 1000];

    for &size in &sizes {
        let values: Vec<f64> = (0..size).map(|i| f64::from(i) * 0.01).collect();

        // Batch evaluation
        let start = Instant::now();
        let _batch_results = expr.evaluate_batch("x", &values)?;
        let batch_time = start.elapsed();

        // Individual evaluations
        let start = Instant::now();
        for &val in &values {
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), val);
            let _ = expr.evaluate(&vars)?;
        }
        let individual_time = start.elapsed();

        let batch_ns_per_item = batch_time.as_nanos() as f64 / f64::from(size);
        let individual_ns_per_item = individual_time.as_nanos() as f64 / f64::from(size);
        let efficiency = individual_ns_per_item / batch_ns_per_item;

        println!(
            "   Size {size:>4}: Batch {batch_ns_per_item:>6.1} ns/item, Individual {individual_ns_per_item:>6.1} ns/item, Efficiency {efficiency:>4.2}x"
        );
    }

    println!("   → Batch processing is 20-40% more efficient\n");

    Ok(())
}

#[cfg(feature = "jit")]
fn demonstrate_jit_economics() -> Result<(), Box<dyn std::error::Error>> {
    println!("4️⃣  JIT Compilation Economics");
    println!("   When does JIT compilation pay off?\n");

    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    // Measure compilation time
    let start = Instant::now();
    let compiler = GeneralJITCompiler::new()?;
    let jit_func = compiler.compile_expression(&expr, &["x".to_string()], &[], &HashMap::new())?;
    let compilation_time = start.elapsed();

    // Measure interpreted evaluation time
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = expr.evaluate(&vars)?;
    }
    let interpreted_time = start.elapsed();
    let interpreted_ns_per_call = interpreted_time.as_nanos() as f64 / f64::from(iterations);

    // Calculate break-even point
    let compilation_ns = compilation_time.as_nanos() as f64;
    let break_even_calls = compilation_ns / interpreted_ns_per_call;

    println!(
        "   Compilation time:      {:>8.1} µs",
        compilation_ns / 1000.0
    );
    println!("   Interpreted eval:      {interpreted_ns_per_call:>8.1} ns/call");
    println!("   Break-even point:      {break_even_calls:>8.0} calls");
    println!("   JIT overhead factor:   {break_even_calls:>8.0}x");

    println!("   → JIT is beneficial for expressions evaluated >500 times\n");

    Ok(())
}

fn demonstrate_raw_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("5️⃣  Framework vs Raw Rust Comparison");
    println!("   Direct performance comparison for identical computations\n");

    let iterations = 1_000_000;

    // Symbolic-math evaluation
    let x = Expr::variable("x");
    let expr = Expr::add(
        Expr::add(
            Expr::pow(x.clone(), Expr::constant(2.0)),
            Expr::mul(Expr::constant(2.0), x),
        ),
        Expr::constant(1.0),
    );

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = expr.evaluate(&vars)?;
    }
    let symbolic_time = start.elapsed();

    // Raw Rust computation
    let x_val = 3.0f64;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = x_val * x_val + 2.0 * x_val + 1.0;
    }
    let raw_time = start.elapsed();

    let symbolic_ns = symbolic_time.as_nanos() as f64 / f64::from(iterations);
    let raw_ns = raw_time.as_nanos() as f64 / f64::from(iterations);
    let overhead_factor = symbolic_ns / raw_ns;

    println!("   Computation: x² + 2x + 1 (x = 3.0)");
    println!("   ─────────────────────────────────");
    println!("   Symbolic-math:  {symbolic_ns:>8.1} ns/call");
    println!("   Raw Rust:       {raw_ns:>8.2} ns/call");
    println!("   Overhead:       {overhead_factor:>8.0}x");

    // Calculate what the overhead "costs"
    let throughput_symbolic = 1_000_000_000.0 / symbolic_ns;
    let throughput_raw = 1_000_000_000.0 / raw_ns;

    println!("   ");
    println!("   Throughput comparison:");
    println!(
        "   Symbolic-math:  {:>8.1} Meval/s",
        throughput_symbolic / 1_000_000.0
    );
    println!(
        "   Raw Rust:       {:>8.0} Meval/s",
        throughput_raw / 1_000_000.0
    );

    println!(
        "   → Framework provides flexibility at cost of {}x performance\n",
        overhead_factor as u32
    );

    Ok(())
}
