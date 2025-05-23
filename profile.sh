#!/bin/bash

echo "🔥 Measures Performance Profiling Suite"
echo "========================================"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found!"
    exit 1
fi

echo ""
echo "📊 Running CPU benchmarks with flamegraph generation..."
echo "This will create detailed performance profiles in target/criterion/"

# Run criterion benchmarks with flamegraph generation
cargo bench --profile profiling

echo ""
echo "🧠 Running memory profiling with dhat..."
echo "This will identify allocation hotspots and cloning overhead"

# Run memory profiling example
CARGO_FEATURE_DHAT_HEAP=1 cargo run --profile profiling --features dhat-heap --example memory_profiling

echo ""
echo "🎯 Running general profiling workload..."
echo "This simulates real-world usage patterns"

# Run the profiling workload for general performance analysis
time cargo run --profile profiling --example profiling_workload

echo ""
echo "📈 Results Summary:"
echo "- CPU Benchmarks: target/criterion/reports/index.html"  
echo "- Flamegraphs: target/criterion/*/profile/flamegraph.svg"
echo "- Memory Profile: dhat-heap.json (use DHAT viewer - search 'dhat viewer' or Valgrind's dh_view.html)"
echo ""

# Optional: Generate additional flamegraph with cargo-flamegraph (if installed)
if command -v cargo-flamegraph &> /dev/null; then
    echo "🔥 Generating additional flamegraph with cargo-flamegraph..."
    cargo flamegraph --profile profiling --example profiling_workload -o profiling_workload.svg
    echo "- Additional flamegraph: profiling_workload.svg"
    
    echo ""
    echo "🧮 Generating Poisson-specific flamegraph..."
    cargo flamegraph --profile profiling --example memory_profiling -o memory_profiling.svg
    echo "- Memory profiling flamegraph: memory_profiling.svg"
else
    echo "💡 Install cargo-flamegraph for additional profiling:"
    echo "   cargo install flamegraph"
fi

# Optional: Generate callgrind analysis (if available)
if command -v valgrind &> /dev/null && cargo --list | grep -q "iai-callgrind"; then
    echo ""
    echo "🔬 Running precise instruction-level analysis with callgrind..."
    cargo bench --features callgrind --profile profiling callgrind
else
    echo "💡 Install valgrind + iai-callgrind for precise analysis:"
    echo "   sudo apt install valgrind  # or equivalent for your system"
fi

echo ""
echo "✅ Profiling complete!"
echo ""
echo "🔍 Analysis Steps:"
echo "1. 📊 Open target/criterion/reports/index.html for benchmark results"
echo "2. 🔥 View flamegraphs to identify CPU hotspots"
echo "3. 🧠 Analyze dhat-heap.json with DHAT viewer (search 'dhat viewer' or use Valgrind's dh_view.html)"
echo "4. 🎯 Look for:"
echo "   - Functions with high sample counts in flamegraphs"
echo "   - Excessive allocations in memory profile"
echo "   - Clone-heavy operations in allocation patterns"
echo "   - Factorial computation scaling in Poisson" 