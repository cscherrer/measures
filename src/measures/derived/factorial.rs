use crate::core::{False, HasLogDensity, Measure, MeasureMarker};
use crate::measures::primitive::counting::CountingMeasure;
use num_traits::Float;

/// A factorial measure for discrete distributions.
///
/// This represents the measure dν = (1/k!) dμ where μ is the counting measure.
/// It's the natural base measure for discrete exponential families like Poisson
/// that have factorial terms in their densities.
#[derive(Clone)]
pub struct FactorialMeasure<F: Float> {
    /// The underlying counting measure
    counting: CountingMeasure<u64>,
    /// Phantom data for the Float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> FactorialMeasure<F> {
    /// Create a new factorial measure
    #[must_use]
    pub fn new() -> Self {
        Self {
            counting: CountingMeasure::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> Default for FactorialMeasure<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> MeasureMarker for FactorialMeasure<F> {
    type IsPrimitive = False;
    type IsExponentialFamily = False;
}

impl<F: Float> Measure<u64> for FactorialMeasure<F> {
    type RootMeasure = CountingMeasure<u64>;

    fn in_support(&self, x: u64) -> bool {
        self.counting.in_support(x)
    }

    fn root_measure(&self) -> Self::RootMeasure {
        self.counting.clone()
    }
}

/// Precomputed log-factorials for k = 0 to 20
///
/// This table contains log(k!) computed exactly for small k values.
/// Values were generated using the more accurate sum-of-logarithms method:
///
/// ```rust,ignore
/// for k in 0u64..=20 {
///     let log_factorial = if k == 0 {
///         0.0  // log(0!) = log(1) = 0
///     } else {
///         (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
///     };
///     println!("{:.15},  // log({}!)", log_factorial, k);
/// }
/// ```
///
/// This method avoids precision loss from computing large factorials directly
/// and is the gold standard for log-factorial computation.
const LOG_FACTORIAL_TABLE: [f64; 21] = [
    0.0,                     // log(0!) = log(1)
    0.0,                     // log(1!) = log(1)
    std::f64::consts::LN_2,  // log(2!) = log(2)
    1.791_759_469_228_055,   // log(3!)
    3.178_053_830_347_945_8, // log(4!)
    4.787_491_742_782_046,   // log(5!)
    6.579_251_212_010_101,   // log(6!)
    8.525_161_361_065_415,   // log(7!)
    10.604_602_902_745_25,   // log(8!)
    12.801_827_480_081_47,   // log(9!)
    15.104_412_573_075_518,  // log(10!)
    17.502_307_845_873_887,  // log(11!)
    19.987_214_495_661_89,   // log(12!)
    22.552_163_853_123_425,  // log(13!)
    25.191_221_182_738_683,  // log(14!)
    27.899_271_383_840_894,  // log(15!)
    30.671_860_106_080_675,  // log(16!)
    33.505_073_450_136_89,   // log(17!)
    36.395_445_208_033_05,   // log(18!)
    39.339_884_187_199_495,  // log(19!)
    42.335_616_460_753_485,  // log(20!)
];

/// Optimized O(1) log-factorial computation using lookup table + Stirling's approximation.
///
/// For k ≤ 20: Uses precomputed exact values from `LOG_FACTORIAL_TABLE`
/// For k > 20: Uses Stirling's approximation with Ramanujan's correction terms
///
/// This provides excellent accuracy with O(1) time complexity, making it suitable
/// for high-performance applications where factorial computation is a bottleneck.
#[inline]
#[must_use]
pub fn log_factorial<F: Float>(k: u64) -> F {
    if k <= 20 {
        // O(1): Direct lookup for small values
        F::from(LOG_FACTORIAL_TABLE[k as usize]).unwrap()
    } else {
        // O(1): Stirling's approximation for large values
        stirling_log_factorial_precise(k)
    }
}

/// Precise Stirling's approximation for log(k!) using Ramanujan's expansion.
///
/// Uses the formula: log(k!) ≈ k*ln(k) - k + 0.5*ln(2πk) + `correction_terms`
/// where `correction_terms` include the first few terms of Ramanujan's series.
///
/// This provides very high accuracy (error < 10^-10 for k >= 10) while maintaining
/// O(1) computational complexity.
#[inline]
fn stirling_log_factorial_precise<F: Float>(k: u64) -> F {
    if k <= 1 {
        return F::zero();
    }

    let k_f = F::from(k).unwrap();
    let two_pi = F::from(2.0).unwrap() * F::from(std::f64::consts::PI).unwrap();

    // Base Stirling's formula: log(k!) ≈ k*log(k) - k + 0.5*log(2πk)
    let base = k_f * k_f.ln() - k_f + F::from(0.5).unwrap() * (two_pi * k_f).ln();

    // Ramanujan's asymptotic expansion for higher accuracy:
    // log(k!) ≈ base + 1/(12k) - 1/(360k³) + 1/(1260k⁵) - 1/(1680k⁷) + ...
    let k_inv = F::from(1.0).unwrap() / k_f;
    let k_inv2 = k_inv * k_inv;
    let k_inv3 = k_inv2 * k_inv;
    let k_inv5 = k_inv3 * k_inv2;
    let k_inv7 = k_inv5 * k_inv2;

    let correction = k_inv * F::from(1.0 / 12.0).unwrap()                      // +1/(12k)
                   - k_inv3 * F::from(1.0 / 360.0).unwrap()          // -1/(360k³)
                   + k_inv5 * F::from(1.0 / 1260.0).unwrap() // +1/(1260k⁵)
                   - k_inv7 * F::from(1.0 / 1680.0).unwrap(); // -1/(1680k⁷)

    base + correction
}

/// Implement `HasLogDensity` for `FactorialMeasure`
/// This provides the factorial term: log(dν/dμ) = -log(k!)
impl<F: Float> HasLogDensity<u64, F> for FactorialMeasure<F> {
    #[cfg(feature = "profiling")]
    #[profiling::function]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        profiling::scope!("factorial_computation");
        let k = *x;

        // Use optimized O(1) log-factorial computation
        -log_factorial::<F>(k)
    }

    #[cfg(not(feature = "profiling"))]
    #[inline]
    fn log_density_wrt_root(&self, x: &u64) -> F {
        let k = *x;

        // Use optimized O(1) log-factorial computation
        -log_factorial::<F>(k)
    }
}
