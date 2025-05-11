mod traits;
pub mod measures;

use traits::{PrimitiveMeasure, Measure, HasDensity, DensityBuilder, DensityWithRespectTo};
use measures::{LebesgueMeasure, CountingMeasure, Dirac};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirac_density() {
        let dirac = Dirac::new(5);
        let counting = CountingMeasure::new();

        // These are all equivalent:
        let density1: f64 = dirac.density(5).wrt(&counting).into();
        let density2 = Into::<f64>::into(dirac.density(5).wrt(&counting));
        let log_density = dirac.density(5).wrt(&counting).log();

        assert_eq!(density1, 1.0);
        assert_eq!(density2, 1.0);
        assert_eq!(log_density, 0.0);
    }
} 