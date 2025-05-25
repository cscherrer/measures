/// Trait for type-level boolean values used in trait dispatch
pub trait TypeLevelBool {
    /// The boolean value at the type level
    const VALUE: bool;
}

/// Type-level representation of `true`
pub struct True;

/// Type-level representation of `false`
pub struct False;

impl TypeLevelBool for True {
    const VALUE: bool = true;
}

impl TypeLevelBool for False {
    const VALUE: bool = false;
}

/// Type-level enum for log density computation method
pub trait LogDensityMethod {
    /// The name of this method
    const NAME: &'static str;
}

/// Use the specialized method for computing log density
pub struct Specialized;

/// Use the exponential family method for computing log density
pub struct ExponentialFamily;

/// Use the default method for computing log density
pub struct Default;

impl LogDensityMethod for Specialized {
    const NAME: &'static str = "Specialized";
}

impl LogDensityMethod for ExponentialFamily {
    const NAME: &'static str = "ExponentialFamily";
}

impl LogDensityMethod for Default {
    const NAME: &'static str = "Default";
}
