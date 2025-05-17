/// Type-level boolean traits and implementations
pub trait TypeLevelBool {
    const VALUE: bool;
}

pub struct True;

pub struct False;

impl TypeLevelBool for True {
    const VALUE: bool = true;
}

impl TypeLevelBool for False {
    const VALUE: bool = false;
}
