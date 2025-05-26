//! Measure combinators for building complex measures from simpler ones.
//!
//! This module provides combinators that allow you to build sophisticated
//! probability measures by composing simpler measures. This follows the
//! compositional approach of MeasureTheory.jl.

pub mod product;
pub mod pushforward;
pub mod restriction;
pub mod superposition;
pub mod transform;
