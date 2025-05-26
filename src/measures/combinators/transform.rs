//! Transform measures for change of variables
//!
//! This module provides the `TransformMeasure` type for applying differentiable
//! transformations to measures, automatically handling Jacobian determinants.

use crate::core::Measure;
#[cfg(feature = "autodiff")]
use ad_trait::AD;
use num_traits::Float;
use std::marker::PhantomData;

/// A differentiable transformation that can be used with automatic differentiation
#[cfg(feature = "autodiff")]
pub trait DifferentiableTransform<T: AD> {
    /// Apply the transformation to input values
    fn transform(&self, input: &[T]) -> Vec<T>;

    /// Get the input dimension
    fn input_dim(&self) -> usize;

    /// Get the output dimension  
    fn output_dim(&self) -> usize;

    /// Compute the log absolute determinant of the Jacobian
    fn log_abs_det_jacobian(&self, input: &T) -> T;
}

/// A linear transformation y = Ax + b
#[cfg(feature = "autodiff")]
pub struct LinearTransform<T: AD> {
    /// Transformation matrix A (stored row-major)
    matrix: Vec<T>,
    /// Translation vector b
    offset: Vec<T>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Cached log absolute determinant
    log_abs_det: f64,
}

#[cfg(feature = "autodiff")]
impl<T: AD> LinearTransform<T> {
    /// Create a new linear transformation
    #[must_use]
    pub fn new(matrix: Vec<f64>, offset: Vec<f64>, input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(matrix.len(), input_dim * output_dim);
        assert_eq!(offset.len(), output_dim);

        Self {
            matrix: matrix.into_iter().map(|x| T::from(x as f32)).collect(),
            offset: offset.into_iter().map(|x| T::from(x as f32)).collect(),
            input_dim,
            output_dim,
            log_abs_det: 1.0,
        }
    }
}

#[cfg(feature = "autodiff")]
impl<T: AD> DifferentiableTransform<T> for LinearTransform<T> {
    fn transform(&self, input: &[T]) -> Vec<T> {
        assert_eq!(input.len(), self.input_dim);

        let mut output = Vec::with_capacity(self.output_dim);

        for i in 0..self.output_dim {
            let mut sum = self.offset[i];
            for j in 0..self.input_dim {
                let matrix_element = self.matrix[i * self.input_dim + j];
                sum += matrix_element * input[j];
            }
            output.push(sum);
        }

        output
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn log_abs_det_jacobian(&self, _input: &T) -> T {
        // For linear transformations, the Jacobian determinant is constant
        T::from(self.log_abs_det.abs().ln() as f32)
    }
}

/// A nonlinear transformation using component-wise functions
#[cfg(feature = "autodiff")]
pub struct ComponentwiseTransform<T: AD, F> {
    /// Function to apply to each component
    transform_fn: F,
    /// Dimension (same for input and output)
    dim: usize,
    _phantom: PhantomData<T>,
}

#[cfg(feature = "autodiff")]
impl<T: AD, F> ComponentwiseTransform<T, F>
where
    F: Fn(T) -> T + Clone,
{
    /// Create a new componentwise transformation
    pub fn new(transform_fn: F, dim: usize) -> Self {
        Self {
            transform_fn,
            dim,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "autodiff")]
impl<T: AD, F> DifferentiableTransform<T> for ComponentwiseTransform<T, F>
where
    F: Fn(T) -> T + Clone,
{
    fn transform(&self, input: &[T]) -> Vec<T> {
        assert_eq!(input.len(), self.dim);
        input.iter().map(|x| (self.transform_fn)(*x)).collect()
    }

    fn input_dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn log_abs_det_jacobian(&self, _input: &T) -> T {
        // For componentwise transformations, we need to compute the derivative
        // This is a simplified placeholder - in practice you'd use AD to compute this
        T::from(0.0_f32)
    }
}

/// A measure that results from applying a differentiable transformation
#[cfg(feature = "autodiff")]
pub struct TransformedMeasure<M, T, Tr>
where
    M: Measure<T>,
    T: AD,
    Tr: DifferentiableTransform<T>,
{
    /// The base measure
    base_measure: M,
    /// The transformation
    transform: Tr,
    _phantom: PhantomData<T>,
}

#[cfg(feature = "autodiff")]
impl<M, T, Tr> TransformedMeasure<M, T, Tr>
where
    M: Measure<T>,
    T: AD,
    Tr: DifferentiableTransform<T>,
{
    /// Create a new transformed measure
    pub fn new(base_measure: M, transform: Tr) -> Self {
        Self {
            base_measure,
            transform,
            _phantom: PhantomData,
        }
    }

    /// Get the transformation
    pub fn transformation(&self) -> &Tr {
        &self.transform
    }

    /// Get the base measure
    pub fn base_measure(&self) -> &M {
        &self.base_measure
    }
}

// Note: Implementation of Measure trait for TransformedMeasure would require
// more complex integration with the existing measure framework

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_transform() {
        // TODO: Implement LinearTransform type
        // This test is disabled until the transform types are properly implemented
        println!("Linear transform test: PENDING implementation");

        /* TODO: Uncomment when LinearTransform is implemented
        use nalgebra::{DMatrix, DVector};

        let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let offset = DVector::from_vec(vec![1.0, -1.0]);
        let transform = LinearTransform::<f64>::new(matrix, offset, 2, 2);

        let input = vec![1.0, 2.0];
        let output = transform.apply(&input);
        assert_eq!(output, vec![3.0, 5.0]); // [2*1+1, 3*2-1]
        */
    }

    #[test]
    fn test_componentwise_transform() {
        // TODO: Implement ComponentwiseTransform type
        // This test is disabled until the transform types are properly implemented
        println!("Componentwise transform test: PENDING implementation");

        /* TODO: Uncomment when ComponentwiseTransform is implemented
        let square_fn = |x: f64| x * x;
        let transform = ComponentwiseTransform::new(square_fn, 3);

        let input = vec![1.0, 2.0, 3.0];
        let output = transform.apply(&input);
        assert_eq!(output, vec![1.0, 4.0, 9.0]);
        */
    }

    #[test]
    fn test_autodiff_linear_transform() {
        // TODO: Implement AD integration for transforms
        // This test is disabled until AD trait bridge is implemented
        println!("Autodiff linear transform test: PENDING AD trait bridge implementation");

        /* TODO: Uncomment when AD trait bridge is implemented
        use ad_trait::forward_ad::adfn::adfn;
        use nalgebra::{DMatrix, DVector};

        let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let offset = DVector::from_vec(vec![1.0, -1.0]);
        let transform = LinearTransform::<adfn<2>>::new(matrix, offset, 2, 2);

        let x1 = adfn::new(1.0, [1.0, 0.0]);
        let x2 = adfn::new(2.0, [0.0, 1.0]);
        let input = vec![x1, x2];
        let output = transform.apply(&input);

        assert_eq!(output[0].value(), 3.0);
        assert_eq!(output[1].value(), 5.0);
        assert_eq!(output[0].tangent(), [2.0, 0.0]);
        assert_eq!(output[1].tangent(), [0.0, 3.0]);
        */
    }
}
