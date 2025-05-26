//! General transformation measures.
//!
//! This module implements general transformation measures
//! beyond simple pushforward operations, including support
//! for automatic differentiation of transformations.

use crate::core::Measure;
use ad_trait::AD;
use std::marker::PhantomData;

/// A differentiable transformation that can be used with automatic differentiation
pub trait DifferentiableTransform<T: AD> {
    /// Apply the transformation to input values
    fn transform(&self, input: &[T]) -> Vec<T>;

    /// Get the input dimension
    fn input_dim(&self) -> usize;

    /// Get the output dimension  
    fn output_dim(&self) -> usize;
}

/// A linear transformation y = Ax + b
pub struct LinearTransform<T: AD> {
    /// Transformation matrix A (stored row-major)
    matrix: Vec<T>,
    /// Translation vector b
    offset: Vec<T>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl<T: AD> LinearTransform<T> {
    /// Create a new linear transformation
    pub fn new(matrix: Vec<f64>, offset: Vec<f64>, input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(matrix.len(), input_dim * output_dim);
        assert_eq!(offset.len(), output_dim);

        Self {
            matrix: matrix.into_iter().map(T::constant).collect(),
            offset: offset.into_iter().map(T::constant).collect(),
            input_dim,
            output_dim,
        }
    }
}

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
}

/// A nonlinear transformation using component-wise functions
pub struct ComponentwiseTransform<T: AD, F> {
    /// Function to apply to each component
    transform_fn: F,
    /// Dimension (same for input and output)
    dim: usize,
    _phantom: PhantomData<T>,
}

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

impl<T: AD, F> DifferentiableTransform<T> for ComponentwiseTransform<T, F>
where
    F: Fn(T) -> T + Clone,
{
    fn transform(&self, input: &[T]) -> Vec<T> {
        assert_eq!(input.len(), self.dim);
        input.iter().map(|&x| (self.transform_fn)(x)).collect()
    }

    fn input_dim(&self) -> usize {
        self.dim
    }

    fn output_dim(&self) -> usize {
        self.dim
    }
}

/// A measure that results from applying a differentiable transformation
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
    use ad_trait::forward_ad::adfn::adfn;

    #[test]
    fn test_linear_transform() {
        // Test 2D rotation by 90 degrees: [x, y] -> [-y, x]
        let matrix = vec![0.0, -1.0, 1.0, 0.0]; // Row-major: [[0, -1], [1, 0]]
        let offset = vec![0.0, 0.0];
        let transform = LinearTransform::<f64>::new(matrix, offset, 2, 2);

        let input = vec![1.0, 0.0];
        let output = transform.transform(&input);

        assert!((output[0] - 0.0).abs() < 1e-10);
        assert!((output[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_componentwise_transform() {
        // Test squaring transformation
        let square_fn = |x: f64| x * x;
        let transform = ComponentwiseTransform::new(square_fn, 3);

        let input = vec![1.0, -2.0, 3.0];
        let output = transform.transform(&input);

        assert_eq!(output, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_autodiff_linear_transform() {
        // Test with automatic differentiation
        let matrix = vec![2.0, 0.0, 0.0, 3.0]; // [[2, 0], [0, 3]]
        let offset = vec![1.0, -1.0];
        let transform = LinearTransform::<adfn<2>>::new(matrix, offset, 2, 2);

        // Create input variables with tangent vectors
        let input = vec![
            adfn::new(1.0, [1.0, 0.0]), // x with tangent [1, 0]
            adfn::new(2.0, [0.0, 1.0]), // y with tangent [0, 1]
        ];
        let output = transform.transform(&input);

        // Expected: [2*1 + 1, 3*2 - 1] = [3, 5]
        assert!((output[0].value() - 3.0).abs() < 1e-10);
        assert!((output[1].value() - 5.0).abs() < 1e-10);

        // Check derivatives: d/dx [2x + 1, 3y - 1] = [2, 0]
        assert!((output[0].tangent()[0] - 2.0).abs() < 1e-10);
        assert!((output[1].tangent()[0] - 0.0).abs() < 1e-10);

        // Check derivatives: d/dy [2x + 1, 3y - 1] = [0, 3]
        assert!((output[0].tangent()[1] - 0.0).abs() < 1e-10);
        assert!((output[1].tangent()[1] - 3.0).abs() < 1e-10);
    }
}
