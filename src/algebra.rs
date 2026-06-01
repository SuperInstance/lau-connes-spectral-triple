//! C*-algebra of observables A
//!
//! Represents the algebra of measurable observables as bounded operators on a
//! finite-dimensional Hilbert space. Supports the *-involution, norm, and
//! operator composition required for Connes' spectral triple.

use nalgebra::{DMatrix, DVector, Complex};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

/// An element of the C*-algebra — a bounded operator represented as a matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgebraElement {
    /// Matrix representation of the operator
    matrix: DMatrix<Complex64>,
}

impl AlgebraElement {
    /// Create from a matrix
    pub fn new(matrix: DMatrix<Complex64>) -> Self {
        Self { matrix }
    }

    /// Create a zero operator of given dimension
    pub fn zero(dim: usize) -> Self {
        Self { matrix: DMatrix::zeros(dim, dim) }
    }

    /// Create an identity operator of given dimension
    pub fn identity(dim: usize) -> Self {
        Self { matrix: DMatrix::identity(dim, dim) }
    }

    /// Dimension of the underlying space
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Reference to the underlying matrix
    pub fn as_matrix(&self) -> &DMatrix<Complex64> {
        &self.matrix
    }

    /// Mutable reference to the underlying matrix
    pub fn as_matrix_mut(&mut self) -> &mut DMatrix<Complex64> {
        &mut self.matrix
    }

    /// Operator norm (largest singular value)
    pub fn norm(&self) -> f64 {
        self.matrix.singular_values()[0]
    }

    /// *-involution (conjugate transpose)
    pub fn adjoint(&self) -> Self {
        Self { matrix: self.matrix.adjoint() }
    }

    /// Check if self is self-adjoint (Hermitian)
    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        let diff = &self.matrix - self.matrix.adjoint();
        diff.norm() < tol
    }

    /// Check if self is positive semidefinite
    pub fn is_positive(&self, tol: f64) -> bool {
        if !self.is_self_adjoint(tol) {
            return false;
        }
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().all(|&v| v.re >= -tol)
    }

    /// Trace
    pub fn trace(&self) -> Complex64 {
        self.matrix.trace()
    }

    /// Operator composition (matrix multiplication)
    pub fn compose(&self, other: &Self) -> Self {
        Self { matrix: &self.matrix * &other.matrix }
    }

    /// Scale by complex number
    pub fn scale(&self, c: Complex64) -> Self {
        Self { matrix: self.matrix.scale(c) }
    }

    /// Add two operators
    pub fn add(&self, other: &Self) -> Self {
        Self { matrix: &self.matrix + &other.matrix }
    }

    /// Subtract two operators
    pub fn sub(&self, other: &Self) -> Self {
        Self { matrix: &self.matrix - &other.matrix }
    }

    /// Create a diagonal operator from eigenvalues
    pub fn diagonal(values: &[Complex64]) -> Self {
        let n = values.len();
        let mut m = DMatrix::zeros(n, n);
        for (i, &v) in values.iter().enumerate() {
            m[(i, i)] = v;
        }
        Self { matrix: m }
    }

    /// Projection onto a state vector
    pub fn projection(v: &DVector<Complex64>) -> Self {
        Self { matrix: v * v.adjoint() }
    }

    /// Create from real symmetric matrix
    pub fn from_real_matrix(m: DMatrix<f64>) -> Self {
        Self {
            matrix: m.map(|x| Complex64::new(x, 0.0)),
        }
    }
}

/// The C*-algebra of bounded operators on a finite-dimensional space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CStarAlgebra {
    /// Dimension of the underlying Hilbert space
    dim: usize,
    /// Generators of the algebra (if finite-dimensional)
    generators: Vec<AlgebraElement>,
}

impl CStarAlgebra {
    /// Create a new C*-algebra of given dimension with optional generators
    pub fn new(dim: usize, generators: Vec<AlgebraElement>) -> Self {
        Self { dim, generators }
    }

    /// Full matrix algebra M_n(C)
    pub fn full_matrix_algebra(dim: usize) -> Self {
        Self { dim, generators: vec![] }
    }

    /// Commutative algebra — functions on a finite set of points
    pub fn commutative(n: usize) -> Self {
        // Generators: diagonal matrix units
        let generators: Vec<AlgebraElement> = (0..n)
            .map(|i| {
                let mut m = DMatrix::zeros(n, n);
                m[(i, i)] = Complex64::new(1.0, 0.0);
                AlgebraElement::new(m)
            })
            .collect();
        Self { dim: n, generators }
    }

    /// Dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of generators
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// Reference to generators
    pub fn generators(&self) -> &[AlgebraElement] {
        &self.generators
    }

    /// Check if two elements commute (within tolerance)
    pub fn commutes(a: &AlgebraElement, b: &AlgebraElement, tol: f64) -> bool {
        let comm = a.compose(b).sub(&b.compose(a));
        comm.matrix.norm() < tol
    }

    /// Center of the algebra (elements commuting with everything)
    pub fn center_element(&self) -> AlgebraElement {
        AlgebraElement::identity(self.dim)
    }

    /// Bracket [a, b] = ab - ba
    pub fn bracket(a: &AlgebraElement, b: &AlgebraElement) -> AlgebraElement {
        a.compose(b).sub(&b.compose(a))
    }

    /// Is the algebra commutative?
    pub fn is_commutative(&self, tol: f64) -> bool {
        for i in 0..self.generators.len() {
            for j in (i + 1)..self.generators.len() {
                if !Self::commutes(&self.generators[i], &self.generators[j], tol) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_identity_norm() {
        let id = AlgebraElement::identity(3);
        assert_relative_eq!(id.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_involution() {
        let m = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0),
            Complex64::new(0.0, 1.0), Complex64::new(2.0, 0.0),
        ]);
        let a = AlgebraElement::new(m);
        let aa = a.adjoint().adjoint();
        assert_relative_eq!((a.matrix - aa.matrix).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_self_adjoint() {
        let m = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0),
            Complex64::new(2.0, -1.0), Complex64::new(3.0, 0.0),
        ]);
        let a = AlgebraElement::new(m);
        assert!(a.is_self_adjoint(1e-10));
    }

    #[test]
    fn test_positive_semidefinite() {
        let m = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
        ]);
        let a = AlgebraElement::new(m);
        assert!(a.is_positive(1e-10));
    }

    #[test]
    fn test_composition() {
        let a = AlgebraElement::identity(2);
        let b = AlgebraElement::identity(2);
        let c = a.compose(&b);
        assert_relative_eq!(c.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bracket_commutative() {
        let m1 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        ]);
        let m2 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ]);
        let a = AlgebraElement::new(m1);
        let b = AlgebraElement::new(m2);
        let br = CStarAlgebra::bracket(&a, &b);
        assert_relative_eq!(br.matrix.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bracket_noncommutative() {
        let m1 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
        ]);
        let m2 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ]);
        let a = AlgebraElement::new(m1);
        let b = AlgebraElement::new(m2);
        let br = CStarAlgebra::bracket(&a, &b);
        assert!(br.matrix.norm() > 0.5);
    }

    #[test]
    fn test_diagonal() {
        let d = AlgebraElement::diagonal(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
        assert_relative_eq!(d.norm(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_commutative_algebra() {
        let alg = CStarAlgebra::commutative(3);
        assert!(alg.is_commutative(1e-10));
    }

    #[test]
    fn test_full_matrix_algebra_dim() {
        let alg = CStarAlgebra::full_matrix_algebra(5);
        assert_eq!(alg.dim(), 5);
    }

    #[test]
    fn test_trace() {
        let d = AlgebraElement::diagonal(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let tr = d.trace();
        assert_relative_eq!(tr.re, 6.0, epsilon = 1e-10);
    }
}
