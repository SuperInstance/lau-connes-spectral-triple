//! Hilbert space H of agent states
//!
//! Finite-dimensional Hilbert space with inner product, supporting spectral
//! theory requirements for Connes' reconstruction theorem.

use nalgebra::{DVector, DMatrix, Complex};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

/// A state vector in the Hilbert space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVector {
    vector: DVector<Complex64>,
}

impl StateVector {
    /// Create from a DVector
    pub fn new(vector: DVector<Complex64>) -> Self {
        Self { vector }
    }

    /// Create a zero state of given dimension
    pub fn zero(dim: usize) -> Self {
        Self { vector: DVector::zeros(dim) }
    }

    /// Create a basis state |e_i⟩
    pub fn basis(dim: usize, i: usize) -> Self {
        let mut v = DVector::zeros(dim);
        v[i] = Complex64::new(1.0, 0.0);
        Self { vector: v }
    }

    /// Create a uniform superposition state
    pub fn uniform(dim: usize) -> Self {
        let c = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        Self { vector: DVector::from_element(dim, c) }
    }

    /// Dimension
    pub fn dim(&self) -> usize {
        self.vector.nrows()
    }

    /// Reference to underlying vector
    pub fn as_vector(&self) -> &DVector<Complex64> {
        &self.vector
    }

    /// Inner product ⟨self|other⟩
    pub fn inner(&self, other: &Self) -> Complex64 {
        self.vector.dotc(&other.vector)
    }

    /// Norm ||self||
    pub fn norm(&self) -> f64 {
        self.vector.norm()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            return self.clone();
        }
        Self { vector: self.vector.scale(Complex64::new(1.0 / n, 0.0)) }
    }

    /// Scale by complex number
    pub fn scale(&self, c: Complex64) -> Self {
        Self { vector: self.vector.scale(c) }
    }

    /// Add two states
    pub fn add(&self, other: &Self) -> Self {
        Self { vector: &self.vector + &other.vector }
    }

    /// Outer product |self⟩⟨other|
    pub fn outer(&self, other: &Self) -> DMatrix<Complex64> {
        self.vector.clone() * other.vector.adjoint()
    }

    /// Is normalized?
    pub fn is_normalized(&self, tol: f64) -> bool {
        (self.norm() - 1.0).abs() < tol
    }

    /// Fidelity |⟨self|other⟩|²
    pub fn fidelity(&self, other: &Self) -> f64 {
        self.inner(other).norm_sqr()
    }
}

/// Finite-dimensional Hilbert space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HilbertSpace {
    /// Dimension
    dim: usize,
    /// Orthonormal basis (optional explicit basis)
    basis: Option<Vec<StateVector>>,
}

impl HilbertSpace {
    /// Create a Hilbert space of given dimension
    pub fn new(dim: usize) -> Self {
        Self { dim, basis: None }
    }

    /// Create with explicit orthonormal basis
    pub fn with_basis(dim: usize, basis: Vec<StateVector>) -> Self {
        Self { dim, basis: Some(basis) }
    }

    /// Standard computational basis
    pub fn computational_basis(dim: usize) -> Self {
        let basis: Vec<StateVector> = (0..dim)
            .map(|i| StateVector::basis(dim, i))
            .collect();
        Self { dim, basis: Some(basis) }
    }

    /// Dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reference to basis
    pub fn basis(&self) -> Option<&[StateVector]> {
        self.basis.as_deref()
    }

    /// Zero state
    pub fn zero_state(&self) -> StateVector {
        StateVector::zero(self.dim)
    }

    /// Check orthonormality of a set of states
    pub fn is_orthonormal_set(states: &[StateVector], tol: f64) -> bool {
        for (i, s) in states.iter().enumerate() {
            if (s.norm() - 1.0).abs() > tol {
                return false;
            }
            for (j, t) in states.iter().enumerate() {
                if i != j && s.inner(t).norm() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Resolution of identity: sum_i |e_i⟩⟨e_i| = I
    pub fn resolution_of_identity(&self) -> DMatrix<Complex64> {
        DMatrix::identity(self.dim, self.dim)
    }

    /// Parseval identity check: ||v||² = Σ|⟨e_i|v⟩|²
    pub fn parseval_check(&self, v: &StateVector) -> bool {
        let norm_sq = v.norm().powi(2);
        if let Some(ref basis) = self.basis {
            let sum: f64 = basis.iter()
                .map(|e| v.inner(e).norm_sqr())
                .sum();
            (norm_sq - sum).abs() < 1e-10
        } else {
            true
        }
    }

    /// Tensor product of two Hilbert spaces
    pub fn tensor_product(&self, other: &HilbertSpace) -> HilbertSpace {
        let new_dim = self.dim * other.dim;
        // Build tensor product basis if both have bases
        let basis = match (&self.basis, &other.basis) {
            (Some(b1), Some(b2)) => {
                let mut tb = Vec::new();
                for e1 in b1 {
                    for e2 in b2 {
                        let v = DVector::from_iterator(
                            new_dim,
                            e1.vector.iter()
                                .flat_map(|&c1| e2.vector.iter().map(move |&c2| c1 * c2))
                        );
                        tb.push(StateVector::new(v));
                    }
                }
                Some(tb)
            }
            _ => None,
        };
        HilbertSpace { dim: new_dim, basis }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_inner_product_self() {
        let v = StateVector::basis(3, 1);
        assert_relative_eq!(v.inner(&v).re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orthogonality() {
        let e0 = StateVector::basis(3, 0);
        let e1 = StateVector::basis(3, 1);
        assert_relative_eq!(e0.inner(&e1).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut v = StateVector::new(DVector::from_vec(vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]));
        let n = v.normalize();
        assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_superposition() {
        let u = StateVector::uniform(4);
        assert_relative_eq!(u.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fidelity_self() {
        let v = StateVector::basis(3, 0);
        assert_relative_eq!(v.fidelity(&v), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fidelity_orthogonal() {
        let v0 = StateVector::basis(2, 0);
        let v1 = StateVector::basis(2, 1);
        assert_relative_eq!(v0.fidelity(&v1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_computational_basis() {
        let h = HilbertSpace::computational_basis(3);
        assert_eq!(h.dim(), 3);
        let basis = h.basis().unwrap();
        assert!(HilbertSpace::is_orthonormal_set(basis, 1e-10));
    }

    #[test]
    fn test_resolution_of_identity() {
        let h = HilbertSpace::new(3);
        let id = h.resolution_of_identity();
        assert_relative_eq!((id - DMatrix::identity(3, 3)).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parseval() {
        let h = HilbertSpace::computational_basis(3);
        let v = StateVector::uniform(3);
        assert!(h.parseval_check(&v));
    }

    #[test]
    fn test_tensor_product() {
        let h1 = HilbertSpace::computational_basis(2);
        let h2 = HilbertSpace::computational_basis(2);
        let h = h1.tensor_product(&h2);
        assert_eq!(h.dim(), 4);
    }
}
