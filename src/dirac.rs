//! Dirac operator D — the learning/differentiation operator
//!
//! The Dirac operator encodes the "derivative structure" of the noncommutative
//! space. Its commutator with elements of A gives Lipschitz information,
//! and its spectrum determines the metric.

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

use crate::algebra::AlgebraElement;
use crate::commutator::Commutator;

/// Dirac operator as a self-adjoint (possibly unbounded in infinite dim) operator.
/// In finite dimensions, represented as a Hermitian matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiracOperator {
    matrix: DMatrix<Complex64>,
    /// Eigenvalues (cached)
    eigenvalues: Option<Vec<f64>>,
}

impl DiracOperator {
    /// Create from a matrix (should be Hermitian)
    pub fn new(matrix: DMatrix<Complex64>) -> Self {
        Self { matrix, eigenvalues: None }
    }

    /// Create a flat Dirac operator: D = diag(-n/2, ..., 0, ..., n/2)
    pub fn flat(dim: usize) -> Self {
        let mut eigenvalues = Vec::new();
        let half = (dim as f64 - 1.0) / 2.0;
        for i in 0..dim {
            eigenvalues.push(half - i as f64);
        }
        let mut m = DMatrix::zeros(dim, dim);
        for (i, &ev) in eigenvalues.iter().enumerate() {
            m[(i, i)] = Complex64::new(ev, 0.0);
        }
        Self { matrix: m, eigenvalues: Some(eigenvalues) }
    }

    /// Create from eigenvalues with standard basis
    pub fn from_eigenvalues(eigenvalues: Vec<f64>) -> Self {
        let n = eigenvalues.len();
        let mut m = DMatrix::zeros(n, n);
        for (i, &ev) in eigenvalues.iter().enumerate() {
            m[(i, i)] = Complex64::new(ev, 0.0);
        }
        Self { matrix: m, eigenvalues: Some(eigenvalues) }
    }

    /// Create a Dirac operator with random spectrum
    pub fn with_spectrum(dim: usize, eigenvalues: Vec<f64>) -> Self {
        assert_eq!(eigenvalues.len(), dim);
        Self::from_eigenvalues(eigenvalues)
    }

    /// Dimension
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Reference to matrix
    pub fn as_matrix(&self) -> &DMatrix<Complex64> {
        &self.matrix
    }

    /// Operator norm
    pub fn norm(&self) -> f64 {
        self.matrix.singular_values()[0]
    }

    /// Eigenvalues (compute and cache if needed)
    pub fn eigenvalues(&mut self) -> Vec<f64> {
        if let Some(ref ev) = self.eigenvalues {
            return ev.clone();
        }
        let eigen = self.matrix.symmetric_eigenvalues();
        let ev: Vec<f64> = eigen.iter().map(|c| c.re).collect();
        self.eigenvalues = Some(ev.clone());
        ev
    }

    /// Sorted eigenvalues
    pub fn sorted_eigenvalues(&mut self) -> Vec<f64> {
        let mut ev = self.eigenvalues();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ev
    }

    /// Sign of D: D|D|^{-1}
    pub fn sign(&self) -> Self {
        let svd = self.matrix.clone().svd(true, true);
        let s = svd.singular_values;
        let v = svd.v.unwrap();
        let u = svd.u.unwrap();

        let mut sign_diag = DMatrix::zeros(self.dim(), self.dim());
        for i in 0..self.dim() {
            if s[i] > 1e-15 {
                // Determine sign from matrix diagonal
                let mid = &u.column(i) * &self.matrix * &v.column(i);
                sign_diag[(i, i)] = Complex64::new(mid.re.signum(), 0.0);
            }
        }
        Self { matrix: &u * &sign_diag * &v.adjoint(), eigenvalues: None }
    }

    /// |D| = sqrt(D* D)
    pub fn abs(&self) -> Self {
        let dtd = &self.matrix.adjoint() * &self.matrix;
        let eigen = dtd.symmetric_eigen();
        let vals: Vec<Complex64> = eigen.eigenvalues.iter()
            .map(|&v| Complex64::new(v.re.sqrt().max(0.0), 0.0))
            .collect();
        let abs_matrix = &eigen.eigenvectors * DMatrix::from_diagonal(&vals) * &eigen.eigenvectors.adjoint();
        Self { matrix: abs_matrix, eigenvalues: Some(vals.iter().map(|v| v.re).collect()) }
    }

    /// D^{-s} for spectral zeta function (returns diagonal of eigenvalues^{-s})
    pub fn power(&mut self, s: f64) -> DMatrix<Complex64> {
        let ev = self.eigenvalues();
        let n = ev.len();
        let mut result = DMatrix::zeros(n, n);
        for (i, &lambda) in ev.iter().enumerate() {
            if lambda.abs() < 1e-15 {
                result[(i, i)] = Complex64::new(0.0, 0.0);
            } else {
                result[(i, i)] = Complex64::new(lambda.abs().powf(-s), 0.0);
            }
        }
        result
    }

    /// Spectral zeta function: ζ(s) = tr(|D|^{-s})
    pub fn zeta(&mut self, s: f64) -> f64 {
        let ev = self.eigenvalues();
        ev.iter()
            .filter(|&&l| l.abs() > 1e-15)
            .map(|&l| l.abs().powf(-s))
            .sum()
    }

    /// Is self-adjoint?
    pub fn is_self_adjoint(&self, tol: f64) -> bool {
        let diff = &self.matrix - self.matrix.adjoint();
        diff.norm() < tol
    }

    /// Kernel dimension
    pub fn kernel_dim(&mut self) -> usize {
        let ev = self.eigenvalues();
        ev.iter().filter(|&&l| l.abs() < 1e-10).count()
    }

    /// Commutator [D, f] for an algebra element
    pub fn commutator_with(&self, f: &AlgebraElement) -> Commutator {
        let dm = &self.matrix;
        let fm = f.as_matrix();
        let comm = dm * fm - fm * dm;
        Commutator::new(AlgebraElement::new(comm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_flat_dirac() {
        let d = DiracOperator::flat(3);
        assert_eq!(d.dim(), 3);
        assert!(d.is_self_adjoint(1e-10));
    }

    #[test]
    fn test_dirac_eigenvalues() {
        let mut d = DiracOperator::from_eigenvalues(vec![1.0, 2.0, 3.0]);
        let ev = d.eigenvalues();
        assert_relative_eq!(ev[0], 1.0);
        assert_relative_eq!(ev[2], 3.0);
    }

    #[test]
    fn test_dirac_norm() {
        let d = DiracOperator::from_eigenvalues(vec![-2.0, 0.0, 2.0]);
        assert_relative_eq!(d.norm(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirac_zeta() {
        let mut d = DiracOperator::from_eigenvalues(vec![1.0, 2.0, 3.0]);
        let z = d.zeta(0.0);
        assert_relative_eq!(z, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dirac_power() {
        let mut d = DiracOperator::from_eigenvalues(vec![1.0, 2.0, 4.0]);
        let p = d.power(1.0);
        assert_relative_eq!(p[(0, 0)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p[(1, 1)].re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(p[(2, 2)].re, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_abs() {
        let d = DiracOperator::from_eigenvalues(vec![-3.0, 1.0, 2.0]);
        let abs_d = d.abs();
        let sv = abs_d.matrix.singular_values();
        assert_relative_eq!(sv[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_dim() {
        let mut d = DiracOperator::from_eigenvalues(vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(d.kernel_dim(), 2);
    }

    #[test]
    fn test_commutator_with() {
        let d = DiracOperator::flat(3);
        let id = AlgebraElement::identity(3);
        let comm = d.commutator_with(&id);
        // [D, I] = 0
        assert_relative_eq!(comm.element().as_matrix().norm(), 0.0, epsilon = 1e-10);
    }
}
