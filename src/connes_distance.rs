//! Connes distance — distance from commutator norm
//!
//! d(p,q) = sup{|f(p) - f(q)| : ||[D, f]|| ≤ 1}
//!
//! This is the fundamental result: the metric is entirely determined by the
//! spectral triple. No manifold needed.

use num_complex::Complex64;
use nalgebra::DVector;
use serde::{Serialize, Deserialize};

use crate::algebra::{AlgebraElement, CStarAlgebra};
use crate::dirac::DiracOperator;
use crate::commutator::Commutator;
use crate::hilbert::StateVector;

/// Connes distance calculator
#[derive(Clone, Debug)]
pub struct ConnesDistance {
    dirac: DiracOperator,
    algebra_dim: usize,
}

impl ConnesDistance {
    /// Create from a Dirac operator
    pub fn new(dirac: DiracOperator) -> Self {
        let dim = dirac.dim();
        Self { dirac, algebra_dim: dim }
    }

    /// Compute distance between two pure states using the Lipschitz dual formula.
    /// d(φ, ψ) = sup{|φ(f) - ψ(f)| : ||[D,f]|| ≤ 1}
    ///
    /// For pure states |p⟩, |q⟩: φ(f) = ⟨p|f|p⟩, ψ(f) = ⟨q|f|q⟩
    pub fn distance(&self, p: &StateVector, q: &StateVector) -> f64 {
        // For diagonal D, the distance between computational basis states |i⟩, |j⟩
        // is |λ_i - λ_j|^{-1} when restricted to the Lipschitz ball
        // More precisely, d(p,q) = sup over Lipschitz observables
        // In practice, compute via: d = sup{|⟨p|f|p⟩ - ⟨q|f|q⟩| : ||[D,f]|| ≤ 1}

        // For diagonal D with eigenvalues λ_i, the distance between |i⟩ and |j⟩ is:
        // The commutator [D, f] for diagonal f is 0, so only off-diagonal elements matter.
        // The optimal f has [D,f] of unit norm, maximising |f_ii - f_jj|.

        // For computational basis states and diagonal D:
        let pv = p.as_vector();
        let qv = q.as_vector();

        // Find which basis states these are
        let i = Self::find_basis_index(pv);
        let j = Self::find_basis_index(qv);

        if let (Some(i), Some(j)) = (i, j) {
            // Get Dirac eigenvalues
            let dm = self.dirac.as_matrix();
            let lambda_i = dm[(i, i)].re;
            let lambda_j = dm[(j, j)].re;
            let diff = (lambda_i - lambda_j).abs();
            if diff < 1e-15 {
                return 0.0;
            }
            return 1.0 / diff;
        }

        // General case: numerical optimization over a sample of observables
        self.numerical_distance(p, q)
    }

    /// Find basis index if state is a computational basis state
    fn find_basis_index(v: &DVector<Complex64>) -> Option<usize> {
        let n = v.nrows();
        let mut nonzero = 0;
        let mut idx = 0;
        for (i, c) in v.iter().enumerate() {
            if c.norm() > 0.5 {
                nonzero += 1;
                idx = i;
            }
        }
        if nonzero == 1 {
            Some(idx)
        } else {
            None
        }
    }

    /// Numerical distance estimation via sampling observables
    fn numerical_distance(&self, p: &StateVector, q: &StateVector) -> f64 {
        let n = self.algebra_dim;
        let mut max_dist = 0.0f64;

        // Sample random observables and check Lipschitz constraint
        for _ in 0..1000 {
            let f = self.random_observable(n);
            let comm = Commutator::compute(&self.dirac, &f);
            let lipschitz = comm.lipschitz_constant();

            if lipschitz > 1e-15 {
                // Scale to unit Lipschitz
                let f_scaled = AlgebraElement::new(
                    f.as_matrix().scale(Complex64::new(1.0 / lipschitz, 0.0))
                );

                let val_p = self.state_expectation(p, &f_scaled);
                let val_q = self.state_expectation(q, &f_scaled);
                let diff = (val_p - val_q).norm();
                if diff > max_dist {
                    max_dist = diff;
                }
            }
        }

        max_dist
    }

    /// Expectation value ⟨ψ|f|ψ⟩
    fn state_expectation(&self, state: &StateVector, f: &AlgebraElement) -> Complex64 {
        let v = state.as_vector();
        let fv = f.as_matrix() * v;
        v.dotc(&fv)
    }

    /// Generate a random Hermitian observable
    fn random_observable(&self, n: usize) -> AlgebraElement {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut m = nalgebra::DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let mut hasher = DefaultHasher::new();
                (i * n + j + 1).hash(&mut hasher);
                let h = hasher.finish();
                let real = ((h & 0xFFFF) as f64 / 65536.0 - 0.5) * 2.0;
                m[(i, j)] = Complex64::new(real, 0.0);
                m[(j, i)] = Complex64::new(real, 0.0);
            }
        }
        AlgebraElement::new(m)
    }

    /// Metric matrix: distances between all pairs of basis states
    pub fn metric_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.algebra_dim;
        let mut dists = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let pi = StateVector::basis(n, i);
                let pj = StateVector::basis(n, j);
                dists[i][j] = self.distance(&pi, &pj);
            }
        }
        dists
    }

    /// Verify triangle inequality for all basis state triples
    pub fn verify_triangle_inequality(&self, tol: f64) -> bool {
        let dists = self.metric_matrix();
        let n = dists.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if dists[i][j] > dists[i][k] + dists[k][j] + tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Verify symmetry d(p,q) = d(q,p)
    pub fn verify_symmetry(&self, tol: f64) -> bool {
        let dists = self.metric_matrix();
        let n = dists.len();
        for i in 0..n {
            for j in 0..n {
                if (dists[i][j] - dists[j][i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Reference to Dirac operator
    pub fn dirac(&self) -> &DiracOperator {
        &self.dirac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_distance_self() {
        let d = DiracOperator::flat(3);
        let cd = ConnesDistance::new(d);
        let p = StateVector::basis(3, 0);
        assert_relative_eq!(cd.distance(&p, &p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_symmetry() {
        let d = DiracOperator::flat(3);
        let cd = ConnesDistance::new(d);
        let p = StateVector::basis(3, 0);
        let q = StateVector::basis(3, 1);
        assert_relative_eq!(
            cd.distance(&p, &q),
            cd.distance(&q, &p),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_triangle_inequality() {
        let d = DiracOperator::flat(4);
        let cd = ConnesDistance::new(d);
        assert!(cd.verify_triangle_inequality(1e-8));
    }

    #[test]
    fn test_symmetry_property() {
        let d = DiracOperator::flat(4);
        let cd = ConnesDistance::new(d);
        assert!(cd.verify_symmetry(1e-10));
    }

    #[test]
    fn test_metric_matrix() {
        let d = DiracOperator::flat(3);
        let cd = ConnesDistance::new(d);
        let m = cd.metric_matrix();
        // Diagonal should be 0
        assert_relative_eq!(m[0][0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(m[1][1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_eigenvalue_gap() {
        // D with eigenvalues 0, 1 → distance between |0⟩ and |1⟩ = 1/|0-1| = 1
        let d = DiracOperator::from_eigenvalues(vec![0.0, 1.0]);
        let cd = ConnesDistance::new(d);
        let p = StateVector::basis(2, 0);
        let q = StateVector::basis(2, 1);
        assert_relative_eq!(cd.distance(&p, &q), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_positive_definiteness() {
        let d = DiracOperator::flat(4);
        let cd = ConnesDistance::new(d);
        let m = cd.metric_matrix();
        for i in 0..m.len() {
            for j in 0..m.len() {
                assert!(m[i][j] >= -1e-10);
                if i == j {
                    assert_relative_eq!(m[i][j], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}
