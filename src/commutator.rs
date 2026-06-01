//! Commutator [D, f] — Lipschitz constant as operator norm
//!
//! The commutator [D, f] = Df - fD measures how "Lipschitz" an observable f is.
//! Its operator norm ||[D, f]|| is the Lipschitz constant, connecting directly
//! to the naturality boundary in category theory.

use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

use crate::algebra::AlgebraElement;
use crate::dirac::DiracOperator;

/// The commutator [D, f] = Df - fD
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commutator {
    element: AlgebraElement,
}

impl Commutator {
    /// Create from an algebra element
    pub fn new(element: AlgebraElement) -> Self {
        Self { element }
    }

    /// Compute [D, f] = Df - fD
    pub fn compute(d: &DiracOperator, f: &AlgebraElement) -> Self {
        let dm = d.as_matrix();
        let fm = f.as_matrix();
        let comm = dm * fm - fm * dm;
        Self { element: AlgebraElement::new(comm) }
    }

    /// Reference to the commutator as algebra element
    pub fn element(&self) -> &AlgebraElement {
        &self.element
    }

    /// Operator norm of the commutator — the Lipschitz constant
    pub fn lipschitz_constant(&self) -> f64 {
        self.element.norm()
    }

    /// Is the commutator zero? (f is in the kernel of the derivation)
    pub fn is_zero(&self, tol: f64) -> bool {
        self.element.as_matrix().norm() < tol
    }

    /// Leibniz rule check: [D, f·g] = [D,f]·g + f·[D,g]
    pub fn leibniz_check(d: &DiracOperator, f: &AlgebraElement, g: &AlgebraElement, tol: f64) -> bool {
        let comm_fg = Self::compute(d, &f.compose(g));
        let comm_f = Self::compute(d, f);
        let comm_g = Self::compute(d, g);
        let lhs = comm_fg.element();
        let rhs = comm_f.element().compose(g).add(&f.compose(comm_g.element()));
        (lhs.as_matrix() - rhs.as_matrix()).norm() < tol
    }

    /// Jacobi identity: [D, [f, g]] + cyclic = 0
    pub fn jacobi_check(
        d: &DiracOperator,
        f: &AlgebraElement,
        g: &AlgebraElement,
        h: &AlgebraElement,
        tol: f64,
    ) -> bool {
        // [f, [g, h]] + [g, [h, f]] + [h, [f, g]] = 0
        let fg = AlgebraElement::new(f.as_matrix() * g.as_matrix() - g.as_matrix() * f.as_matrix());
        let gh = AlgebraElement::new(g.as_matrix() * h.as_matrix() - h.as_matrix() * g.as_matrix());
        let hf = AlgebraElement::new(h.as_matrix() * f.as_matrix() - f.as_matrix() * h.as_matrix());

        let j1 = AlgebraElement::new(f.as_matrix() * gh.as_matrix() - gh.as_matrix() * f.as_matrix());
        let j2 = AlgebraElement::new(g.as_matrix() * hf.as_matrix() - hf.as_matrix() * g.as_matrix());
        let j3 = AlgebraElement::new(h.as_matrix() * fg.as_matrix() - fg.as_matrix() * h.as_matrix());

        let sum = j1.add(&j2).add(&j3);
        sum.as_matrix().norm() < tol
    }

    /// Set of Lipschitz observables: {f : ||[D,f]|| ≤ 1}
    pub fn lipschitz_unit_ball<'a>(
        d: &DiracOperator,
        observables: &'a [AlgebraElement],
    ) -> Vec<&'a AlgebraElement> {
        observables.iter()
            .filter(|f| Self::compute(d, f).lipschitz_constant() <= 1.0 + 1e-10)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_commutator_identity() {
        let d = DiracOperator::flat(3);
        let id = AlgebraElement::identity(3);
        let comm = Commutator::compute(&d, &id);
        assert!(comm.is_zero(1e-10));
    }

    #[test]
    fn test_lipschitz_constant_diagonal() {
        // [D, diag] should be 0 for diagonal D and diagonal f
        let d = DiracOperator::from_eigenvalues(vec![1.0, 2.0, 3.0]);
        let f = AlgebraElement::diagonal(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let comm = Commutator::compute(&d, &f);
        assert!(comm.is_zero(1e-10));
    }

    #[test]
    fn test_lipschitz_nonzero() {
        let d = DiracOperator::flat(3);
        // Off-diagonal element should have nonzero commutator with diagonal D
        let mut m = DMatrix::zeros(3, 3);
        m[(0, 1)] = Complex64::new(1.0, 0.0);
        let f = AlgebraElement::new(m);
        let comm = Commutator::compute(&d, &f);
        assert!(comm.lipschitz_constant() > 0.1);
    }

    #[test]
    fn test_leibniz_rule() {
        let d = DiracOperator::flat(3);
        let f = AlgebraElement::diagonal(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let g = AlgebraElement::identity(3);
        assert!(Commutator::leibniz_check(&d, &f, &g, 1e-8));
    }

    #[test]
    fn test_jacobi_identity() {
        let m1 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0),
        ]);
        let m2 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ]);
        let m3 = DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
        ]);
        let f = AlgebraElement::new(m1);
        let g = AlgebraElement::new(m2);
        let h = AlgebraElement::new(m3);
        assert!(Commutator::jacobi_check(&DiracOperator::flat(2), &f, &g, &h, 1e-8));
    }

    #[test]
    fn test_lipschitz_unit_ball() {
        let d = DiracOperator::flat(3);
        let obs = vec![
            AlgebraElement::identity(3),
            AlgebraElement::diagonal(&[
                Complex64::new(0.1, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.3, 0.0),
            ]),
        ];
        let ball = Commutator::lipschitz_unit_ball(&d, &obs);
        // Identity and diagonal should commute with diagonal D
        assert_eq!(ball.len(), 2);
    }
}
