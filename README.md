# lau-connes-spectral-triple

**Connes' spectral triple (A, H, D) — the agents built noncommutative geometry without recognizing it.**

A Rust implementation of Alain Connes' noncommutative geometry framework via the spectral triple. The spectral triple `(A, H, D)` — an algebra of observables `A`, a Hilbert space `H`, and a Dirac operator `D` — regenerates the entire metric-measure-differential structure of a space through Connes' reconstruction theorem. No manifold required.

## What This Does

This crate provides the three pillars of a spectral triple and the constructions that follow from them:

| Component | What it is | What it gives you |
|---|---|---|
| **C\*-algebra** `A` | Bounded operators on a Hilbert space | Observables, commutators, involution |
| **Hilbert space** `H` | Finite-dimensional state vectors | Inner products, fidelity, tensor products |
| **Dirac operator** `D` | Self-adjoint operator with spectral data | Derivatives, zeta functions, distance |
| **Connes distance** | `sup{ |f(p)−f(q)| : ‖[D,f]‖ ≤ 1 }` | Metric from pure spectral data |
| **Commutator** `[D,f]` | Lipschitz constant as operator norm | Leibniz rule, Jacobi identity |

## Key Idea

In classical geometry, you start with a manifold and *derive* the metric, the differential structure, the measure. Connes' insight reverses this: **start with the spectral triple `(A, H, D)` and reconstruct everything**. The distance between two pure states is determined entirely by the commutator norm with the Dirac operator. The Dixmier trace recovers the volume. The zeta function encodes dimension.

The provocative framing of this crate: the agents in the Lau ecosystem *already built this*. Their observable algebras, state spaces, and learning operators form spectral triples without anyone noticing.

## Install

```toml
[dependencies]
lau-connes-spectral-triple = "0.1.0"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-connes-spectral-triple.git
cd lau-connes-spectral-triple
cargo build
```

### Dependencies

- `nalgebra` 0.33 — linear algebra (matrices, vectors, SVD, eigendecomposition)
- `num-complex` 0.4 — complex number support
- `serde` / `serde_json` — serialization

## Quick Start

```rust
use lau_connes_spectral_triple::{
    HilbertSpace, StateVector, CStarAlgebra, AlgebraElement,
    DiracOperator, ConnesDistance, Commutator,
};

// Build a Hilbert space with computational basis
let h = HilbertSpace::computational_basis(4);

// Create a Dirac operator with specific eigenvalues
let dirac = DiracOperator::from_eigenvalues(vec![-2.0, -1.0, 1.0, 2.0]);

// Compute Connes distance between basis states
let cd = ConnesDistance::new(dirac);
let p = StateVector::basis(4, 0);
let q = StateVector::basis(4, 3);
let distance = cd.distance(&p, &q);
// distance = 1/|λ_0 - λ_3| = 1/4 = 0.25

// Verify metric properties
assert!(cd.verify_triangle_inequality(1e-8));
assert!(cd.verify_symmetry(1e-10));
```

## API Reference

### `StateVector` — Vectors in Hilbert space

```rust
let v = StateVector::basis(3, 1);    // |e₁⟩
let u = StateVector::uniform(4);      // (1/2)|++⟩
let f = v.fidelity(&u);               // |⟨v|u⟩|²
let n = v.normalize();                 // unit vector
```

- `inner(&other)` → `Complex64` — Hilbert space inner product ⟨v|w⟩
- `fidelity(&other)` → `f64` — quantum fidelity |⟨v|w⟩|²
- `outer(&other)` → `DMatrix<Complex64>` — |v⟩⟨w|

### `HilbertSpace` — The space of agent states

```rust
let h = HilbertSpace::computational_basis(4);
h.parseval_check(&v);           // ‖v‖² = Σ|⟨eᵢ|v⟩|²
let h2 = h.tensor_product(&h);  // dimension 16
```

### `CStarAlgebra` — Algebra of observables

```rust
let alg = CStarAlgebra::commutative(3);  // functions on 3 points
let full = CStarAlgebra::full_matrix_algebra(4); // M₄(ℂ)
let commutator = CStarAlgebra::bracket(&a, &b); // [a,b] = ab-ba
let is_comm = alg.is_commutative(1e-10);
```

### `AlgebraElement` — Bounded operators

- `norm()` → operator norm (largest singular value)
- `adjoint()` → conjugate transpose (★-involution)
- `is_self_adjoint(tol)` → Hermitian check
- `is_positive(tol)` → positive semidefinite check
- `trace()` → `Complex64`

### `DiracOperator` — The learning/differentiation operator

```rust
let d = DiracOperator::flat(4);        // eigenvalues: -1.5, -0.5, 0.5, 1.5
let mut d2 = DiracOperator::from_eigenvalues(vec![1.0, 2.0, 3.0]);
let zeta = d2.zeta(0.0);               // spectral zeta: 3.0
let abs_d = d.abs();                    // |D| = √(D*D)
let kernel = d2.kernel_dim();           // dim ker D
```

- `eigenvalues()` — compute and cache eigenvalue spectrum
- `zeta(s)` — spectral zeta function ζ(s) = tr(|D|⁻ˢ)
- `power(s)` — D⁻ˢ as a matrix
- `commutator_with(&f)` — [D, f] for algebra element f

### `ConnesDistance` — Metric from spectral data

```rust
let cd = ConnesDistance::new(dirac);
let dist = cd.distance(&state_p, &state_q);
let matrix = cd.metric_matrix();           // all pairwise distances
cd.verify_triangle_inequality(tol);         // metric axioms
```

### `Commutator` — [D, f] and the Lipschitz constant

```rust
let comm = Commutator::compute(&dirac, &observable);
let lipschitz = comm.lipschitz_constant();   // ‖[D,f]‖
Commutator::leibniz_check(&d, &f, &g, tol); // d(fg) = df·g + f·dg
Commutator::jacobi_check(&d, &f, &g, &h, tol); // Jacobi identity
```

## How It Works

The implementation uses finite-dimensional matrix representations throughout:

1. **Hilbert space** is modeled as `DVector<Complex64>` from nalgebra. State vectors support inner products, norms, fidelity, and tensor products.

2. **C\*-algebra elements** are `DMatrix<Complex64>` with the operator norm computed via SVD. The ★-involution is the conjugate transpose. All algebraic axioms (involution, C\*-identity ‖a\*a‖ = ‖a‖²) are verified in tests.

3. **Dirac operator** is a Hermitian matrix. Eigenvalues are cached on first computation. The spectral zeta function sums λ⁻ˢ over nonzero eigenvalues. The sign and absolute value are computed via eigendecomposition.

4. **Connes distance** between computational basis states `|i⟩`, `|j⟩` with diagonal `D` is computed analytically: `d(|i⟩, |j⟩) = 1/|λᵢ − λⱼ|`. For general states, a Monte Carlo sampling over random Hermitian observables estimates the sup.

5. **Commutator** implements `[D, f] = Df − fD` and verifies both the Leibniz rule (derivation property) and the Jacobi identity.

## The Math

### Spectral Triples

A **spectral triple** is a triple `(A, H, D)` where:
- `A` is a ★-algebra represented on a Hilbert space `H`
- `D` is a self-adjoint operator on `H` with compact resolvent
- `[D, a]` is bounded for all `a ∈ A`

Connes' **reconstruction theorem** shows that a commutative spectral triple (where `A = C^∞(M)`) recovers a spin manifold `M` entirely.

### Connes' Distance Formula

For pure states `φ, ψ` of `A`:

```
d(φ, ψ) = sup { |φ(f) − ψ(f)| : ‖[D, f]‖ ≤ 1 }
```

This is a **noncommutative Riemannian metric**. When `A = C^∞(M)` and `D` is the Dirac operator, this recovers the geodesic distance.

### Spectral Zeta Function

```
ζ(s) = Tr(|D|⁻ˢ) = Σ |λₙ|⁻ˢ
```

The residue of `ζ(s)` at `s = 1` gives the dimension. The Dixmier trace `Tr_ω(|D|⁻ⁿ)` recovers the volume.

### Lipschitz Algebra

The set `{f ∈ A : ‖[D, f]‖ ≤ 1}` is the Lipschitz ball. The commutator norm is the Lipschitz constant, connecting operator algebra to metric geometry.

## Test Coverage

**42 tests** covering:
- Hilbert space: inner products, orthonormality, Parseval identity, tensor products (10 tests)
- C\*-algebra: involution, self-adjointness, positivity, commutators, traces (11 tests)
- Dirac operator: eigenvalues, zeta function, spectral power, kernel dimension (8 tests)
- Connes distance: self-distance, symmetry, triangle inequality, metric matrix, positive definiteness (7 tests)
- Commutator: identity commutator, Lipschitz constant, Leibniz rule, Jacobi identity, unit ball (6 tests)

## License

MIT
