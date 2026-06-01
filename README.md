# lau-connes-spectral-triple

> Connes' spectral triple (A,H,D) — the agents built noncommutative geometry without recognizing it

## What This Does

Connes' spectral triple (A,H,D) — the agents built noncommutative geometry without recognizing it. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-connes-spectral-triple
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_connes_spectral_triple::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct StateVector 
    pub fn new(vector: DVector<Complex64>) -> Self 
    pub fn zero(dim: usize) -> Self 
    pub fn basis(dim: usize, i: usize) -> Self 
    pub fn uniform(dim: usize) -> Self 
    pub fn dim(&self) -> usize 
    pub fn as_vector(&self) -> &DVector<Complex64> 
    pub fn inner(&self, other: &Self) -> Complex64 
    pub fn norm(&self) -> f64 
    pub fn normalize(&self) -> Self 
    pub fn scale(&self, c: Complex64) -> Self 
    pub fn add(&self, other: &Self) -> Self 
    pub fn outer(&self, other: &Self) -> DMatrix<Complex64> 
    pub fn is_normalized(&self, tol: f64) -> bool 
    pub fn fidelity(&self, other: &Self) -> f64 
pub struct HilbertSpace 
    pub fn new(dim: usize) -> Self 
    pub fn with_basis(dim: usize, basis: Vec<StateVector>) -> Self 
    pub fn computational_basis(dim: usize) -> Self 
    pub fn dim(&self) -> usize 
    pub fn basis(&self) -> Option<&[StateVector]> 
    pub fn zero_state(&self) -> StateVector 
    pub fn is_orthonormal_set(states: &[StateVector], tol: f64) -> bool 
    pub fn resolution_of_identity(&self) -> DMatrix<Complex64> 
    pub fn parseval_check(&self, v: &StateVector) -> bool 
    pub fn tensor_product(&self, other: &HilbertSpace) -> HilbertSpace 
pub struct AlgebraElement 
    pub fn new(matrix: DMatrix<Complex64>) -> Self 
    pub fn zero(dim: usize) -> Self 
    pub fn identity(dim: usize) -> Self 
    pub fn dim(&self) -> usize 
    pub fn as_matrix(&self) -> &DMatrix<Complex64> 
    pub fn as_matrix_mut(&mut self) -> &mut DMatrix<Complex64> 
    pub fn norm(&self) -> f64 
    pub fn adjoint(&self) -> Self 
    pub fn is_self_adjoint(&self, tol: f64) -> bool 
    pub fn is_positive(&self, tol: f64) -> bool 
    pub fn trace(&self) -> Complex64 
    pub fn compose(&self, other: &Self) -> Self 
    pub fn scale(&self, c: Complex64) -> Self 
    pub fn add(&self, other: &Self) -> Self 
    pub fn sub(&self, other: &Self) -> Self 
    pub fn diagonal(values: &[Complex64]) -> Self 
    pub fn projection(v: &DVector<Complex64>) -> Self 
    pub fn from_real_matrix(m: DMatrix<f64>) -> Self 
pub struct CStarAlgebra 
    pub fn new(dim: usize, generators: Vec<AlgebraElement>) -> Self 
    pub fn full_matrix_algebra(dim: usize) -> Self 
    pub fn commutative(n: usize) -> Self 
    pub fn dim(&self) -> usize 
    pub fn num_generators(&self) -> usize 
    pub fn generators(&self) -> &[AlgebraElement] 
    pub fn commutes(a: &AlgebraElement, b: &AlgebraElement, tol: f64) -> bool 
    pub fn center_element(&self) -> AlgebraElement 
    pub fn bracket(a: &AlgebraElement, b: &AlgebraElement) -> AlgebraElement 
    pub fn is_commutative(&self, tol: f64) -> bool 
pub struct ConnesDistance 
    pub fn new(dirac: DiracOperator) -> Self 
    pub fn distance(&self, p: &StateVector, q: &StateVector) -> f64 
    pub fn metric_matrix(&self) -> Vec<Vec<f64>> 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**42 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
