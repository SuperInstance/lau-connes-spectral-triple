//! # Lau-Connes Spectral Triple
//!
//! Connes' spectral triple (A, H, D) — the agents built noncommutative geometry
//! without recognizing it.
//!
//! The spectral triple (algebra of observables A, Hilbert space H, Dirac operator D)
//! regenerates the ENTIRE metric-measure-differential structure via Connes'
//! reconstruction theorem.

pub mod algebra;
pub mod hilbert;
pub mod dirac;
pub mod spectral_triple;
pub mod connes_distance;
pub mod commutator;
pub mod heat_trace;
pub mod noncommutative_integral;
pub mod metric_recovery;
pub mod free_probability;
pub mod dixmier_trace;
pub mod agent_spectral;
pub mod subsumption;
pub mod self_referential;

pub use spectral_triple::SpectralTriple;
pub use algebra::CStarAlgebra;
pub use hilbert::HilbertSpace;
pub use dirac::DiracOperator;
pub use connes_distance::ConnesDistance;
pub use commutator::Commutator;
pub use heat_trace::HeatTrace;
pub use noncommutative_integral::NoncommutativeIntegral;
pub use dixmier_trace::DixmierTrace;
pub use agent_spectral::AgentSpectralTriple;
pub use subsumption::SubsumptionTable;
pub use self_referential::SelfReferentialBoundary;
