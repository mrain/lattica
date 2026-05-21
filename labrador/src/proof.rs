//! Multi-level LaBRADOR proof structure (Phase 5).
//!
//! Holds the proof data for all recursion levels plus the final last-level proof.

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::poly::ring::NegacyclicMulRing;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;

use crate::last_level::LastLevelProof;
use crate::main_protocol::step_prover::LevelProof;

/// Complete multi-level LaBRADOR proof.
///
/// `levels` contains the proof data for each main recursion step (§5.2).
/// `last` contains the proof data for the final recursion level (§5.6).
///
/// The number of levels is determined by the parameter configuration and
/// the rate at which the witness shrinks across recursion steps.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct LabradorProof<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N> + UniformRand,
{
    /// Proof data for each main recursion level.
    pub levels: Vec<LevelProof<R, N>>,
    /// Proof data for the final recursion level.
    pub last: LastLevelProof<R, N>,
}

impl<R, const N: usize> LabradorProof<R, N>
where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + UniformRand
        + CanonicalSerialize
        + CanonicalDeserialize,
{
    /// Returns the total number of recursion levels in this proof.
    pub fn num_levels(&self) -> usize {
        self.levels.len() + 1
    }
}
