//! Shared trait definitions for LaBRADOR.

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::poly::ring::NegacyclicMulRing;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;

/// Combined trait for the LaBRADOR proof ring.
///
/// Bundles the bounds required by the proof ring so that function signatures
/// don't repeat the same wall of traits.
pub trait LabradorProofRing<const N: usize>:
    IntegerRing<Canonical = u64>
    + NegacyclicMulRing<N>
    + UniformRand
    + CanonicalSerialize
    + CanonicalDeserialize
{
}

/// Blanket implementation: any ring that satisfies the bounds
/// automatically gets `LabradorProofRing`.
impl<R, const N: usize> LabradorProofRing<N> for R where
    R: IntegerRing<Canonical = u64>
        + NegacyclicMulRing<N>
        + UniformRand
        + CanonicalSerialize
        + CanonicalDeserialize
{
}
