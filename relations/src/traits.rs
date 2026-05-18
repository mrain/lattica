//! Shared relation-system traits.

use grid_algebra::arith::ring::Ring;

use crate::RelationsError;

/// A relation system that can check whether a witness satisfies a stored instance.
pub trait ConstraintSystem<R: Ring> {
    /// Witness type accepted by this relation system.
    type Witness;

    /// Check whether `witness` satisfies `self`.
    fn is_satisfied(&self, witness: &Self::Witness) -> Result<bool, RelationsError>;
}

/// A synthesizer that can materialize a concrete relation instance and witness pair.
pub trait ConstraintSynthesizer<R: Ring> {
    /// Concrete relation-system type produced by this synthesizer.
    type System: ConstraintSystem<R, Witness = Self::Witness>;
    /// Concrete witness type.
    type Witness;

    /// Produce a concrete instance and satisfying witness.
    fn synthesize(&self) -> Result<(Self::System, Self::Witness), RelationsError>;
}
