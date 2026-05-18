//! Commitment-scheme traits.

use grid_algebra::arith::ring::Ring;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

/// A commitment scheme over a ring-backed message space.
pub trait CommitmentScheme {
    /// The underlying ring/backend.
    type Ring: Ring + CanonicalSerialize + CanonicalDeserialize + Valid;
    /// The committed message type.
    type Message: CanonicalSerialize + CanonicalDeserialize + Valid;
    /// The commitment output type.
    type Commitment: CanonicalSerialize + CanonicalDeserialize + Valid;
    /// The opening type.
    type Opening: CanonicalSerialize + CanonicalDeserialize + Valid;
    /// Setup-time parameters.
    type SetupParams: CanonicalSerialize + CanonicalDeserialize + Valid;
    /// Scheme error type.
    type Error;

    /// Create a scheme instance from setup parameters.
    fn setup<Rng: grid_std::rand::Rng>(
        rng: &mut Rng,
        params: &Self::SetupParams,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Commit to a message, sampling fresh opening material.
    fn commit<Rng: grid_std::rand::Rng>(
        &self,
        message: &Self::Message,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, Self::Opening), Self::Error>;

    /// Commit using explicit opening material.
    fn commit_with_opening(
        &self,
        message: &Self::Message,
        opening: &Self::Opening,
    ) -> Result<Self::Commitment, Self::Error>;

    /// Verify a commitment/opening pair against a message.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        message: &Self::Message,
        opening: &Self::Opening,
    ) -> Result<bool, Self::Error>;
}

/// Additively homomorphic commitment operations.
pub trait HomomorphicCommitment: CommitmentScheme {
    /// Add two commitments.
    fn add_commitments(
        &self,
        lhs: &Self::Commitment,
        rhs: &Self::Commitment,
    ) -> Result<Self::Commitment, Self::Error>;

    /// Add two openings.
    fn add_openings(
        &self,
        lhs: &Self::Opening,
        rhs: &Self::Opening,
    ) -> Result<Self::Opening, Self::Error>;
}
