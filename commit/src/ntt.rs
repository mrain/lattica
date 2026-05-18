//! Shared NTT-native commitment interfaces and wrapper types.

use grid_algebra::arith::ntt::NTTRing;
use grid_algebra::arith::ring::Field;
use grid_algebra::lattice::params::{NormBound, NormStats, NormedRing};
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_algebra::poly::{TwistedNttPoly, finish_twisted_ring_vec, prepare_twisted_ring_vec};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::error::CommitmentError;
use crate::traits::CommitmentScheme;

/// A validated message cached in the twisted NTT domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedNttMessage<C, const N: usize>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    pub(crate) prepared: RingVec<TwistedNttPoly<C, N>>,
}

impl<C, const N: usize> PreparedNttMessage<C, N>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    pub(crate) fn from_coeff_message(
        message: &RingVec<CyclotomicPolyRing<C, N>>,
    ) -> Result<Self, CommitmentError> {
        Ok(Self {
            prepared: prepare_twisted_ring_vec(message)
                .map_err(|_| CommitmentError::InvalidParameters)?,
        })
    }

    pub(crate) fn from_ntt_message(
        message: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<Self, CommitmentError> {
        Ok(Self {
            prepared: message.clone(),
        })
    }

    /// Recover the canonical coefficient-domain message on demand.
    pub fn finish(&self) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError> {
        finish_twisted_ring_vec(&self.prepared).map_err(|_| CommitmentError::InvalidParameters)
    }
}

/// A validated opening cached in the twisted NTT domain, together with exact coefficient norms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedNttOpening<C, const N: usize>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    pub(crate) prepared: RingVec<TwistedNttPoly<C, N>>,
    pub(crate) exact_norms: NormStats,
}

impl<C, const N: usize> PreparedNttOpening<C, N>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
    CyclotomicPolyRing<C, N>: NormedRing,
{
    pub(crate) fn from_coeff_randomness(
        randomness: &RingVec<CyclotomicPolyRing<C, N>>,
    ) -> Result<Self, CommitmentError> {
        Ok(Self {
            prepared: prepare_twisted_ring_vec(randomness)
                .map_err(|_| CommitmentError::InvalidParameters)?,
            exact_norms: NormStats::compute(randomness),
        })
    }

    pub(crate) fn from_ntt_randomness(
        randomness: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<Self, CommitmentError> {
        let coeff_randomness =
            finish_twisted_ring_vec(randomness).map_err(|_| CommitmentError::InvalidParameters)?;
        let exact_norms = NormStats::compute(&coeff_randomness);
        Ok(Self {
            prepared: randomness.clone(),
            exact_norms,
        })
    }

    pub(crate) fn within_bound(&self, bound: &NormBound) -> bool {
        self.exact_norms.l2_sq <= bound.max_l2_sq && self.exact_norms.linf <= bound.max_linf
    }

    /// Recover the canonical coefficient-domain opening randomness on demand.
    pub fn finish_randomness(&self) -> Result<RingVec<CyclotomicPolyRing<C, N>>, CommitmentError> {
        finish_twisted_ring_vec(&self.prepared).map_err(|_| CommitmentError::InvalidParameters)
    }

    /// Borrow the exact coefficient-domain norms derived for this opening.
    pub fn norm_stats(&self) -> NormStats {
        self.exact_norms
    }
}

/// Extension trait for commitment schemes that accept NTT-domain polynomial inputs directly.
pub trait NttCommitmentScheme<C, const N: usize>:
    CommitmentScheme<Ring = CyclotomicPolyRing<C, N>>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    /// Validate and wrap an already-twisted message for repeated use.
    fn prepare_ntt(
        &self,
        message: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttMessage<C, N>, Self::Error>;

    /// Validate and wrap an already-twisted opening for repeated use.
    fn prepare_opening_ntt(
        &self,
        opening: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttOpening<C, N>, Self::Error>;

    /// Commit to an already-prepared NTT-domain message using fresh randomness.
    fn commit_ntt<Rng: grid_std::rand::Rng>(
        &self,
        message: &PreparedNttMessage<C, N>,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, PreparedNttOpening<C, N>), Self::Error>;

    /// Commit using explicit prepared NTT-domain opening material.
    fn commit_with_opening_ntt(
        &self,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<Self::Commitment, Self::Error>;

    /// Verify against prepared NTT-domain message and opening material.
    fn verify_ntt(
        &self,
        commitment: &Self::Commitment,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<bool, Self::Error>;
}
