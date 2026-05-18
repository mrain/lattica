//! Ajtai-style lattice commitments.

mod params;

#[cfg(test)]
mod tests;

use grid_algebra::arith::ntt::NTTRing;
use grid_algebra::arith::ring::Field;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::{NormBound, NormedRing, VectorNormBound};
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::TwistedNttPoly;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::Valid;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;

use crate::error::CommitmentError;
use crate::linear::{
    PreparedLinearCommitmentKey, PreparedLinearOps, prepare_linear_key,
    recompute_linear_commitment_prepared, validate_commitment_value, validate_linear_matrices,
    validate_message, validate_opening_randomness,
};
use crate::ntt::{NttCommitmentScheme, PreparedNttMessage, PreparedNttOpening};
use crate::sampling::{CommitmentSampleRing, sample_opening_vec, sample_uniform_mat};
use crate::traits::{CommitmentScheme, HomomorphicCommitment};

pub use params::{
    AjtaiCommitment, AjtaiCommitmentKey, AjtaiCommitmentScheme, AjtaiOpening, AjtaiParams,
};

/// Public marker trait for ring backends supported by the Ajtai commitment implementation.
///
/// This bridges the current public API to the concrete internal backend requirements without
/// exposing the private helper traits used inside `grid-commit`.
#[doc(hidden)]
pub trait AjtaiCommitmentRing: CommitmentSampleRing + NormedRing + PreparedLinearOps {}

impl<R> AjtaiCommitmentRing for R where R: CommitmentSampleRing + NormedRing + PreparedLinearOps {}

/// Ajtai message cached in the shared twisted NTT wrapper.
pub type PreparedAjtaiMessage<C, const N: usize> = PreparedNttMessage<C, N>;

/// Ajtai opening cached in the shared twisted NTT wrapper.
pub type PreparedAjtaiOpening<C, const N: usize> = PreparedNttOpening<C, N>;

fn recompute_commitment_from_validated<R, B>(
    scheme: &AjtaiCommitmentScheme<R, B>,
    message: &RingVec<R>,
    opening: &AjtaiOpening<R>,
) -> Result<AjtaiCommitment<R>, CommitmentError>
where
    R: PreparedLinearOps,
{
    let value = R::recompute_linear_commitment_runtime(
        scheme.prepared_key.as_deref(),
        &scheme.key.a_msg,
        &scheme.key.a_open,
        message,
        &opening.randomness,
        scheme.params.dims,
    )?
    .value;
    Ok(AjtaiCommitment { value })
}

fn prepared_key_ref<C, const N: usize>(
    scheme: &AjtaiCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>,
) -> Option<&PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N, Uint = u64>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand
        + Valid
        + Send
        + Sync
        + 'static,
{
    scheme.prepared_key.as_deref().and_then(|cache| {
        let r = cache.downcast_ref::<PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>>();
        debug_assert!(
            r.is_some(),
            "prepared key cache type mismatch: expected PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>"
        );
        r
    })
}

impl<R: Ring, B> AjtaiCommitmentScheme<R, B> {
    /// Borrow the setup parameters for this scheme instance.
    pub fn params(&self) -> &AjtaiParams<B> {
        &self.params
    }

    /// Borrow the public commitment key for this scheme instance.
    pub fn key(&self) -> &AjtaiCommitmentKey<R> {
        &self.key
    }
}

impl<R, B> AjtaiCommitmentScheme<R, B>
where
    R: Ring + Valid,
    B: Valid,
{
    /// Rebuild a scheme instance from validated public setup artifacts.
    pub fn from_parts(
        params: AjtaiParams<B>,
        key: AjtaiCommitmentKey<R>,
    ) -> Result<Self, CommitmentError> {
        if !params.is_valid() {
            return Err(CommitmentError::InvalidParameters);
        }
        if !key.is_valid() {
            return Err(CommitmentError::InvalidParameters);
        }
        validate_linear_matrices(&key.a_msg, &key.a_open, params.dims)?;
        Ok(Self {
            params,
            key,
            prepared_key: None,
        })
    }
}

impl<R: Ring + Valid, B> AjtaiCommitmentScheme<R, B> {
    fn ensure_opening_within_bound(&self, opening: &AjtaiOpening<R>) -> Result<(), CommitmentError>
    where
        B: VectorNormBound<R>,
    {
        if self.params.opening_bound.check_vector(&opening.randomness) {
            Ok(())
        } else {
            Err(CommitmentError::OpeningNormExceeded)
        }
    }

    fn validate_public_inputs(
        &self,
        message: &RingVec<R>,
        opening: Option<&AjtaiOpening<R>>,
        commitment: Option<&AjtaiCommitment<R>>,
    ) -> Result<(), CommitmentError> {
        let dims = self.params.dims;
        validate_linear_matrices(&self.key.a_msg, &self.key.a_open, dims)?;
        validate_message(message, dims)?;
        if let Some(opening) = opening {
            if !opening.is_valid() {
                return Err(CommitmentError::InvalidOpening);
            }
            validate_opening_randomness(&opening.randomness, dims)?;
        }
        if let Some(commitment) = commitment {
            validate_commitment_value(&commitment.value, dims)?;
        }
        Ok(())
    }

    fn validate_opening_input(&self, opening: &AjtaiOpening<R>) -> Result<(), CommitmentError> {
        if !opening.is_valid() {
            return Err(CommitmentError::InvalidOpening);
        }
        validate_opening_randomness(&opening.randomness, self.params.dims)
    }
}

impl<C, const N: usize> AjtaiCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>
where
    C: Field
        + NTTRing
        + CommitmentSampleRing
        + NegacyclicMulRing<N, Uint = u64>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand
        + Valid
        + Send
        + Sync
        + 'static,
    CyclotomicPolyRing<C, N>: Valid + NormedRing,
{
    fn recompute_prepared_commitment(
        &self,
        message: &PreparedAjtaiMessage<C, N>,
        opening: &PreparedAjtaiOpening<C, N>,
    ) -> Result<AjtaiCommitment<CyclotomicPolyRing<C, N>>, CommitmentError> {
        if let Some(prepared_key) = prepared_key_ref(self) {
            return recompute_linear_commitment_prepared(
                prepared_key,
                &message.prepared,
                &opening.prepared,
                self.params.dims,
            )
            .map(|prepared| AjtaiCommitment {
                value: prepared.value,
            });
        }

        let prepared_key = prepare_linear_key(&self.key.a_msg, &self.key.a_open)?;
        recompute_linear_commitment_prepared(
            &prepared_key,
            &message.prepared,
            &opening.prepared,
            self.params.dims,
        )
        .map(|prepared| AjtaiCommitment {
            value: prepared.value,
        })
    }

    /// Prepare a coefficient-domain message once for repeated commitments over the same input.
    pub fn prepare_message(
        &self,
        message: &RingVec<CyclotomicPolyRing<C, N>>,
    ) -> Result<PreparedAjtaiMessage<C, N>, CommitmentError> {
        self.validate_public_inputs(message, None, None)?;
        PreparedAjtaiMessage::from_coeff_message(message)
    }

    /// Prepare explicit opening randomness once for repeated commitments over the same input.
    pub fn prepare_opening(
        &self,
        opening: &AjtaiOpening<CyclotomicPolyRing<C, N>>,
    ) -> Result<PreparedAjtaiOpening<C, N>, CommitmentError> {
        self.validate_opening_input(opening)?;
        PreparedAjtaiOpening::from_coeff_randomness(&opening.randomness)
    }

    /// Prepare an already-twisted message once for repeated commitments over the same input.
    pub fn prepare_ntt(
        &self,
        message: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedAjtaiMessage<C, N>, CommitmentError> {
        let prepared = PreparedAjtaiMessage::from_ntt_message(message)?;
        let dims = self.params.dims;
        validate_linear_matrices(&self.key.a_msg, &self.key.a_open, dims)?;
        validate_message(&prepared.prepared, dims)?;
        Ok(prepared)
    }

    /// Prepare already-twisted opening randomness once for repeated commitments.
    pub fn prepare_opening_ntt(
        &self,
        opening: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedAjtaiOpening<C, N>, CommitmentError> {
        let prepared = PreparedAjtaiOpening::from_ntt_randomness(opening)?;
        validate_opening_randomness(&prepared.prepared, self.params.dims)?;
        Ok(prepared)
    }

    /// Commit using preprocessed NTT-domain message/opening data.
    pub fn commit_with_opening_ntt(
        &self,
        message: &PreparedAjtaiMessage<C, N>,
        opening: &PreparedAjtaiOpening<C, N>,
    ) -> Result<AjtaiCommitment<CyclotomicPolyRing<C, N>>, CommitmentError> {
        let dims = self.params.dims;
        validate_linear_matrices(&self.key.a_msg, &self.key.a_open, dims)?;
        validate_message(&message.prepared, dims)?;
        validate_opening_randomness(&opening.prepared, dims)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Err(CommitmentError::OpeningNormExceeded);
        }
        self.recompute_prepared_commitment(message, opening)
    }

    /// Commit to a prepared NTT-domain message using fresh randomness.
    pub fn commit_ntt<Rng: grid_std::rand::Rng>(
        &self,
        message: &PreparedAjtaiMessage<C, N>,
        rng: &mut Rng,
    ) -> Result<
        (
            AjtaiCommitment<CyclotomicPolyRing<C, N>>,
            PreparedAjtaiOpening<C, N>,
        ),
        CommitmentError,
    > {
        let dims = self.params.dims;
        validate_linear_matrices(&self.key.a_msg, &self.key.a_open, dims)?;
        validate_message(&message.prepared, dims)?;
        let randomness =
            sample_opening_vec(rng, self.params.dims.opening_len, self.params.opening_eta);
        let opening = PreparedAjtaiOpening::from_coeff_randomness(&randomness)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Err(CommitmentError::OpeningNormExceeded);
        }
        let commitment = self.recompute_prepared_commitment(message, &opening)?;
        Ok((commitment, opening))
    }

    /// Verify using prepared NTT-domain message/opening data.
    pub fn verify_ntt(
        &self,
        commitment: &AjtaiCommitment<CyclotomicPolyRing<C, N>>,
        message: &PreparedAjtaiMessage<C, N>,
        opening: &PreparedAjtaiOpening<C, N>,
    ) -> Result<bool, CommitmentError> {
        let dims = self.params.dims;
        validate_linear_matrices(&self.key.a_msg, &self.key.a_open, dims)?;
        validate_message(&message.prepared, dims)?;
        validate_commitment_value(&commitment.value, dims)?;
        validate_opening_randomness(&opening.prepared, dims)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Ok(false);
        }

        let expected = self.recompute_prepared_commitment(message, opening)?;
        Ok(expected == *commitment)
    }

    /// Commit using preprocessed message/opening data.
    pub fn commit_prepared(
        &self,
        message: &PreparedAjtaiMessage<C, N>,
        opening: &PreparedAjtaiOpening<C, N>,
    ) -> Result<AjtaiCommitment<CyclotomicPolyRing<C, N>>, CommitmentError> {
        self.commit_with_opening_ntt(message, opening)
    }

    /// Verify using preprocessed message/opening data.
    pub fn verify_prepared(
        &self,
        commitment: &AjtaiCommitment<CyclotomicPolyRing<C, N>>,
        message: &PreparedAjtaiMessage<C, N>,
        opening: &PreparedAjtaiOpening<C, N>,
    ) -> Result<bool, CommitmentError> {
        self.verify_ntt(commitment, message, opening)
    }
}

impl<C, const N: usize> NttCommitmentScheme<C, N>
    for AjtaiCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>
where
    C: Field
        + NTTRing
        + CommitmentSampleRing
        + NegacyclicMulRing<N, Uint = u64>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand
        + Valid
        + Send
        + Sync
        + 'static,
    CyclotomicPolyRing<C, N>: Valid + NormedRing,
{
    fn prepare_ntt(
        &self,
        message: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttMessage<C, N>, Self::Error> {
        AjtaiCommitmentScheme::prepare_ntt(self, message)
    }

    fn prepare_opening_ntt(
        &self,
        opening: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttOpening<C, N>, Self::Error> {
        AjtaiCommitmentScheme::prepare_opening_ntt(self, opening)
    }

    fn commit_ntt<Rng: grid_std::rand::Rng>(
        &self,
        message: &PreparedNttMessage<C, N>,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, PreparedNttOpening<C, N>), Self::Error> {
        AjtaiCommitmentScheme::commit_ntt(self, message, rng)
    }

    fn commit_with_opening_ntt(
        &self,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<Self::Commitment, Self::Error> {
        AjtaiCommitmentScheme::commit_with_opening_ntt(self, message, opening)
    }

    fn verify_ntt(
        &self,
        commitment: &Self::Commitment,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<bool, Self::Error> {
        AjtaiCommitmentScheme::verify_ntt(self, commitment, message, opening)
    }
}

impl<R, B> CommitmentScheme for AjtaiCommitmentScheme<R, B>
where
    R: CommitmentSampleRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    type Ring = R;
    type Message = RingVec<R>;
    type Commitment = AjtaiCommitment<R>;
    type Opening = AjtaiOpening<R>;
    type SetupParams = AjtaiParams<B>;
    type Error = CommitmentError;

    fn setup<Rng: grid_std::rand::Rng>(
        rng: &mut Rng,
        params: &Self::SetupParams,
    ) -> Result<Self, Self::Error> {
        if !params.is_valid() {
            return Err(CommitmentError::InvalidParameters);
        }

        let dims = params.dims;
        dims.validate()?;

        let key = AjtaiCommitmentKey {
            a_msg: sample_uniform_mat(rng, dims.commitment_len, dims.message_len),
            a_open: sample_uniform_mat(rng, dims.commitment_len, dims.opening_len),
        };
        validate_linear_matrices(&key.a_msg, &key.a_open, dims)?;
        let prepared_key = R::build_linear_key_cache(&key.a_msg, &key.a_open);

        Ok(Self {
            params: params.clone(),
            key,
            prepared_key,
        })
    }

    fn commit<Rng: grid_std::rand::Rng>(
        &self,
        message: &Self::Message,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, Self::Opening), Self::Error> {
        self.validate_public_inputs(message, None, None)?;
        let randomness =
            sample_opening_vec(rng, self.params.dims.opening_len, self.params.opening_eta);
        let opening = AjtaiOpening { randomness };
        self.ensure_opening_within_bound(&opening)?;
        let commitment = recompute_commitment_from_validated(self, message, &opening)?;
        Ok((commitment, opening))
    }

    fn commit_with_opening(
        &self,
        message: &Self::Message,
        opening: &Self::Opening,
    ) -> Result<Self::Commitment, Self::Error> {
        self.validate_public_inputs(message, None, None)?;
        self.validate_opening_input(opening)?;
        self.ensure_opening_within_bound(opening)?;
        recompute_commitment_from_validated(self, message, opening)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        message: &Self::Message,
        opening: &Self::Opening,
    ) -> Result<bool, Self::Error> {
        self.validate_public_inputs(message, None, Some(commitment))?;
        self.validate_opening_input(opening)?;
        if !self.params.opening_bound.check_vector(&opening.randomness) {
            return Ok(false);
        }

        let expected = recompute_commitment_from_validated(self, message, opening)?;
        Ok(expected == *commitment)
    }
}

impl<R, B> HomomorphicCommitment for AjtaiCommitmentScheme<R, B>
where
    R: CommitmentSampleRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    fn add_commitments(
        &self,
        lhs: &Self::Commitment,
        rhs: &Self::Commitment,
    ) -> Result<Self::Commitment, Self::Error> {
        validate_commitment_value(&lhs.value, self.params.dims)?;
        validate_commitment_value(&rhs.value, self.params.dims)?;
        Ok(AjtaiCommitment {
            value: lhs.value.clone() + &rhs.value,
        })
    }

    fn add_openings(
        &self,
        lhs: &Self::Opening,
        rhs: &Self::Opening,
    ) -> Result<Self::Opening, Self::Error> {
        if !lhs.is_valid() || !rhs.is_valid() {
            return Err(CommitmentError::InvalidOpening);
        }
        validate_opening_randomness(&lhs.randomness, self.params.dims)?;
        validate_opening_randomness(&rhs.randomness, self.params.dims)?;
        let opening = AjtaiOpening {
            randomness: lhs.randomness.clone() + &rhs.randomness,
        };
        self.ensure_opening_within_bound(&opening)?;
        Ok(opening)
    }
}
