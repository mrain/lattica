//! BDLOP-style lattice commitments.

mod params;

#[cfg(test)]
mod tests;

use grid_algebra::arith::ntt::NTTRing;
use grid_algebra::arith::ring::Field;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::{NormBound, NormedRing, VectorNormBound};
use grid_algebra::lattice::types::RingMat;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::TwistedNttPoly;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_serialize::Valid;
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};
use grid_std::UniformRand;

use crate::error::CommitmentError;
use crate::linear::{
    PreparedLinearCommitmentKey, PreparedLinearOps, mul_prepared_matrix_vector, prepare_linear_key,
    prepare_poly_matrix, recompute_linear_commitment_prepared, validate_commitment_value,
    validate_linear_matrices, validate_message, validate_opening_randomness,
};
use crate::ntt::{NttCommitmentScheme, PreparedNttMessage, PreparedNttOpening};
use crate::sampling::{CommitmentSampleRing, sample_opening_vec, sample_uniform_mat};
use crate::traits::{CommitmentScheme, HomomorphicCommitment};

pub use params::{
    BdlopCommitment, BdlopCommitmentKey, BdlopCommitmentScheme, BdlopOpening, BdlopParams,
};

fn recompute_commitment_from_validated<R, B>(
    scheme: &BdlopCommitmentScheme<R, B>,
    message: &RingVec<R>,
    opening: &BdlopOpening<R>,
) -> Result<BdlopCommitment<R>, CommitmentError>
where
    R: PreparedLinearOps,
{
    let (u, v) = R::recompute_bdlop_parts_runtime(
        scheme.prepared_mask.as_deref(),
        &scheme.key.a_mask,
        scheme.prepared_key.as_deref(),
        &scheme.key.a_msg,
        &scheme.key.a_open,
        message,
        &opening.randomness,
        scheme.params.dims,
    )?;
    Ok(BdlopCommitment { u, v })
}

fn validate_bdlop_key<R: Ring>(
    key: &BdlopCommitmentKey<R>,
    dims: crate::linear::CommitmentDimensions,
) -> Result<(), CommitmentError> {
    validate_linear_matrices(&key.a_msg, &key.a_open, dims)?;
    if key.a_mask.rows() != dims.commitment_len || key.a_mask.cols() != dims.opening_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    Ok(())
}

fn prepared_key_ref<C, const N: usize>(
    scheme: &BdlopCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>,
) -> Option<&PreparedLinearCommitmentKey<TwistedNttPoly<C, N>>>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N, Canonical = u64>
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

fn prepared_mask_ref<C, const N: usize>(
    scheme: &BdlopCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>,
) -> Option<&RingMat<TwistedNttPoly<C, N>>>
where
    C: Field
        + NTTRing
        + NegacyclicMulRing<N, Canonical = u64>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand
        + Valid
        + Send
        + Sync
        + 'static,
{
    scheme.prepared_mask.as_deref().and_then(|cache| {
        let r = cache.downcast_ref::<RingMat<TwistedNttPoly<C, N>>>();
        debug_assert!(
            r.is_some(),
            "prepared mask cache type mismatch: expected RingMat<TwistedNttPoly<C, N>>"
        );
        r
    })
}

impl<R: Ring, B> BdlopCommitmentScheme<R, B> {
    /// Borrow the setup parameters for this scheme instance.
    pub fn params(&self) -> &BdlopParams<B> {
        &self.params
    }

    /// Borrow the public commitment key for this scheme instance.
    pub fn key(&self) -> &BdlopCommitmentKey<R> {
        &self.key
    }
}

impl<R: Ring + Valid, B> BdlopCommitmentScheme<R, B> {
    fn ensure_opening_within_bound(&self, opening: &BdlopOpening<R>) -> Result<(), CommitmentError>
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
        opening: Option<&BdlopOpening<R>>,
        commitment: Option<&BdlopCommitment<R>>,
    ) -> Result<(), CommitmentError> {
        let dims = self.params.dims;
        validate_bdlop_key(&self.key, dims)?;
        validate_message(message, dims)?;
        if let Some(opening) = opening {
            if !opening.is_valid() {
                return Err(CommitmentError::InvalidOpening);
            }
            validate_opening_randomness(&opening.randomness, dims)?;
        }
        if let Some(commitment) = commitment {
            if !commitment.is_valid() {
                return Err(CommitmentError::InvalidMessageEncoding);
            }
            if commitment.u.len() != dims.commitment_len {
                return Err(CommitmentError::DimensionMismatch);
            }
            validate_commitment_value(&commitment.v, dims)?;
        }
        Ok(())
    }

    fn validate_opening_input(&self, opening: &BdlopOpening<R>) -> Result<(), CommitmentError> {
        if !opening.is_valid() {
            return Err(CommitmentError::InvalidOpening);
        }
        validate_opening_randomness(&opening.randomness, self.params.dims)
    }
}

impl<C, const N: usize> BdlopCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>
where
    C: Field
        + NTTRing
        + CommitmentSampleRing
        + NegacyclicMulRing<N, Canonical = u64>
        + CanonicalSerialize
        + CanonicalDeserialize
        + UniformRand
        + Valid
        + Send
        + Sync
        + 'static,
    CyclotomicPolyRing<C, N>: Valid + NormedRing,
{
    fn recompute_ntt_commitment(
        &self,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<BdlopCommitment<CyclotomicPolyRing<C, N>>, CommitmentError> {
        let u = if let Some(prepared_mask) = prepared_mask_ref(self) {
            mul_prepared_matrix_vector(prepared_mask, &opening.prepared)?
        } else {
            let prepared_mask = prepare_poly_matrix(&self.key.a_mask)?;
            mul_prepared_matrix_vector(&prepared_mask, &opening.prepared)?
        };

        let v = if let Some(prepared_key) = prepared_key_ref(self) {
            recompute_linear_commitment_prepared(
                prepared_key,
                &message.prepared,
                &opening.prepared,
                self.params.dims,
            )?
            .value
        } else {
            let prepared_key = prepare_linear_key(&self.key.a_msg, &self.key.a_open)?;
            recompute_linear_commitment_prepared(
                &prepared_key,
                &message.prepared,
                &opening.prepared,
                self.params.dims,
            )?
            .value
        };

        Ok(BdlopCommitment { u, v })
    }

    /// Prepare an already-twisted message once for repeated commitments over the same input.
    pub fn prepare_ntt(
        &self,
        message: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttMessage<C, N>, CommitmentError> {
        let prepared = PreparedNttMessage::from_ntt_message(message)?;
        let dims = self.params.dims;
        validate_bdlop_key(&self.key, dims)?;
        validate_message(&prepared.prepared, dims)?;
        Ok(prepared)
    }

    /// Prepare already-twisted opening randomness once for repeated commitments.
    pub fn prepare_opening_ntt(
        &self,
        opening: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttOpening<C, N>, CommitmentError> {
        let prepared = PreparedNttOpening::from_ntt_randomness(opening)?;
        validate_opening_randomness(&prepared.prepared, self.params.dims)?;
        Ok(prepared)
    }

    /// Commit using preprocessed NTT-domain message/opening data.
    pub fn commit_with_opening_ntt(
        &self,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<BdlopCommitment<CyclotomicPolyRing<C, N>>, CommitmentError> {
        let dims = self.params.dims;
        validate_bdlop_key(&self.key, dims)?;
        validate_message(&message.prepared, dims)?;
        validate_opening_randomness(&opening.prepared, dims)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Err(CommitmentError::OpeningNormExceeded);
        }
        self.recompute_ntt_commitment(message, opening)
    }

    /// Commit to a prepared NTT-domain message using fresh randomness.
    pub fn commit_ntt<Rng: grid_std::rand::Rng>(
        &self,
        message: &PreparedNttMessage<C, N>,
        rng: &mut Rng,
    ) -> Result<
        (
            BdlopCommitment<CyclotomicPolyRing<C, N>>,
            PreparedNttOpening<C, N>,
        ),
        CommitmentError,
    > {
        let dims = self.params.dims;
        validate_bdlop_key(&self.key, dims)?;
        validate_message(&message.prepared, dims)?;
        let randomness =
            sample_opening_vec(rng, self.params.dims.opening_len, self.params.opening_eta);
        let opening = PreparedNttOpening::from_coeff_randomness(&randomness)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Err(CommitmentError::OpeningNormExceeded);
        }
        let commitment = self.recompute_ntt_commitment(message, &opening)?;
        Ok((commitment, opening))
    }

    /// Verify using prepared NTT-domain message/opening data.
    pub fn verify_ntt(
        &self,
        commitment: &BdlopCommitment<CyclotomicPolyRing<C, N>>,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<bool, CommitmentError> {
        let dims = self.params.dims;
        validate_bdlop_key(&self.key, dims)?;
        validate_message(&message.prepared, dims)?;
        if !commitment.is_valid() {
            return Err(CommitmentError::InvalidMessageEncoding);
        }
        if commitment.u.len() != dims.commitment_len {
            return Err(CommitmentError::DimensionMismatch);
        }
        validate_commitment_value(&commitment.v, dims)?;
        validate_opening_randomness(&opening.prepared, dims)?;
        if !opening.within_bound(&self.params.opening_bound) {
            return Ok(false);
        }

        let expected = self.recompute_ntt_commitment(message, opening)?;
        Ok(expected == *commitment)
    }
}

impl<C, const N: usize> NttCommitmentScheme<C, N>
    for BdlopCommitmentScheme<CyclotomicPolyRing<C, N>, NormBound>
where
    C: Field
        + NTTRing
        + CommitmentSampleRing
        + NegacyclicMulRing<N, Canonical = u64>
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
        BdlopCommitmentScheme::prepare_ntt(self, message)
    }

    fn prepare_opening_ntt(
        &self,
        opening: &RingVec<TwistedNttPoly<C, N>>,
    ) -> Result<PreparedNttOpening<C, N>, Self::Error> {
        BdlopCommitmentScheme::prepare_opening_ntt(self, opening)
    }

    fn commit_ntt<Rng: grid_std::rand::Rng>(
        &self,
        message: &PreparedNttMessage<C, N>,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, PreparedNttOpening<C, N>), Self::Error> {
        BdlopCommitmentScheme::commit_ntt(self, message, rng)
    }

    fn commit_with_opening_ntt(
        &self,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<Self::Commitment, Self::Error> {
        BdlopCommitmentScheme::commit_with_opening_ntt(self, message, opening)
    }

    fn verify_ntt(
        &self,
        commitment: &Self::Commitment,
        message: &PreparedNttMessage<C, N>,
        opening: &PreparedNttOpening<C, N>,
    ) -> Result<bool, Self::Error> {
        BdlopCommitmentScheme::verify_ntt(self, commitment, message, opening)
    }
}

impl<R, B> CommitmentScheme for BdlopCommitmentScheme<R, B>
where
    R: CommitmentSampleRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    type Ring = R;
    type Message = RingVec<R>;
    type Commitment = BdlopCommitment<R>;
    type Opening = BdlopOpening<R>;
    type SetupParams = BdlopParams<B>;
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

        let key = BdlopCommitmentKey {
            a_mask: sample_uniform_mat(rng, dims.commitment_len, dims.opening_len),
            a_msg: sample_uniform_mat(rng, dims.commitment_len, dims.message_len),
            a_open: sample_uniform_mat(rng, dims.commitment_len, dims.opening_len),
        };
        validate_bdlop_key(&key, dims)?;
        let prepared_key = R::build_linear_key_cache(&key.a_msg, &key.a_open);
        let prepared_mask = R::build_matrix_cache(&key.a_mask);

        Ok(Self {
            params: params.clone(),
            key,
            prepared_key,
            prepared_mask,
        })
    }

    fn commit<Rng: grid_std::rand::Rng>(
        &self,
        message: &Self::Message,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, Self::Opening), Self::Error> {
        self.validate_public_inputs(message, None, None)?;
        let opening = BdlopOpening {
            randomness: sample_opening_vec(
                rng,
                self.params.dims.opening_len,
                self.params.opening_eta,
            ),
        };
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

impl<R, B> HomomorphicCommitment for BdlopCommitmentScheme<R, B>
where
    R: CommitmentSampleRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    fn add_commitments(
        &self,
        lhs: &Self::Commitment,
        rhs: &Self::Commitment,
    ) -> Result<Self::Commitment, Self::Error> {
        self.validate_public_inputs(
            &RingVec::zero(self.params.dims.message_len),
            None,
            Some(lhs),
        )?;
        self.validate_public_inputs(
            &RingVec::zero(self.params.dims.message_len),
            None,
            Some(rhs),
        )?;
        Ok(BdlopCommitment {
            u: lhs.u.clone() + &rhs.u,
            v: lhs.v.clone() + &rhs.v,
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
        let opening = BdlopOpening {
            randomness: lhs.randomness.clone() + &rhs.randomness,
        };
        self.ensure_opening_within_bound(&opening)?;
        Ok(opening)
    }
}
