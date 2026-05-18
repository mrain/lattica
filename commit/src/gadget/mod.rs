//! Gadget-based lattice commitments.

mod params;

#[cfg(test)]
mod tests;

use alloc::vec;
use alloc::vec::Vec;
use core::array::from_fn;

use grid_algebra::arith::prime::{PrimeField, PrimeFieldLimb};
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::arith::z2k::Z2K;
use grid_algebra::lattice::params::VectorNormBound;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::decomposition::gadget_vector;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::Valid;
use grid_std::UniformRand;

use crate::error::CommitmentError;
use crate::linear::{
    PreparedLinearOps, validate_commitment_value, validate_message, validate_opening_randomness,
};
use crate::sampling::{CommitmentSampleRing, sample_opening_vec, sample_uniform_mat};
use crate::traits::{CommitmentScheme, HomomorphicCommitment};

pub use params::{GadgetCommitment, GadgetCommitmentScheme, GadgetOpening, GadgetParams};

fn recompute_commitment_from_validated<R, B>(
    scheme: &GadgetCommitmentScheme<R, B>,
    message: &RingVec<R>,
    opening: &GadgetOpening<R>,
) -> Result<GadgetCommitment<R>, CommitmentError>
where
    R: GadgetRing + PreparedLinearOps,
{
    // After the digit opening has been validated, the gadget-digit component has already been
    // proven to recompose to `message`, so only the random matrix multiply remains hot.
    let value = R::mul_matrix_vector_runtime(
        scheme.prepared_a_open.as_deref(),
        &scheme.a_open,
        &opening.randomness,
    )? + message;
    Ok(GadgetCommitment { value })
}

trait GadgetRing: CommitmentSampleRing {
    fn gadget_digit_count(base: u64) -> usize;
    fn gadget_vector_lifted(base: u64) -> Vec<Self>;
    fn decompose_element(value: &Self, base: u64) -> Result<Vec<Self>, CommitmentError>;
    fn recompose_element(digits: &[Self], base: u64) -> Result<Self, CommitmentError>;
    fn digit_is_canonical(value: &Self, base: u64) -> bool;
}

impl<const Q: u64, L: PrimeFieldLimb> GadgetRing for PrimeField<Q, L> {
    fn gadget_digit_count(base: u64) -> usize {
        gadget_vector::<Self>(base).len()
    }

    fn gadget_vector_lifted(base: u64) -> Vec<Self> {
        gadget_vector::<Self>(base)
    }

    fn decompose_element(value: &Self, base: u64) -> Result<Vec<Self>, CommitmentError> {
        let digits = Self::gadget_digit_count(base);
        let mut out = Vec::with_capacity(digits);
        let mut remainder = value.to_u64();
        for _ in 0..digits {
            out.push(Self::from_u64(remainder % base));
            remainder /= base;
        }
        Ok(out)
    }

    fn recompose_element(digits: &[Self], base: u64) -> Result<Self, CommitmentError> {
        if base < 2 {
            return Err(CommitmentError::InvalidParameters);
        }
        let mut acc = 0u64;
        let mut place = 1u64;
        for digit in digits {
            acc = acc.wrapping_add(digit.to_u64().wrapping_mul(place));
            place = place.wrapping_mul(base);
        }
        Ok(Self::from_u64(acc))
    }

    fn digit_is_canonical(value: &Self, base: u64) -> bool {
        value.to_u64() < base
    }
}

impl<const K: u32> GadgetRing for Z2K<K> {
    fn gadget_digit_count(base: u64) -> usize {
        gadget_vector::<Self>(base).len()
    }

    fn gadget_vector_lifted(base: u64) -> Vec<Self> {
        gadget_vector::<Self>(base)
    }

    fn decompose_element(value: &Self, base: u64) -> Result<Vec<Self>, CommitmentError> {
        let digits = Self::gadget_digit_count(base);
        let mut out = Vec::with_capacity(digits);
        let mut remainder = value.to_u64();
        for _ in 0..digits {
            out.push(Self::from_u64(remainder % base));
            remainder /= base;
        }
        Ok(out)
    }

    fn recompose_element(digits: &[Self], base: u64) -> Result<Self, CommitmentError> {
        if base < 2 {
            return Err(CommitmentError::InvalidParameters);
        }
        let mut acc = 0u64;
        let mut place = 1u64;
        for digit in digits {
            acc = acc.wrapping_add(digit.to_u64().wrapping_mul(place));
            place = place.wrapping_mul(base);
        }
        Ok(Self::from_u64(acc))
    }

    fn digit_is_canonical(value: &Self, base: u64) -> bool {
        value.to_u64() < base
    }
}

impl<C, const N: usize> GadgetRing for CyclotomicPolyRing<C, N>
where
    C: NegacyclicMulRing<N, Uint = u64> + CommitmentSampleRing + UniformRand + Valid,
{
    fn gadget_digit_count(base: u64) -> usize {
        gadget_vector::<C>(base).len()
    }

    fn gadget_vector_lifted(base: u64) -> Vec<Self> {
        gadget_vector::<C>(base)
            .into_iter()
            .map(|coeff| {
                Self::from_array(from_fn(|i| if i == 0 { coeff.clone() } else { C::zero() }))
            })
            .collect()
    }

    fn decompose_element(value: &Self, base: u64) -> Result<Vec<Self>, CommitmentError> {
        let digits = Self::gadget_digit_count(base);
        let mut digit_coeffs = vec![from_fn(|_| C::zero()); digits];
        for (coeff_idx, coeff) in value.coeffs().iter().enumerate() {
            let mut remainder = coeff.to_u64();
            for digit_coeffs in digit_coeffs.iter_mut().take(digits) {
                digit_coeffs[coeff_idx] = C::from_u64(remainder % base);
                remainder /= base;
            }
        }
        Ok(digit_coeffs.into_iter().map(Self::from_array).collect())
    }

    fn recompose_element(digits: &[Self], base: u64) -> Result<Self, CommitmentError> {
        if base < 2 {
            return Err(CommitmentError::InvalidParameters);
        }
        let mut coeffs = vec![C::zero(); N];
        let mut place = C::from_u64(1);
        let base_elem = C::from_u64(base);
        for digit in digits {
            for (dst, coeff) in coeffs.iter_mut().zip(digit.coeffs().iter()) {
                *dst += Ring::mul_ref(coeff, &place);
            }
            place *= &base_elem;
        }
        Self::try_from_coeffs(&coeffs).map_err(|_| CommitmentError::InvalidOpening)
    }

    fn digit_is_canonical(value: &Self, base: u64) -> bool {
        value.coeffs().iter().all(|coeff| coeff.to_u64() < base)
    }
}

fn build_g_matrix<R: GadgetRing, B>(params: &GadgetParams<B>) -> RingMat<R> {
    let rows = params.dims.commitment_len;
    let cols = params
        .dims
        .message_len
        .checked_mul(params.digits)
        .expect("validated gadget dimensions must not overflow");
    let gadget = R::gadget_vector_lifted(params.base);
    let mut entries = vec![R::zero(); rows * cols];
    for row in 0..rows {
        for (digit_idx, value) in gadget.iter().enumerate() {
            let col = row * params.digits + digit_idx;
            entries[row * cols + col] = value.clone();
        }
    }
    RingMat::new(rows, cols, entries)
}

fn validate_digit_opening<R: GadgetRing, B>(
    params: &GadgetParams<B>,
    message: &RingVec<R>,
    opening: &GadgetOpening<R>,
) -> Result<(), CommitmentError> {
    validate_message(message, params.dims)?;
    validate_opening_randomness(&opening.randomness, params.dims)?;
    let expected_digit_len = params
        .dims
        .message_len
        .checked_mul(params.digits)
        .ok_or(CommitmentError::InvalidParameters)?;
    if opening.digits.len() != expected_digit_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    if opening
        .digits
        .entries()
        .iter()
        .any(|digit| !R::digit_is_canonical(digit, params.base))
    {
        return Err(CommitmentError::InvalidOpening);
    }
    for i in 0..params.dims.message_len {
        let start = i * params.digits;
        let end = start + params.digits;
        let recomposed = R::recompose_element(&opening.digits.entries()[start..end], params.base)?;
        if recomposed != *message.get(i) {
            return Err(CommitmentError::InvalidOpening);
        }
    }
    Ok(())
}

fn canonicalize_digit_sum<R: GadgetRing, B>(
    params: &GadgetParams<B>,
    lhs: &RingVec<R>,
    rhs: &RingVec<R>,
) -> Result<RingVec<R>, CommitmentError> {
    let expected_digit_len = params
        .dims
        .message_len
        .checked_mul(params.digits)
        .ok_or(CommitmentError::InvalidParameters)?;
    if lhs.len() != expected_digit_len || rhs.len() != expected_digit_len {
        return Err(CommitmentError::DimensionMismatch);
    }
    if lhs
        .entries()
        .iter()
        .chain(rhs.entries().iter())
        .any(|digit| !R::digit_is_canonical(digit, params.base))
    {
        return Err(CommitmentError::InvalidOpening);
    }

    let mut digits = Vec::with_capacity(expected_digit_len);
    for i in 0..params.dims.message_len {
        let start = i * params.digits;
        let end = start + params.digits;
        let lhs_message = R::recompose_element(&lhs.entries()[start..end], params.base)?;
        let rhs_message = R::recompose_element(&rhs.entries()[start..end], params.base)?;
        let sum = lhs_message + &rhs_message;
        let decomposed = R::decompose_element(&sum, params.base)?;
        if decomposed.len() != params.digits {
            return Err(CommitmentError::InvalidParameters);
        }
        digits.extend(decomposed);
    }

    Ok(RingVec::new(digits))
}

impl<R: Ring, B> GadgetCommitmentScheme<R, B> {
    /// Borrow the setup parameters for this scheme instance.
    pub fn params(&self) -> &GadgetParams<B> {
        &self.params
    }

    /// Borrow the sampled opening matrix.
    pub fn a_open(&self) -> &RingMat<R> {
        &self.a_open
    }

    /// Borrow the deterministic gadget matrix.
    pub fn g_matrix(&self) -> &RingMat<R> {
        &self.g_matrix
    }
}

impl<R: Ring, B> GadgetCommitmentScheme<R, B> {
    fn ensure_opening_within_bound(&self, opening: &GadgetOpening<R>) -> Result<(), CommitmentError>
    where
        B: VectorNormBound<R>,
    {
        if self.params.opening_bound.check_vector(&opening.randomness) {
            Ok(())
        } else {
            Err(CommitmentError::OpeningNormExceeded)
        }
    }
}

impl<R, B> CommitmentScheme for GadgetCommitmentScheme<R, B>
where
    R: GadgetRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    type Ring = R;
    type Message = RingVec<R>;
    type Commitment = GadgetCommitment<R>;
    type Opening = GadgetOpening<R>;
    type SetupParams = GadgetParams<B>;
    type Error = CommitmentError;

    fn setup<Rng: grid_std::rand::Rng>(
        rng: &mut Rng,
        params: &Self::SetupParams,
    ) -> Result<Self, Self::Error> {
        if !params.is_valid() {
            return Err(CommitmentError::InvalidParameters);
        }
        if params.digits != R::gadget_digit_count(params.base) {
            return Err(CommitmentError::InvalidParameters);
        }
        params.dims.validate()?;

        let a_open = sample_uniform_mat(rng, params.dims.commitment_len, params.dims.opening_len);
        let g_matrix = build_g_matrix::<R, B>(params);
        let prepared_a_open = R::build_matrix_cache(&a_open);
        Ok(Self {
            params: params.clone(),
            a_open,
            g_matrix,
            prepared_a_open,
        })
    }

    fn commit<Rng: grid_std::rand::Rng>(
        &self,
        message: &Self::Message,
        rng: &mut Rng,
    ) -> Result<(Self::Commitment, Self::Opening), Self::Error> {
        validate_message(message, self.params.dims)?;
        let randomness =
            sample_opening_vec(rng, self.params.dims.opening_len, self.params.opening_eta);
        let mut digits = Vec::with_capacity(self.params.dims.message_len * self.params.digits);
        for value in message.entries() {
            let decomposed = R::decompose_element(value, self.params.base)?;
            if decomposed.len() != self.params.digits {
                return Err(CommitmentError::InvalidParameters);
            }
            digits.extend(decomposed);
        }
        let opening = GadgetOpening {
            randomness,
            digits: RingVec::new(digits),
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
        if !opening.is_valid() {
            return Err(CommitmentError::InvalidOpening);
        }
        validate_digit_opening(&self.params, message, opening)?;
        self.ensure_opening_within_bound(opening)?;
        recompute_commitment_from_validated(self, message, opening)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        message: &Self::Message,
        opening: &Self::Opening,
    ) -> Result<bool, Self::Error> {
        if !commitment.is_valid() {
            return Err(CommitmentError::InvalidMessageEncoding);
        }
        validate_commitment_value(&commitment.value, self.params.dims)?;
        if !opening.is_valid() {
            return Err(CommitmentError::InvalidOpening);
        }
        validate_digit_opening(&self.params, message, opening)?;
        if !self.params.opening_bound.check_vector(&opening.randomness) {
            return Ok(false);
        }
        let expected = recompute_commitment_from_validated(self, message, opening)?;
        Ok(expected == *commitment)
    }
}

impl<R, B> HomomorphicCommitment for GadgetCommitmentScheme<R, B>
where
    R: GadgetRing + PreparedLinearOps,
    B: VectorNormBound<R>,
{
    fn add_commitments(
        &self,
        lhs: &Self::Commitment,
        rhs: &Self::Commitment,
    ) -> Result<Self::Commitment, Self::Error> {
        validate_commitment_value(&lhs.value, self.params.dims)?;
        validate_commitment_value(&rhs.value, self.params.dims)?;
        Ok(GadgetCommitment {
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
        let expected_digit_len = self
            .params
            .dims
            .message_len
            .checked_mul(self.params.digits)
            .ok_or(CommitmentError::InvalidParameters)?;
        if lhs.digits.len() != expected_digit_len || rhs.digits.len() != expected_digit_len {
            return Err(CommitmentError::DimensionMismatch);
        }
        let opening = GadgetOpening {
            randomness: lhs.randomness.clone() + &rhs.randomness,
            digits: canonicalize_digit_sum(&self.params, &lhs.digits, &rhs.digits)?,
        };
        self.ensure_opening_within_bound(&opening)?;
        Ok(opening)
    }
}
