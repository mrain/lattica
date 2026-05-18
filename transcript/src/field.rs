//! Field-native transcript traits and framing helpers.

use alloc::vec::Vec;

use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
use grid_algebra::arith::ring::Field;

use crate::TranscriptError;
use crate::canonical::{CanonicalTranscriptEncode, CanonicalTranscriptEncoder};

const APPEND_TAG: u64 = 0;
const CHALLENGE_TAG: u64 = 1;
const MAX_ENCODED_COUNT: usize = u32::MAX as usize;

/// A field approved for the field-native transcript surface.
pub trait TranscriptField: Field {
    /// Maximum number of raw bytes that may be packed losslessly into one field element.
    const PACKED_BYTES: usize;
}

impl TranscriptField for PrimeField<GOLDILOCKS_MODULUS> {
    const PACKED_BYTES: usize = 7;
}

pub(crate) fn encode_count_u64(count: usize) -> Result<u64, TranscriptError> {
    if count > MAX_ENCODED_COUNT {
        return Err(TranscriptError::LengthOverflow);
    }
    Ok(count as u64)
}

pub(crate) fn encode_count<F: TranscriptField>(count: usize) -> Result<F, TranscriptError> {
    Ok(F::from_u64(encode_count_u64(count)?))
}

pub(crate) fn pack_bytes<F: TranscriptField>(bytes: &[u8]) -> Vec<F> {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(F::PACKED_BYTES));
    for chunk in bytes.chunks(F::PACKED_BYTES) {
        let mut word = 0u64;
        for (offset, byte) in chunk.iter().enumerate() {
            word |= (*byte as u64) << (offset * 8);
        }
        out.push(F::from_u64(word));
    }
    out
}

pub(crate) fn frame_append_elements<F: TranscriptField>(
    label: &'static [u8],
    values: &[F],
) -> Result<Vec<F>, TranscriptError> {
    if label.is_empty() {
        return Err(TranscriptError::EmptyLabel);
    }

    let label_chunks = pack_bytes::<F>(label);
    let mut out = Vec::with_capacity(4 + label_chunks.len() + values.len());
    out.push(F::from_u64(APPEND_TAG));
    out.push(encode_count::<F>(label.len())?);
    out.push(encode_count::<F>(label_chunks.len())?);
    out.extend(label_chunks);
    out.push(encode_count::<F>(values.len())?);
    out.extend_from_slice(values);
    Ok(out)
}

pub(crate) fn frame_challenge_request<F: TranscriptField>(
    label: &'static [u8],
    out_len: usize,
) -> Result<Vec<F>, TranscriptError> {
    if label.is_empty() {
        return Err(TranscriptError::EmptyLabel);
    }
    if out_len == 0 {
        return Err(TranscriptError::EmptyChallenge);
    }

    let label_chunks = pack_bytes::<F>(label);
    let mut out = Vec::with_capacity(4 + label_chunks.len());
    out.push(F::from_u64(CHALLENGE_TAG));
    out.push(encode_count::<F>(label.len())?);
    out.push(encode_count::<F>(label_chunks.len())?);
    out.extend(label_chunks);
    out.push(encode_count::<F>(out_len)?);
    Ok(out)
}

/// Shared field-native Fiat-Shamir transcript interface.
pub trait FieldTranscript<F: TranscriptField> {
    /// Low-level hook to append elements that have already been field-framed by the caller.
    fn append_preframed_elements(&mut self, elements: &[F]) -> Result<(), TranscriptError>;

    /// Low-level hook to derive challenge elements after the caller has field-framed the request.
    fn challenge_preframed_elements(
        &mut self,
        request_frame: &[F],
        out_len: usize,
    ) -> Result<Vec<F>, TranscriptError>;

    /// Append a slice of field elements using the shared framing rules.
    fn append_elements(
        &mut self,
        label: &'static [u8],
        values: &[F],
    ) -> Result<(), TranscriptError> {
        let framed = frame_append_elements(label, values)?;
        self.append_preframed_elements(&framed)
    }

    /// Append a single field element using the shared framing rules.
    fn append_element(&mut self, label: &'static [u8], value: &F) -> Result<(), TranscriptError> {
        self.append_elements(label, core::slice::from_ref(value))
    }

    /// Derive `len` field challenge elements using the shared framing rules.
    fn challenge_elements(
        &mut self,
        label: &'static [u8],
        len: usize,
    ) -> Result<Vec<F>, TranscriptError> {
        let request = frame_challenge_request::<F>(label, len)?;
        self.challenge_preframed_elements(&request, len)
    }

    /// Derive a single field challenge element using the shared framing rules.
    fn challenge_element(&mut self, label: &'static [u8]) -> Result<F, TranscriptError> {
        let mut values = self.challenge_elements(label, 1)?;
        Ok(values
            .pop()
            .expect("challenge_elements(1) returns one value"))
    }

    /// Append a structured value using the canonical field-native encoding surface.
    fn append_encoded<T: CanonicalTranscriptEncode<F>>(
        &mut self,
        label: &'static [u8],
        value: &T,
    ) -> Result<(), TranscriptError>
    where
        Self: Sized,
    {
        let mut encoder = CanonicalTranscriptEncoder::new(self, label);
        value.encode(label, &mut encoder)?;
        encoder.finish()
    }
}

#[cfg(test)]
mod tests {
    use grid_algebra::arith::ring::IntegerRing;

    use super::*;

    type Goldilocks = PrimeField<GOLDILOCKS_MODULUS>;

    #[derive(Default)]
    struct RecordingFieldTranscript {
        frames: Vec<Vec<Goldilocks>>,
        counter: u64,
    }

    impl FieldTranscript<Goldilocks> for RecordingFieldTranscript {
        fn append_preframed_elements(
            &mut self,
            elements: &[Goldilocks],
        ) -> Result<(), TranscriptError> {
            self.frames.push(elements.to_vec());
            Ok(())
        }

        fn challenge_preframed_elements(
            &mut self,
            request_frame: &[Goldilocks],
            out_len: usize,
        ) -> Result<Vec<Goldilocks>, TranscriptError> {
            self.frames.push(request_frame.to_vec());
            let seed = self.counter;
            self.counter += 1;
            Ok((0..out_len)
                .map(|offset| Goldilocks::from_u64(seed + offset as u64))
                .collect())
        }
    }

    #[test]
    fn test_append_and_challenge_frames_are_domain_separated() {
        let append = frame_append_elements::<Goldilocks>(
            b"msg",
            &[Goldilocks::from_u64(1), Goldilocks::from_u64(2)],
        )
        .unwrap();
        let challenge = frame_challenge_request::<Goldilocks>(b"msg", 2).unwrap();
        assert_ne!(append, challenge);
    }

    #[test]
    fn test_empty_label_is_rejected() {
        assert_eq!(
            frame_append_elements::<Goldilocks>(b"", &[Goldilocks::from_u64(1)]),
            Err(TranscriptError::EmptyLabel)
        );
    }

    #[test]
    fn test_empty_challenge_is_rejected() {
        assert_eq!(
            frame_challenge_request::<Goldilocks>(b"msg", 0),
            Err(TranscriptError::EmptyChallenge)
        );
    }

    #[test]
    fn test_count_overflow_is_rejected() {
        assert_eq!(
            encode_count::<Goldilocks>(u32::MAX as usize + 1),
            Err(TranscriptError::LengthOverflow)
        );
    }

    #[test]
    fn test_equal_append_sequences_share_frames() {
        let mut lhs = RecordingFieldTranscript::default();
        let mut rhs = RecordingFieldTranscript::default();

        lhs.append_elements(
            b"alpha",
            &[Goldilocks::from_u64(3), Goldilocks::from_u64(5)],
        )
        .unwrap();
        rhs.append_elements(
            b"alpha",
            &[Goldilocks::from_u64(3), Goldilocks::from_u64(5)],
        )
        .unwrap();

        assert_eq!(lhs.frames, rhs.frames);
    }
}
