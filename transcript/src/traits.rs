//! Byte-oriented transcript traits and helpers.

use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_serialize::CanonicalSerialize;

use crate::TranscriptError;
use crate::encoding::{frame_append, frame_challenge_request};

/// Shared transcript interface for Fiat-Shamir style challenge generation.
pub trait Transcript {
    /// Low-level hook to append bytes that have already been framed by the caller.
    fn append_preframed_bytes(&mut self, bytes: &[u8]) -> Result<(), TranscriptError>;

    /// Append raw bytes using the shared label/length framing rules.
    fn append_bytes(
        &mut self,
        label: &'static [u8],
        payload: &[u8],
    ) -> Result<(), TranscriptError> {
        let framed = frame_append(label, payload)?;
        self.append_preframed_bytes(&framed)
    }

    /// Derive challenge bytes and bind the challenge request into the transcript state.
    fn challenge_bytes(
        &mut self,
        label: &'static [u8],
        out_len: usize,
    ) -> Result<Vec<u8>, TranscriptError>;

    /// Append a canonically serializable value using the shared framing rules.
    fn append_serializable<T: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        value: &T,
    ) -> Result<(), TranscriptError> {
        let payload = value.serialize()?;
        self.append_bytes(label, &payload)
    }

    /// Derive a single scalar challenge by reducing challenge bytes into a backend scalar.
    fn challenge_scalar<R: IntegerRing>(
        &mut self,
        label: &'static [u8],
    ) -> Result<R, TranscriptError> {
        let mut bytes = self.challenge_bytes(label, 8)?;
        bytes.resize(8, 0);
        Ok(R::from_u64(u64::from_le_bytes(
            bytes[..8].try_into().unwrap(),
        )))
    }

    /// Derive `len` sequential scalar challenges under the same label.
    fn challenge_vector<R: IntegerRing>(
        &mut self,
        label: &'static [u8],
        len: usize,
    ) -> Result<Vec<R>, TranscriptError> {
        if len == 0 {
            return Err(TranscriptError::EmptyChallenge);
        }
        let _ = frame_challenge_request(label, len)?;
        (0..len).map(|_| self.challenge_scalar(label)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::prime::PrimeField;

    use crate::TranscriptError;

    #[derive(Default)]
    struct RecordingTranscript {
        frames: Vec<Vec<u8>>,
        counter: u64,
    }

    impl Transcript for RecordingTranscript {
        fn append_preframed_bytes(&mut self, bytes: &[u8]) -> Result<(), TranscriptError> {
            self.frames.push(bytes.to_vec());
            Ok(())
        }

        fn challenge_bytes(
            &mut self,
            label: &'static [u8],
            out_len: usize,
        ) -> Result<Vec<u8>, TranscriptError> {
            let framed = frame_challenge_request(label, out_len)?;
            self.frames.push(framed);
            let seed = self.counter.to_le_bytes();
            self.counter += 1;
            Ok((0..out_len).map(|i| seed[i % seed.len()]).collect())
        }
    }

    #[test]
    fn test_append_serializable_uses_framing() {
        let mut transcript = RecordingTranscript::default();
        transcript
            .append_serializable(b"x", &PrimeField::<17>::from_u64(8))
            .unwrap();
        assert_eq!(transcript.frames.len(), 1);
        assert_eq!(transcript.frames[0][0], 0x00);
    }

    #[test]
    fn test_challenge_scalar_reduces_bytes() {
        let mut transcript = RecordingTranscript::default();
        let challenge = transcript.challenge_scalar::<PrimeField<17>>(b"r").unwrap();
        assert_eq!(challenge, PrimeField::<17>::from_u64(0));
    }

    #[test]
    fn test_challenge_vector_returns_requested_length() {
        let mut transcript = RecordingTranscript::default();
        let challenges = transcript
            .challenge_vector::<PrimeField<17>>(b"vec", 3)
            .unwrap();
        assert_eq!(challenges.len(), 3);
    }

    #[test]
    fn test_challenge_vector_rejects_empty_length() {
        let mut transcript = RecordingTranscript::default();
        assert_eq!(
            transcript.challenge_vector::<PrimeField<17>>(b"vec", 0),
            Err(TranscriptError::EmptyChallenge)
        );
    }
}
