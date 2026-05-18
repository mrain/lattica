//! SHAKE-based transcript backend.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

use crate::Transcript;
use crate::TranscriptError;
use crate::encoding::frame_challenge_request;

/// Fiat-Shamir transcript backed by an incremental SHAKE256 sponge.
#[derive(Clone, Default)]
pub struct ShakeTranscript {
    sponge: Shake256,
}

impl ShakeTranscript {
    /// Create a new empty SHAKE transcript.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Debug for ShakeTranscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ShakeTranscript { .. }")
    }
}

impl Transcript for ShakeTranscript {
    fn append_preframed_bytes(&mut self, bytes: &[u8]) -> Result<(), TranscriptError> {
        self.sponge.update(bytes);
        Ok(())
    }

    fn challenge_bytes(
        &mut self,
        label: &'static [u8],
        out_len: usize,
    ) -> Result<Vec<u8>, TranscriptError> {
        let request = frame_challenge_request(label, out_len)?;

        let mut sponge = self.sponge.clone();
        sponge.update(&request);

        let mut reader = sponge.finalize_xof();
        let mut output = vec![0u8; out_len];
        reader.read(&mut output);

        self.sponge.update(&request);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::prime::PrimeField;

    use crate::encoding::frame_append;

    #[test]
    fn test_equal_append_sequences_yield_equal_challenges() {
        let mut lhs = ShakeTranscript::new();
        let mut rhs = ShakeTranscript::new();

        lhs.append_bytes(b"msg", b"first").unwrap();
        lhs.append_bytes(b"msg", b"second").unwrap();
        rhs.append_bytes(b"msg", b"first").unwrap();
        rhs.append_bytes(b"msg", b"second").unwrap();

        assert_eq!(
            lhs.challenge_bytes(b"chal", 32).unwrap(),
            rhs.challenge_bytes(b"chal", 32).unwrap()
        );
    }

    #[test]
    fn test_distinct_append_order_yields_distinct_challenges() {
        let mut lhs = ShakeTranscript::new();
        let mut rhs = ShakeTranscript::new();

        lhs.append_bytes(b"msg", b"first").unwrap();
        lhs.append_bytes(b"msg", b"second").unwrap();
        rhs.append_bytes(b"msg", b"second").unwrap();
        rhs.append_bytes(b"msg", b"first").unwrap();

        assert_ne!(
            lhs.challenge_bytes(b"chal", 32).unwrap(),
            rhs.challenge_bytes(b"chal", 32).unwrap()
        );
    }

    #[test]
    fn test_repeated_challenges_advance_transcript_state() {
        let mut transcript = ShakeTranscript::new();
        transcript.append_bytes(b"msg", b"payload").unwrap();

        let first = transcript.challenge_scalar::<PrimeField<17>>(b"r").unwrap();
        let second = transcript.challenge_scalar::<PrimeField<17>>(b"r").unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn test_preframed_append_matches_safe_append_path() {
        let mut safe = ShakeTranscript::new();
        let mut explicit = ShakeTranscript::new();

        safe.append_bytes(b"msg", b"payload").unwrap();
        explicit
            .append_preframed_bytes(&frame_append(b"msg", b"payload").unwrap())
            .unwrap();

        assert_eq!(
            safe.challenge_bytes(b"chal", 32).unwrap(),
            explicit.challenge_bytes(b"chal", 32).unwrap()
        );
    }

    #[test]
    fn test_unframed_preframed_append_is_distinct_from_safe_append() {
        let mut safe = ShakeTranscript::new();
        let mut unframed = ShakeTranscript::new();

        safe.append_bytes(b"msg", b"payload").unwrap();
        unframed.append_preframed_bytes(b"payload").unwrap();

        assert_ne!(
            safe.challenge_bytes(b"chal", 32).unwrap(),
            unframed.challenge_bytes(b"chal", 32).unwrap()
        );
    }
}
