//! Poseidon2-backed field-native transcript backend.

mod params;
mod permutation;
pub mod profile_goldilocks;

use alloc::vec::Vec;
use core::array;
use core::fmt;

use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};

use crate::TranscriptError;
use crate::field::{FieldTranscript, TranscriptField};

use self::params::ExpandedPoseidon2Parameters;
pub use self::params::Poseidon2Parameters;
use self::permutation::permute_state;
pub use self::profile_goldilocks::{
    GOLDILOCKS_T12_POSEIDON2_PARAMS, GoldilocksPoseidon2Field, GoldilocksPoseidon2Transcript,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpongeMode {
    Absorbing,
    Squeezing,
}

/// Field-native Fiat-Shamir transcript backed by a Poseidon2 duplex sponge.
#[derive(Clone, PartialEq, Eq)]
pub struct Poseidon2Transcript<F: TranscriptField, const WIDTH: usize> {
    params: ExpandedPoseidon2Parameters<F, WIDTH>,
    state: [F; WIDTH],
    cursor: usize,
    mode: SpongeMode,
}

/// Historical compatibility alias for the Poseidon2 transcript backend.
pub type PoseidonTranscript<F, const WIDTH: usize> = Poseidon2Transcript<F, WIDTH>;

impl<F: TranscriptField, const WIDTH: usize> Poseidon2Transcript<F, WIDTH> {
    /// Create a new transcript from a checked-in Poseidon2 parameter object.
    pub fn new(params: &'static Poseidon2Parameters<WIDTH>) -> Self {
        Self {
            params: ExpandedPoseidon2Parameters::from_raw(params),
            state: array::from_fn(|_| F::zero()),
            cursor: 0,
            mode: SpongeMode::Absorbing,
        }
    }

    fn permute(&mut self) {
        permute_state(&mut self.state, &self.params);
    }

    fn absorb(&mut self, values: &[F]) {
        if matches!(self.mode, SpongeMode::Squeezing) {
            self.permute();
            self.cursor = 0;
            self.mode = SpongeMode::Absorbing;
        }

        for value in values {
            if self.cursor == self.params.rate {
                self.permute();
                self.cursor = 0;
            }
            self.state[self.cursor] += value.clone();
            self.cursor += 1;
        }
    }

    fn squeeze(&mut self, out_len: usize) -> Vec<F> {
        if matches!(self.mode, SpongeMode::Absorbing) {
            self.permute();
            self.cursor = 0;
            self.mode = SpongeMode::Squeezing;
        }

        let mut out = Vec::with_capacity(out_len);
        while out.len() < out_len {
            if self.cursor == self.params.rate {
                self.permute();
                self.cursor = 0;
            }
            out.push(self.state[self.cursor].clone());
            self.cursor += 1;
        }
        out
    }
}

impl Poseidon2Transcript<PrimeField<GOLDILOCKS_MODULUS>, 12> {
    /// Create the first shipped Goldilocks `t = 12` Poseidon2 transcript profile.
    pub fn goldilocks_t12() -> Self {
        Self::new(&GOLDILOCKS_T12_POSEIDON2_PARAMS)
    }
}

impl<F: TranscriptField, const WIDTH: usize> fmt::Debug for Poseidon2Transcript<F, WIDTH> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Poseidon2Transcript { .. }")
    }
}

impl<F: TranscriptField, const WIDTH: usize> FieldTranscript<F> for Poseidon2Transcript<F, WIDTH> {
    fn append_preframed_elements(&mut self, elements: &[F]) -> Result<(), TranscriptError> {
        self.absorb(elements);
        Ok(())
    }

    fn challenge_preframed_elements(
        &mut self,
        request_frame: &[F],
        out_len: usize,
    ) -> Result<Vec<F>, TranscriptError> {
        self.absorb(request_frame);
        Ok(self.squeeze(out_len))
    }
}

#[cfg(test)]
mod tests {
    use crate::field::frame_append_elements;
    use grid_algebra::arith::ring::IntegerRing;

    use super::*;

    #[test]
    fn test_equal_transcripts_share_challenges() {
        let mut lhs = GoldilocksPoseidon2Transcript::goldilocks_t12();
        let mut rhs = GoldilocksPoseidon2Transcript::goldilocks_t12();

        lhs.append_elements(
            b"seed",
            &[
                GoldilocksPoseidon2Field::from_u64(3),
                GoldilocksPoseidon2Field::from_u64(5),
            ],
        )
        .unwrap();
        rhs.append_elements(
            b"seed",
            &[
                GoldilocksPoseidon2Field::from_u64(3),
                GoldilocksPoseidon2Field::from_u64(5),
            ],
        )
        .unwrap();

        assert_eq!(
            lhs.challenge_elements(b"challenge", 4).unwrap(),
            rhs.challenge_elements(b"challenge", 4).unwrap()
        );
    }

    #[test]
    fn test_repeated_challenges_advance_state() {
        let mut transcript = GoldilocksPoseidon2Transcript::goldilocks_t12();
        transcript
            .append_element(b"seed", &GoldilocksPoseidon2Field::from_u64(9))
            .unwrap();

        let first = transcript.challenge_element(b"r").unwrap();
        let second = transcript.challenge_element(b"r").unwrap();

        assert_ne!(first, second);
    }

    #[test]
    fn test_append_after_squeeze_remains_deterministic() {
        let mut lhs = GoldilocksPoseidon2Transcript::goldilocks_t12();
        let mut rhs = GoldilocksPoseidon2Transcript::goldilocks_t12();

        let _ = lhs.challenge_elements(b"warm", 10).unwrap();
        let _ = rhs.challenge_elements(b"warm", 10).unwrap();
        lhs.append_element(b"msg", &GoldilocksPoseidon2Field::from_u64(1))
            .unwrap();
        rhs.append_element(b"msg", &GoldilocksPoseidon2Field::from_u64(1))
            .unwrap();

        assert_eq!(
            lhs.challenge_elements(b"next", 3)
                .unwrap()
                .iter()
                .map(IntegerRing::to_u64)
                .collect::<Vec<_>>(),
            rhs.challenge_elements(b"next", 3)
                .unwrap()
                .iter()
                .map(IntegerRing::to_u64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_preframed_absorb_chunking_matches_single_call() {
        let frame = frame_append_elements(
            b"payload",
            &(0..11)
                .map(|index| GoldilocksPoseidon2Field::from_u64(index as u64 + 1))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let mut single = GoldilocksPoseidon2Transcript::goldilocks_t12();
        let mut chunked = GoldilocksPoseidon2Transcript::goldilocks_t12();

        single.append_preframed_elements(&frame).unwrap();
        chunked.append_preframed_elements(&frame[..7]).unwrap();
        chunked.append_preframed_elements(&frame[7..]).unwrap();

        assert_eq!(
            single.challenge_elements(b"next", 4).unwrap(),
            chunked.challenge_elements(b"next", 4).unwrap()
        );
    }
}
