//! Poseidon2 parameter objects and runtime-expanded profile data.

use alloc::vec::Vec;
use core::array;

use crate::field::TranscriptField;

/// Raw checked-in Poseidon2 profile data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poseidon2Parameters<const WIDTH: usize> {
    /// Sponge rate.
    pub rate: usize,
    /// Number of full rounds.
    pub full_rounds: usize,
    /// Number of partial rounds.
    pub partial_rounds: usize,
    /// External round constants in raw canonical field-word form.
    pub external_constants_raw: &'static [[u64; WIDTH]],
    /// Internal round constants in raw canonical field-word form.
    pub internal_constants_raw: &'static [u64],
    /// Diagonal values for the internal linear layer.
    pub internal_matrix_diag_raw: [u64; WIDTH],
}

/// Field-valued Poseidon2 profile data expanded from the raw checked-in words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpandedPoseidon2Parameters<F: TranscriptField, const WIDTH: usize> {
    pub rate: usize,
    pub full_rounds: usize,
    pub partial_rounds: usize,
    pub external_constants: Vec<[F; WIDTH]>,
    pub internal_constants: Vec<F>,
    pub internal_matrix_diag: [F; WIDTH],
}

impl<F: TranscriptField, const WIDTH: usize> ExpandedPoseidon2Parameters<F, WIDTH> {
    /// Expand a raw checked-in profile into field-valued constants.
    pub fn from_raw(params: &'static Poseidon2Parameters<WIDTH>) -> Self {
        assert!(params.rate > 0, "Poseidon2 rate must be non-zero");
        assert!(
            params.rate < WIDTH,
            "Poseidon2 rate must be smaller than width"
        );
        assert_eq!(
            params.full_rounds % 2,
            0,
            "Poseidon2 full rounds must be even"
        );
        assert_eq!(
            params.external_constants_raw.len(),
            params.full_rounds,
            "external constant rows must match full rounds"
        );
        assert_eq!(
            params.internal_constants_raw.len(),
            params.partial_rounds,
            "internal constants must match partial rounds"
        );

        let external_constants = params
            .external_constants_raw
            .iter()
            .map(|row| array::from_fn(|index| F::from_u64(row[index])))
            .collect();
        let internal_constants = params
            .internal_constants_raw
            .iter()
            .map(|value| F::from_u64(*value))
            .collect();
        let internal_matrix_diag =
            array::from_fn(|index| F::from_u64(params.internal_matrix_diag_raw[index]));

        Self {
            rate: params.rate,
            full_rounds: params.full_rounds,
            partial_rounds: params.partial_rounds,
            external_constants,
            internal_constants,
            internal_matrix_diag,
        }
    }
}
