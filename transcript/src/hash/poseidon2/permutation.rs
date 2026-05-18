//! Poseidon2 permutation helpers for the transcript backend.

use crate::field::TranscriptField;

use super::params::ExpandedPoseidon2Parameters;

fn sbox_degree_7<F: TranscriptField>(value: &F) -> F {
    let square = value.square();
    let cube = square.clone() * value;
    let sixth = cube.square();
    sixth * value
}

fn external_linear_layer<F: TranscriptField, const WIDTH: usize>(state: &mut [F; WIDTH]) {
    assert_eq!(
        WIDTH % 4,
        0,
        "Poseidon2 external layer expects widths divisible by 4"
    );

    for chunk in state.chunks_exact_mut(4) {
        let s0 = chunk[0].clone();
        let s1 = chunk[1].clone();
        let s2 = chunk[2].clone();
        let s3 = chunk[3].clone();

        let t0 = s0.clone() + &s1;
        let t1 = s2.clone() + &s3;
        let t2 = t0.clone() + &t1;
        let t3 = t2.clone() + &s1;
        let t4 = t2.clone() + &s3;
        let t5 = s0.double();
        let t6 = s2.double();

        chunk[0] = t3.clone() + &t0;
        chunk[1] = t6 + &t3;
        chunk[2] = t1 + &t4;
        chunk[3] = t5 + &t4;
    }

    let mut sums = [F::zero(), F::zero(), F::zero(), F::zero()];
    for chunk in state.chunks_exact(4) {
        for (column, value) in chunk.iter().enumerate() {
            sums[column] += value.clone();
        }
    }
    for (index, value) in state.iter_mut().enumerate() {
        *value += sums[index % 4].clone();
    }
}

fn internal_linear_layer<F: TranscriptField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    diag: &[F; WIDTH],
) {
    let sum = state.iter().fold(F::zero(), |mut acc, value| {
        acc += value.clone();
        acc
    });
    for (value, diag_value) in state.iter_mut().zip(diag.iter()) {
        *value = value.clone() * diag_value + &sum;
    }
}

fn add_external_round_constants<F: TranscriptField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    constants: &[F; WIDTH],
) {
    for (value, constant) in state.iter_mut().zip(constants.iter()) {
        *value += constant.clone();
    }
}

fn add_internal_round_constant<F: TranscriptField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    constant: &F,
) {
    state[0] += constant.clone();
}

pub(crate) fn permute_state<F: TranscriptField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    params: &ExpandedPoseidon2Parameters<F, WIDTH>,
) {
    let half_full_rounds = params.full_rounds / 2;

    external_linear_layer(state);

    for round in 0..half_full_rounds {
        add_external_round_constants(state, &params.external_constants[round]);
        for value in state.iter_mut() {
            *value = sbox_degree_7(value);
        }
        external_linear_layer(state);
    }

    for constant in params.internal_constants.iter() {
        add_internal_round_constant(state, constant);
        state[0] = sbox_degree_7(&state[0]);
        internal_linear_layer(state, &params.internal_matrix_diag);
    }

    for round in half_full_rounds..params.full_rounds {
        add_external_round_constants(state, &params.external_constants[round]);
        for value in state.iter_mut() {
            *value = sbox_degree_7(value);
        }
        external_linear_layer(state);
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
    use grid_algebra::arith::ring::{IntegerRing, Ring};

    use super::*;
    use crate::hash::poseidon2::params::ExpandedPoseidon2Parameters;
    use crate::hash::poseidon2::profile_goldilocks::{
        GOLDILOCKS_T12_POSEIDON2_PARAMS, GoldilocksPoseidon2Field,
    };

    #[test]
    fn test_internal_linear_layer_matches_sum_plus_diag_form() {
        let mut state = array::from_fn::<GoldilocksPoseidon2Field, 12, _>(|i| {
            GoldilocksPoseidon2Field::from_u64((i + 1) as u64)
        });
        let diag = array::from_fn::<GoldilocksPoseidon2Field, 12, _>(|i| {
            GoldilocksPoseidon2Field::from_u64((10 + i) as u64)
        });
        let original = state;

        internal_linear_layer(&mut state, &diag);

        let sum = original
            .iter()
            .fold(GoldilocksPoseidon2Field::zero(), |mut acc, value| {
                acc += *value;
                acc
            });
        for index in 0..12 {
            assert_eq!(state[index], original[index] * diag[index] + sum);
        }
    }

    #[test]
    fn test_permute_zero_state_matches_reference_vector() {
        let params = ExpandedPoseidon2Parameters::<GoldilocksPoseidon2Field, 12>::from_raw(
            &GOLDILOCKS_T12_POSEIDON2_PARAMS,
        );
        let mut state = [PrimeField::<GOLDILOCKS_MODULUS>::zero(); 12];
        permute_state(&mut state, &params);

        let expected = [
            7182099517097165596u64,
            9311216678150108034,
            8831900494918587432,
            10774846510254277933,
            10601329242472021962,
            5629867288322699978,
            140799316430260029,
            16680789625189310103,
            16589856342819292996,
            4940126994627441183,
            14089387953811494999,
            8340711910841427341,
        ];

        assert_eq!(
            state.map(|value: GoldilocksPoseidon2Field| value.to_u64()),
            expected,
            "zero-state Poseidon2 permutation must match the reference vector"
        );
    }
}
