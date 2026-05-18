//! Transcript backend modules.

pub mod poseidon2;
pub mod shake;

pub use poseidon2::{
    GoldilocksPoseidon2Field, GoldilocksPoseidon2Transcript, Poseidon2Parameters,
    Poseidon2Transcript, PoseidonTranscript,
};
pub use shake::ShakeTranscript;

#[cfg(test)]
mod tests {
    use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
    use grid_algebra::arith::ring::IntegerRing;

    use crate::{FieldTranscript, Transcript};

    use super::{GoldilocksPoseidon2Transcript, ShakeTranscript};

    fn run_shared_backend_suite<T, F>(make: F)
    where
        T: Transcript,
        F: Fn() -> T,
    {
        let mut lhs = make();
        let mut rhs = make();

        lhs.append_serializable(b"alpha", &PrimeField::<17>::from_u64(3))
            .unwrap();
        lhs.append_serializable(b"beta", &PrimeField::<17>::from_u64(5))
            .unwrap();
        rhs.append_serializable(b"alpha", &PrimeField::<17>::from_u64(3))
            .unwrap();
        rhs.append_serializable(b"beta", &PrimeField::<17>::from_u64(5))
            .unwrap();
        assert_eq!(
            lhs.challenge_bytes(b"gamma", 32).unwrap(),
            rhs.challenge_bytes(b"gamma", 32).unwrap()
        );

        let mut label_a = make();
        let mut label_b = make();
        label_a
            .append_serializable(b"state", &PrimeField::<17>::from_u64(9))
            .unwrap();
        label_b
            .append_serializable(b"state", &PrimeField::<17>::from_u64(9))
            .unwrap();
        assert_ne!(
            label_a.challenge_bytes(b"left", 32).unwrap(),
            label_b.challenge_bytes(b"right", 32).unwrap()
        );

        let mut ordered = make();
        let mut reordered = make();
        ordered
            .append_serializable(b"x", &PrimeField::<17>::from_u64(1))
            .unwrap();
        ordered
            .append_serializable(b"y", &PrimeField::<17>::from_u64(2))
            .unwrap();
        reordered
            .append_serializable(b"y", &PrimeField::<17>::from_u64(2))
            .unwrap();
        reordered
            .append_serializable(b"x", &PrimeField::<17>::from_u64(1))
            .unwrap();
        assert_ne!(
            ordered.challenge_bytes(b"order", 32).unwrap(),
            reordered.challenge_bytes(b"order", 32).unwrap()
        );

        let mut vector_transcript = make();
        let mut scalar_transcript = make();
        vector_transcript
            .append_serializable(b"seed", &PrimeField::<17>::from_u64(11))
            .unwrap();
        scalar_transcript
            .append_serializable(b"seed", &PrimeField::<17>::from_u64(11))
            .unwrap();
        let vector = vector_transcript
            .challenge_vector::<PrimeField<17>>(b"vec", 3)
            .unwrap();
        let repeated = [
            scalar_transcript
                .challenge_scalar::<PrimeField<17>>(b"vec")
                .unwrap(),
            scalar_transcript
                .challenge_scalar::<PrimeField<17>>(b"vec")
                .unwrap(),
            scalar_transcript
                .challenge_scalar::<PrimeField<17>>(b"vec")
                .unwrap(),
        ];
        assert_eq!(vector.as_slice(), repeated.as_slice());
    }

    #[test]
    fn test_shake_transcript_passes_shared_backend_suite() {
        run_shared_backend_suite(ShakeTranscript::new);
    }

    fn run_shared_field_backend_suite<T, F>(make: F)
    where
        T: FieldTranscript<PrimeField<GOLDILOCKS_MODULUS>>,
        F: Fn() -> T,
    {
        type Goldilocks = PrimeField<GOLDILOCKS_MODULUS>;

        let mut lhs = make();
        let mut rhs = make();

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
        assert_eq!(
            lhs.challenge_elements(b"gamma", 4).unwrap(),
            rhs.challenge_elements(b"gamma", 4).unwrap()
        );

        let mut label_a = make();
        let mut label_b = make();
        label_a
            .append_element(b"state", &Goldilocks::from_u64(9))
            .unwrap();
        label_b
            .append_element(b"state", &Goldilocks::from_u64(9))
            .unwrap();
        assert_ne!(
            label_a.challenge_elements(b"left", 4).unwrap(),
            label_b.challenge_elements(b"right", 4).unwrap()
        );

        let mut ordered = make();
        let mut reordered = make();
        ordered
            .append_element(b"x", &Goldilocks::from_u64(1))
            .unwrap();
        ordered
            .append_element(b"y", &Goldilocks::from_u64(2))
            .unwrap();
        reordered
            .append_element(b"y", &Goldilocks::from_u64(2))
            .unwrap();
        reordered
            .append_element(b"x", &Goldilocks::from_u64(1))
            .unwrap();
        assert_ne!(
            ordered.challenge_elements(b"order", 4).unwrap(),
            reordered.challenge_elements(b"order", 4).unwrap()
        );

        let mut repeated = make();
        repeated
            .append_element(b"seed", &Goldilocks::from_u64(11))
            .unwrap();
        let first = repeated.challenge_elements(b"vec", 3).unwrap();
        let second = repeated.challenge_elements(b"vec", 3).unwrap();
        assert_ne!(first, second);

        let mut append_after_squeeze_a = make();
        let mut append_after_squeeze_b = make();
        let _ = append_after_squeeze_a
            .challenge_elements(b"warm", 2)
            .unwrap();
        let _ = append_after_squeeze_b
            .challenge_elements(b"warm", 2)
            .unwrap();
        append_after_squeeze_a
            .append_element(b"msg", &Goldilocks::from_u64(7))
            .unwrap();
        append_after_squeeze_b
            .append_element(b"msg", &Goldilocks::from_u64(7))
            .unwrap();
        assert_eq!(
            append_after_squeeze_a
                .challenge_elements(b"next", 3)
                .unwrap(),
            append_after_squeeze_b
                .challenge_elements(b"next", 3)
                .unwrap()
        );
    }

    #[test]
    fn test_poseidon2_transcript_passes_shared_field_backend_suite() {
        run_shared_field_backend_suite(GoldilocksPoseidon2Transcript::goldilocks_t12);
    }
}
