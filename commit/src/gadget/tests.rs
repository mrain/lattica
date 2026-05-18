use super::*;

use grid_algebra::arith::bigint::BigUint;
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::{LargeNormBound, NormBound};
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_serialize::{CanonicalSerialize, Valid};
use grid_std::rand::rand_core::{Infallible, TryRng};

use crate::linear::CommitmentDimensions;

type F17 = PrimeField<17>;
type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

fn small_params() -> GadgetParams {
    GadgetParams {
        dims: CommitmentDimensions {
            message_len: 2,
            opening_len: 2,
            commitment_len: 2,
        },
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 2,
            max_linf: 1,
        },
        base: 4,
        digits: 3,
        security_bits: 128,
    }
}

fn rq23_np8_params() -> GadgetParams {
    GadgetParams {
        dims: CommitmentDimensions {
            message_len: 4,
            opening_len: 4,
            commitment_len: 4,
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4096,
            max_linf: 2,
        },
        base: 256,
        digits: 3,
        security_bits: 128,
    }
}

fn zero_bound_params() -> GadgetParams {
    GadgetParams {
        dims: CommitmentDimensions {
            message_len: 1,
            opening_len: 1,
            commitment_len: 1,
        },
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 0,
            max_linf: 0,
        },
        base: 4,
        digits: 3,
        security_bits: 128,
    }
}

struct AlternatingBitRng {
    emit_zero: bool,
}

impl AlternatingBitRng {
    fn new() -> Self {
        Self { emit_zero: true }
    }

    fn next_word(&mut self) -> u64 {
        let word = if self.emit_zero { 0 } else { u64::MAX };
        self.emit_zero = !self.emit_zero;
        word
    }
}

impl TryRng for AlternatingBitRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.next_word() as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next_word())
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        let mut offset = 0;
        while offset < dest.len() {
            let word = self.next_word().to_le_bytes();
            let take = core::cmp::min(word.len(), dest.len() - offset);
            dest[offset..offset + take].copy_from_slice(&word[..take]);
            offset += take;
        }
        Ok(())
    }
}

#[test]
fn test_gadget_params_reject_dimension_rule_violation() {
    let params = GadgetParams {
        dims: CommitmentDimensions {
            message_len: 2,
            opening_len: 1,
            commitment_len: 1,
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4,
            max_linf: 2,
        },
        base: 256,
        digits: 3,
        security_bits: 128,
    };
    assert!(!params.is_valid());
}

#[test]
fn test_gadget_params_serialize_round_trip() {
    let params = small_params();
    let bytes = params.serialize().unwrap();
    let decoded = GadgetParams::deserialize_and_validate_exact(&bytes).unwrap();
    assert_eq!(decoded, params);
}

#[test]
fn test_gadget_params_large_bound_serialize_round_trip() {
    let params = GadgetParams::<LargeNormBound<BigUint<8>>> {
        dims: CommitmentDimensions {
            message_len: 2,
            opening_len: 2,
            commitment_len: 2,
        },
        opening_eta: 1,
        opening_bound: LargeNormBound {
            max_l2_sq: BigUint::<8>::from_u64(123),
            max_linf: BigUint::<8>::from_u64(7),
        },
        base: 4,
        digits: 3,
        security_bits: 128,
    };
    let bytes = params.serialize().unwrap();
    let decoded =
        GadgetParams::<LargeNormBound<BigUint<8>>>::deserialize_and_validate_exact(&bytes).unwrap();
    assert_eq!(decoded, params);
}

#[test]
fn test_gadget_commitment_opening_serialize_round_trip() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(6)]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    let commitment_bytes = commitment.serialize().unwrap();
    let decoded_commitment =
        GadgetCommitment::<F17>::deserialize_and_validate_exact(&commitment_bytes).unwrap();
    assert_eq!(decoded_commitment, commitment);

    let opening_bytes = opening.serialize().unwrap();
    let decoded_opening =
        GadgetOpening::<F17>::deserialize_and_validate_exact(&opening_bytes).unwrap();
    assert_eq!(decoded_opening, opening);
}

#[test]
fn test_setup_rejects_wrong_digit_count_for_backend() {
    let mut rng = grid_std::test_rng();
    let params = GadgetParams {
        digits: 2,
        ..small_params()
    };
    assert_eq!(
        GadgetCommitmentScheme::<F17>::setup(&mut rng, &params),
        Err(CommitmentError::InvalidParameters)
    );
}

#[test]
fn test_g_matrix_is_deterministic() {
    let mut rng = grid_std::test_rng();
    let scheme_a = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let scheme_b = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();

    assert_eq!(scheme_a.g_matrix(), scheme_b.g_matrix());
    assert_ne!(scheme_a.a_open(), scheme_b.a_open());
}

#[test]
fn test_gadget_round_trip_over_prime_field() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(6)]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert_eq!(opening.digits.len(), 6);
    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_gadget_round_trip_over_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert_eq!(opening.digits.len(), 12);
    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_gadget_commit_matches_explicit_g_matrix_formula_over_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let opening = GadgetOpening {
        randomness: RingVec::new(vec![
            Rq23Np8::zero(),
            Rq23Np8::one(),
            Rq23Np8::zero(),
            Rq23Np8::one(),
        ]),
        digits: RingVec::new(
            message
                .entries()
                .iter()
                .flat_map(|value| {
                    Rq23Np8::decompose_element(value, rq23_np8_params().base).unwrap()
                })
                .collect(),
        ),
    };

    let commitment = scheme.commit_with_opening(&message, &opening).unwrap();
    let explicit =
        scheme.a_open().mul_vec(&opening.randomness) + &scheme.g_matrix().mul_vec(&opening.digits);

    assert_eq!(commitment.value, explicit);
}

#[test]
fn test_gadget_commit_is_deterministic_with_explicit_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(2), F17::from_u64(3)]);
    let opening = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
            F17::from_u64(3),
            F17::zero(),
            F17::zero(),
        ]),
    };

    let c1 = scheme.commit_with_opening(&message, &opening).unwrap();
    let c2 = scheme.commit_with_opening(&message, &opening).unwrap();

    assert_eq!(c1, c2);
}

#[test]
fn test_gadget_rejects_malformed_digit_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(2), F17::from_u64(3)]);

    let wrong_len = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![F17::from_u64(2)]),
    };
    assert_eq!(
        scheme.commit_with_opening(&message, &wrong_len),
        Err(CommitmentError::DimensionMismatch)
    );

    let non_canonical = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(4),
            F17::zero(),
            F17::zero(),
            F17::from_u64(3),
            F17::zero(),
            F17::zero(),
        ]),
    };
    assert_eq!(
        scheme.commit_with_opening(&message, &non_canonical),
        Err(CommitmentError::InvalidOpening)
    );

    let wrong_digits = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
            F17::from_u64(3),
            F17::zero(),
            F17::zero(),
        ]),
    };
    assert_eq!(
        scheme.commit_with_opening(&message, &wrong_digits),
        Err(CommitmentError::InvalidOpening)
    );
}

#[test]
fn test_gadget_rejects_over_bound_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(2), F17::from_u64(3)]);
    let opening = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(2), F17::from_u64(1)]),
        digits: RingVec::new(vec![
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
            F17::from_u64(3),
            F17::zero(),
            F17::zero(),
        ]),
    };
    assert_eq!(
        scheme.commit_with_opening(&message, &opening),
        Err(CommitmentError::OpeningNormExceeded)
    );
}

#[test]
fn test_gadget_commit_rejects_sampled_opening_outside_bound() {
    let mut setup_rng = grid_std::test_rng();
    let scheme =
        GadgetCommitmentScheme::<F17>::setup(&mut setup_rng, &zero_bound_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(1)]);
    let mut rng = AlternatingBitRng::new();

    assert_eq!(
        scheme.commit(&message, &mut rng),
        Err(CommitmentError::OpeningNormExceeded)
    );
}

#[test]
fn test_gadget_additive_homomorphism_without_carry() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message_a = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2)]);
    let message_b = RingVec::new(vec![F17::from_u64(2), F17::from_u64(1)]);
    let opening_a = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
        ]),
    };
    let opening_b = GadgetOpening {
        randomness: RingVec::new(vec![F17::zero(), F17::from_u64(1)]),
        digits: RingVec::new(vec![
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
        ]),
    };

    let commitment_a = scheme.commit_with_opening(&message_a, &opening_a).unwrap();
    let commitment_b = scheme.commit_with_opening(&message_b, &opening_b).unwrap();
    let summed_commitment = scheme
        .add_commitments(&commitment_a, &commitment_b)
        .unwrap();
    let summed_opening = scheme.add_openings(&opening_a, &opening_b).unwrap();
    let summed_message = message_a + &message_b;
    let expected = scheme
        .commit_with_opening(&summed_message, &summed_opening)
        .unwrap();

    assert_eq!(summed_commitment, expected);
    assert!(
        scheme
            .verify(&summed_commitment, &summed_message, &summed_opening)
            .unwrap()
    );
}

#[test]
fn test_gadget_additive_homomorphism_with_carry_recanonicalizes_digits() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message_a = RingVec::new(vec![F17::from_u64(3), F17::from_u64(2)]);
    let message_b = RingVec::new(vec![F17::from_u64(1), F17::from_u64(1)]);
    let opening_a = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(3),
            F17::zero(),
            F17::zero(),
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
        ]),
    };
    let opening_b = GadgetOpening {
        randomness: RingVec::new(vec![F17::zero(), F17::from_u64(1)]),
        digits: RingVec::new(vec![
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
        ]),
    };

    let commitment_a = scheme.commit_with_opening(&message_a, &opening_a).unwrap();
    let commitment_b = scheme.commit_with_opening(&message_b, &opening_b).unwrap();
    let summed_commitment = scheme
        .add_commitments(&commitment_a, &commitment_b)
        .unwrap();
    let summed_opening = scheme.add_openings(&opening_a, &opening_b).unwrap();
    let summed_message = message_a + &message_b;
    let expected = scheme
        .commit_with_opening(&summed_message, &summed_opening)
        .unwrap();

    assert_eq!(summed_opening.digits.entries()[0].to_u64(), 0);
    assert_eq!(summed_opening.digits.entries()[1].to_u64(), 1);
    assert!(
        summed_opening
            .digits
            .entries()
            .iter()
            .all(|digit| digit.to_u64() < 4)
    );
    assert_eq!(summed_commitment, expected);
    assert!(
        scheme
            .verify(&summed_commitment, &summed_message, &summed_opening)
            .unwrap()
    );
}

#[test]
fn test_gadget_add_openings_rejects_sum_outside_bound() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let opening_a = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
        ]),
    };
    let opening_b = GadgetOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
        digits: RingVec::new(vec![
            F17::from_u64(2),
            F17::zero(),
            F17::zero(),
            F17::from_u64(1),
            F17::zero(),
            F17::zero(),
        ]),
    };

    assert_eq!(
        scheme.add_openings(&opening_a, &opening_b),
        Err(CommitmentError::OpeningNormExceeded)
    );
}
