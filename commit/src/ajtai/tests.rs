use super::*;

use grid_algebra::arith::bigint::BigUint;
use grid_algebra::arith::large_prime::Bn254Fr;
use grid_algebra::arith::large_rns::Rns3V0;
use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::{LargeNormBound, NormBound};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::prepare_twisted_polys;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_serialize::{CanonicalSerialize, SerializationError, Valid};
use grid_std::rand::rand_core::{Infallible, TryRng};

use crate::linear::CommitmentDimensions;
use crate::linear::recompute_linear_commitment_parts;

type F17 = PrimeField<17>;
type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;
type GoldilockRq = CyclotomicPolyRing<PrimeField<GOLDILOCKS_MODULUS>, 256>;

fn small_params() -> AjtaiParams {
    AjtaiParams {
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
        security_bits: 128,
    }
}

fn rq23_np8_params() -> AjtaiParams {
    AjtaiParams {
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
        security_bits: 128,
    }
}

fn zero_bound_params() -> AjtaiParams {
    AjtaiParams {
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
        security_bits: 128,
    }
}

fn bn254_large_params() -> AjtaiParams<LargeNormBound<BigUint<8>>> {
    AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 2,
            opening_len: 2,
            commitment_len: 2,
        },
        opening_eta: 1,
        opening_bound: LargeNormBound {
            max_l2_sq: BigUint::<8>::from_u64(2),
            max_linf: BigUint::<8>::from_u64(1),
        },
        security_bits: 128,
    }
}

fn rns3_large_params() -> AjtaiParams<LargeNormBound<BigUint<6>>> {
    AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 2,
            opening_len: 2,
            commitment_len: 2,
        },
        opening_eta: 1,
        opening_bound: LargeNormBound {
            max_l2_sq: BigUint::<6>::from_u64(2),
            max_linf: BigUint::<6>::from_u64(1),
        },
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
fn test_ajtai_params_reject_invalid_dimensions() {
    let params = AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 0,
            opening_len: 1,
            commitment_len: 1,
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4,
            max_linf: 2,
        },
        security_bits: 128,
    };
    assert!(!params.is_valid());
}

#[test]
fn test_ajtai_params_allow_zero_security_bits_placeholder() {
    let params = AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 1,
            opening_len: 1,
            commitment_len: 1,
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 0,
            max_linf: 0,
        },
        security_bits: 0,
    };
    assert!(params.is_valid());
}

#[test]
fn test_ajtai_params_serialize_round_trip() {
    let params = small_params();
    let bytes = params.serialize().unwrap();
    let decoded = AjtaiParams::deserialize_and_validate_exact(&bytes).unwrap();
    assert_eq!(decoded, params);
}

#[test]
fn test_ajtai_params_large_bound_serialize_round_trip() {
    let params = AjtaiParams::<LargeNormBound<BigUint<8>>> {
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
        security_bits: 128,
    };
    let bytes = params.serialize().unwrap();
    let decoded =
        AjtaiParams::<LargeNormBound<BigUint<8>>>::deserialize_and_validate_exact(&bytes).unwrap();
    assert_eq!(decoded, params);
}

#[test]
fn test_ajtai_key_commitment_opening_serialize_round_trip() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    let key_bytes = scheme.key().serialize().unwrap();
    let decoded_key =
        AjtaiCommitmentKey::<F17>::deserialize_and_validate_exact(&key_bytes).unwrap();
    assert_eq!(decoded_key, *scheme.key());

    let commitment_bytes = commitment.serialize().unwrap();
    let decoded_commitment =
        AjtaiCommitment::<F17>::deserialize_and_validate_exact(&commitment_bytes).unwrap();
    assert_eq!(decoded_commitment, commitment);

    let opening_bytes = opening.serialize().unwrap();
    let decoded_opening =
        AjtaiOpening::<F17>::deserialize_and_validate_exact(&opening_bytes).unwrap();
    assert_eq!(decoded_opening, opening);
}

#[test]
fn test_ajtai_key_deserialize_and_validate_rejects_row_mismatch() {
    let malformed = AjtaiCommitmentKey {
        a_msg: RingMat::new(1, 2, vec![F17::from_u64(1), F17::from_u64(2)]),
        a_open: RingMat::new(
            2,
            2,
            vec![
                F17::from_u64(3),
                F17::from_u64(4),
                F17::from_u64(5),
                F17::from_u64(6),
            ],
        ),
    };
    assert!(!malformed.is_valid());

    let bytes = malformed.serialize().unwrap();
    let err = AjtaiCommitmentKey::<F17>::deserialize_and_validate(&bytes).unwrap_err();
    assert_eq!(
        err,
        SerializationError::InvalidData("deserialized value is invalid".into())
    );
}

#[test]
fn test_setup_rejects_invalid_dimensions() {
    let mut rng = grid_std::test_rng();
    let params = AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 1,
            opening_len: 0,
            commitment_len: 1,
        },
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 1,
            max_linf: 1,
        },
        security_bits: 128,
    };
    assert_eq!(
        AjtaiCommitmentScheme::<F17>::setup(&mut rng, &params),
        Err(CommitmentError::InvalidParameters)
    );
}

#[test]
fn test_ajtai_round_trip_over_prime_field() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_round_trip_over_large_prime_with_large_bounds() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Bn254Fr, LargeNormBound<BigUint<8>>>::setup(
        &mut rng,
        &bn254_large_params(),
    )
    .unwrap();
    let message = RingVec::new(vec![Bn254Fr::from_u64(5), Bn254Fr::from_u64(9)]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_round_trip_over_large_rns_with_large_bounds() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Rns3V0, LargeNormBound<BigUint<6>>>::setup(
        &mut rng,
        &rns3_large_params(),
    )
    .unwrap();
    let message = RingVec::new(vec![Rns3V0::from_u64(5), Rns3V0::from_u64(9)]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_round_trip_over_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();

    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_round_trip_over_goldilocks_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<GoldilockRq>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        GoldilockRq::one(),
        GoldilockRq::zero(),
        GoldilockRq::one(),
        GoldilockRq::zero(),
    ]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![
            GoldilockRq::zero(),
            GoldilockRq::one(),
            GoldilockRq::zero(),
            GoldilockRq::one(),
        ]),
    };

    let commitment = scheme.commit_with_opening(&message, &opening).unwrap();

    assert!(scheme.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_prepared_path_matches_coefficient_reference() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![
            Rq23Np8::zero(),
            Rq23Np8::one(),
            Rq23Np8::zero(),
            Rq23Np8::one(),
        ]),
    };

    let commitment = scheme.commit_with_opening(&message, &opening).unwrap();
    let reference = recompute_linear_commitment_parts(
        &scheme.key().a_msg,
        &scheme.key().a_open,
        &message,
        &opening.randomness,
        rq23_np8_params().dims,
    )
    .unwrap();

    assert_eq!(commitment.value, reference.value);
}

#[test]
fn test_ajtai_prepared_path_matches_coefficient_reference_goldilocks() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<GoldilockRq>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        GoldilockRq::one(),
        GoldilockRq::zero(),
        GoldilockRq::one(),
        GoldilockRq::zero(),
    ]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![
            GoldilockRq::zero(),
            GoldilockRq::one(),
            GoldilockRq::zero(),
            GoldilockRq::one(),
        ]),
    };

    let commitment = scheme.commit_with_opening(&message, &opening).unwrap();
    let reference = recompute_linear_commitment_parts(
        &scheme.key().a_msg,
        &scheme.key().a_open,
        &message,
        &opening.randomness,
        rq23_np8_params().dims,
    )
    .unwrap();

    assert_eq!(commitment.value, reference.value);
}

#[test]
fn test_ajtai_ntt_inputs_match_commit_and_verify_goldilocks() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<GoldilockRq>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        GoldilockRq::one(),
        GoldilockRq::zero(),
        GoldilockRq::one(),
        GoldilockRq::zero(),
    ]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![
            GoldilockRq::zero(),
            GoldilockRq::one(),
            GoldilockRq::zero(),
            GoldilockRq::one(),
        ]),
    };

    let message_ntt = RingVec::new(prepare_twisted_polys(message.entries()).unwrap());
    let opening_ntt = RingVec::new(prepare_twisted_polys(opening.randomness.entries()).unwrap());
    let prepared_message = scheme.prepare_ntt(&message_ntt).unwrap();
    let prepared_opening = scheme.prepare_opening_ntt(&opening_ntt).unwrap();
    let commitment = scheme.commit_with_opening(&message, &opening).unwrap();
    let prepared_commitment = scheme
        .commit_with_opening_ntt(&prepared_message, &prepared_opening)
        .unwrap();

    assert_eq!(prepared_message.finish().unwrap(), message);
    assert_eq!(
        prepared_opening.finish_randomness().unwrap(),
        opening.randomness
    );
    assert_eq!(prepared_commitment, commitment);
    assert!(
        scheme
            .verify_ntt(&commitment, &prepared_message, &prepared_opening)
            .unwrap()
    );
    assert!(
        scheme
            .verify(
                &commitment,
                &prepared_message.finish().unwrap(),
                &AjtaiOpening {
                    randomness: prepared_opening.finish_randomness().unwrap(),
                },
            )
            .unwrap()
    );
}

#[test]
fn test_ajtai_commit_ntt_round_trips_with_bridged_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<GoldilockRq>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        GoldilockRq::one(),
        GoldilockRq::zero(),
        GoldilockRq::one(),
        GoldilockRq::zero(),
    ]);
    let message_ntt = RingVec::new(prepare_twisted_polys(message.entries()).unwrap());
    let prepared_message = scheme.prepare_ntt(&message_ntt).unwrap();

    let (commitment, opening_ntt) = scheme.commit_ntt(&prepared_message, &mut rng).unwrap();

    assert!(
        scheme
            .verify_ntt(&commitment, &prepared_message, &opening_ntt)
            .unwrap()
    );
    assert!(
        scheme
            .verify(
                &commitment,
                &prepared_message.finish().unwrap(),
                &AjtaiOpening {
                    randomness: opening_ntt.finish_randomness().unwrap(),
                },
            )
            .unwrap()
    );
}

#[test]
fn test_ajtai_commit_is_deterministic_with_explicit_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(2), F17::from_u64(3)]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
    };

    let c1 = scheme.commit_with_opening(&message, &opening).unwrap();
    let c2 = scheme.commit_with_opening(&message, &opening).unwrap();

    assert_eq!(c1, c2);
}

#[test]
fn test_ajtai_rejects_wrong_message_and_wrong_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let wrong_message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(8)]);

    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    assert!(
        !scheme
            .verify(&commitment, &wrong_message, &opening)
            .unwrap()
    );

    let wrong_opening = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::from_u64(1)]),
    };
    assert!(
        !scheme
            .verify(&commitment, &message, &wrong_opening)
            .unwrap()
    );
}

#[test]
fn test_ajtai_rejects_wrong_key() {
    let mut rng = grid_std::test_rng();
    let params = small_params();
    let scheme_a = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &params).unwrap();
    let scheme_b = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &params).unwrap();
    let message = RingVec::new(vec![F17::from_u64(7), F17::from_u64(4)]);

    let (commitment, opening) = scheme_a.commit(&message, &mut rng).unwrap();

    assert!(!scheme_b.verify(&commitment, &message, &opening).unwrap());
}

#[test]
fn test_ajtai_rejects_over_bound_opening() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(2), F17::from_u64(1)]),
    };
    assert_eq!(
        scheme.commit_with_opening(&message, &opening),
        Err(CommitmentError::OpeningNormExceeded)
    );
}

#[test]
fn test_ajtai_commit_rejects_sampled_opening_outside_bound() {
    let mut setup_rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut setup_rng, &zero_bound_params()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(1)]);
    let mut rng = AlternatingBitRng::new();

    assert_eq!(
        scheme.commit(&message, &mut rng),
        Err(CommitmentError::OpeningNormExceeded)
    );
}

#[test]
fn test_ajtai_commit_rechecks_dimension_rule() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let malformed_message = RingVec::new(vec![F17::from_u64(1)]);
    let malformed_opening = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1)]),
    };

    assert_eq!(
        scheme.commit(&malformed_message, &mut rng),
        Err(CommitmentError::DimensionMismatch)
    );
    assert_eq!(
        scheme.commit_with_opening(
            &RingVec::new(vec![F17::from_u64(1), F17::from_u64(2)]),
            &malformed_opening,
        ),
        Err(CommitmentError::DimensionMismatch)
    );
}

#[test]
fn test_ajtai_additive_homomorphism() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let message_a = RingVec::new(vec![F17::from_u64(1), F17::from_u64(2)]);
    let message_b = RingVec::new(vec![F17::from_u64(3), F17::from_u64(4)]);
    let opening_a = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
    };
    let opening_b = AjtaiOpening {
        randomness: RingVec::new(vec![F17::zero(), F17::from_u64(1)]),
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
fn test_ajtai_add_openings_rejects_sum_outside_bound() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let opening_a = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
    };
    let opening_b = AjtaiOpening {
        randomness: RingVec::new(vec![F17::from_u64(1), F17::zero()]),
    };

    assert_eq!(
        scheme.add_openings(&opening_a, &opening_b),
        Err(CommitmentError::OpeningNormExceeded)
    );
}
