use criterion::{Criterion, criterion_group, criterion_main};
use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::prepare_twisted_ring_vec;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_commit::CommitmentScheme;
use grid_commit::ajtai::{AjtaiCommitmentScheme, AjtaiOpening, AjtaiParams};
use grid_commit::linear::CommitmentDimensions;
use grid_std::UniformRand;
use std::hint::black_box;

type GoldilockRq = CyclotomicPolyRing<PrimeField<GOLDILOCKS_MODULUS>, 32>;

fn ajtai_goldilocks_params() -> AjtaiParams {
    AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 32_768, // Targets the `1 x 32768` Goldilocks Ajtai workload shape.
            opening_len: 1,      // smallest legal opening to minimize extra work in this API
            commitment_len: 1,   // single-row commitment case
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 0,
            max_linf: 0,
        },
        security_bits: 0, // TBD: Goldilock Ajtai security calibration is not finalized yet.
    }
}

fn bench_ajtai_goldilocks(c: &mut Criterion) {
    let mut rng = grid_std::test_rng();
    let scheme =
        AjtaiCommitmentScheme::<GoldilockRq>::setup(&mut rng, &ajtai_goldilocks_params()).unwrap();

    let message = RingVec::new((0..32_768).map(|_| GoldilockRq::rand(&mut rng)).collect());
    let opening = AjtaiOpening {
        randomness: RingVec::new(vec![GoldilockRq::zero()]),
    };
    // This benchmark measures repeated commits/verifies over a fixed input, so prepare the message
    // and opening once outside the timed section and reuse their twisted-domain form per iteration.
    let message_ntt = prepare_twisted_ring_vec(&message).unwrap();
    let opening_ntt = prepare_twisted_ring_vec(&opening.randomness).unwrap();
    let prepared_message = scheme.prepare_ntt(&message_ntt).unwrap();
    let prepared_opening = scheme.prepare_opening_ntt(&opening_ntt).unwrap();
    let commitment = scheme
        .commit_with_opening_ntt(&prepared_message, &prepared_opening)
        .unwrap();

    let mut group = c.benchmark_group("ajtai_goldilocks");
    group.sample_size(10);

    group.bench_function("ajtai/commit/goldilocks_1x32768", |b| {
        b.iter(|| {
            scheme
                .commit_with_opening_ntt(black_box(&prepared_message), black_box(&prepared_opening))
                .unwrap()
        })
    });
    group.bench_function("ajtai/verify/goldilocks_1x32768", |b| {
        b.iter(|| {
            scheme
                .verify_ntt(
                    black_box(&commitment),
                    black_box(&prepared_message),
                    black_box(&prepared_opening),
                )
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_ajtai_goldilocks);
criterion_main!(benches);
