use criterion::{Criterion, criterion_group, criterion_main};
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_commit::CommitmentScheme;
use grid_commit::ajtai::{AjtaiCommitmentScheme, AjtaiOpening, AjtaiParams};
use grid_commit::linear::CommitmentDimensions;
use grid_std::UniformRand;
use std::hint::black_box;

type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

fn ajtai_large_params() -> AjtaiParams {
    AjtaiParams {
        dims: CommitmentDimensions {
            message_len: 3072, // 3072 * 256 = 786_432 field elements in the large benchmark workload
            opening_len: 1,    // smallest legal opening to minimize extra work in this API
            commitment_len: 1, // single-row commitment case
        },
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 0,
            max_linf: 0,
        },
        security_bits: 0, // TBD: Ajtai security calibration is not finalized yet.
    }
}

fn bench_ajtai_large(c: &mut Criterion) {
    let mut rng = grid_std::test_rng();
    let rq_scheme =
        AjtaiCommitmentScheme::<Rq23Np8>::setup(&mut rng, &ajtai_large_params()).unwrap();

    let rq_message = RingVec::new((0..3072).map(|_| Rq23Np8::rand(&mut rng)).collect());
    let rq_opening = AjtaiOpening {
        randomness: RingVec::new(vec![Rq23Np8::zero()]),
    };
    let rq_commitment = rq_scheme
        .commit_with_opening(&rq_message, &rq_opening)
        .unwrap();

    let mut group = c.benchmark_group("ajtai_large");
    group.sample_size(10); // matching bench.md instructions for sample-size 10

    group.bench_function("ajtai/commit/large_1x3072", |b| {
        b.iter(|| {
            rq_scheme
                .commit_with_opening(black_box(&rq_message), black_box(&rq_opening))
                .unwrap()
        })
    });
    group.bench_function("ajtai/verify/large_1x3072", |b| {
        b.iter(|| {
            rq_scheme
                .verify(
                    black_box(&rq_commitment),
                    black_box(&rq_message),
                    black_box(&rq_opening),
                )
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_ajtai_large);
criterion_main!(benches);
