use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::{prepare_twisted_ring_vec, ring::CyclotomicPolyRing};
use grid_commit::CommitmentScheme;
use grid_commit::ajtai::{AjtaiCommitmentScheme, AjtaiParams};
use grid_commit::bdlop::{BdlopCommitmentScheme, BdlopParams};
use grid_commit::linear::CommitmentDimensions;
use std::hint::black_box;

type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

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

fn bench_prepared_paths(c: &mut Criterion) {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    let prepared_message = scheme.prepare_message(&message).unwrap();
    let prepared_opening = scheme.prepare_opening(&opening).unwrap();

    c.bench_function("prepared/ajtai/prepare_message/rq23_np8", |b| {
        b.iter(|| scheme.prepare_message(black_box(&message)).unwrap())
    });

    c.bench_function("prepared/ajtai/prepare_opening/rq23_np8", |b| {
        b.iter(|| scheme.prepare_opening(black_box(&opening)).unwrap())
    });

    c.bench_function("prepared/ajtai/commit_with_opening/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .commit_with_opening(black_box(&message), black_box(&opening))
                .unwrap()
        })
    });

    c.bench_function("prepared/ajtai/commit_with_opening_ntt/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .commit_with_opening_ntt(black_box(&prepared_message), black_box(&prepared_opening))
                .unwrap()
        })
    });

    c.bench_function("prepared/ajtai/commit_sampled/rq23_np8", |b| {
        b.iter_batched(
            grid_std::test_rng,
            |mut local_rng| scheme.commit(black_box(&message), &mut local_rng).unwrap(),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("prepared/ajtai/verify/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .verify(
                    black_box(&commitment),
                    black_box(&message),
                    black_box(&opening),
                )
                .unwrap()
        })
    });

    c.bench_function("prepared/ajtai/verify_ntt/rq23_np8", |b| {
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

fn rq23_np8_bdlop_params() -> BdlopParams {
    BdlopParams {
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

fn bench_bdlop_prepared_paths(c: &mut Criterion) {
    let mut rng = grid_std::test_rng();
    let scheme =
        BdlopCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_bdlop_params()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    let prepared_message = scheme
        .prepare_ntt(&prepare_twisted_ring_vec(&message).unwrap())
        .unwrap();
    let prepared_opening = scheme
        .prepare_opening_ntt(&prepare_twisted_ring_vec(&opening.randomness).unwrap())
        .unwrap();

    c.bench_function("prepared/bdlop/commit_with_opening/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .commit_with_opening(black_box(&message), black_box(&opening))
                .unwrap()
        })
    });

    c.bench_function("prepared/bdlop/commit_with_opening_ntt/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .commit_with_opening_ntt(black_box(&prepared_message), black_box(&prepared_opening))
                .unwrap()
        })
    });

    c.bench_function("prepared/bdlop/commit_sampled/rq23_np8", |b| {
        b.iter_batched(
            grid_std::test_rng,
            |mut local_rng| scheme.commit(black_box(&message), &mut local_rng).unwrap(),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("prepared/bdlop/verify/rq23_np8", |b| {
        b.iter(|| {
            scheme
                .verify(
                    black_box(&commitment),
                    black_box(&message),
                    black_box(&opening),
                )
                .unwrap()
        })
    });

    c.bench_function("prepared/bdlop/verify_ntt/rq23_np8", |b| {
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

criterion_group!(benches, bench_prepared_paths, bench_bdlop_prepared_paths);
criterion_main!(benches);
