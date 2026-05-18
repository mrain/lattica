use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_commit::CommitmentScheme;
use grid_commit::bdlop::{BdlopCommitmentScheme, BdlopParams};
use grid_commit::linear::CommitmentDimensions;

type F17 = PrimeField<17>;
type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

fn small_params() -> BdlopParams {
    BdlopParams {
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
        security_bits: 0, // TBD: Bdlop security calibration is not finalized yet.
    }
}

fn rq23_np8_params() -> BdlopParams {
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

fn bench_bdlop(c: &mut Criterion) {
    let mut rng = grid_std::test_rng();
    let small_scheme = BdlopCommitmentScheme::<F17>::setup(&mut rng, &small_params()).unwrap();
    let small_message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let (small_commitment, small_opening) = small_scheme.commit(&small_message, &mut rng).unwrap();

    let rq_scheme = BdlopCommitmentScheme::<Rq23Np8>::setup(&mut rng, &rq23_np8_params()).unwrap();
    let rq_message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (rq_commitment, rq_opening) = rq_scheme.commit(&rq_message, &mut rng).unwrap();

    c.bench_function("bdlop/commit/f17", |b| {
        b.iter_batched(
            grid_std::test_rng,
            |mut local_rng| small_scheme.commit(&small_message, &mut local_rng).unwrap(),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("bdlop/verify/f17", |b| {
        b.iter(|| {
            small_scheme
                .verify(&small_commitment, &small_message, &small_opening)
                .unwrap()
        })
    });
    c.bench_function("bdlop/commit/rq23_np8", |b| {
        b.iter_batched(
            grid_std::test_rng,
            |mut local_rng| rq_scheme.commit(&rq_message, &mut local_rng).unwrap(),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("bdlop/verify/rq23_np8", |b| {
        b.iter(|| {
            rq_scheme
                .verify(&rq_commitment, &rq_message, &rq_opening)
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_bdlop);
criterion_main!(benches);
