use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::Ring;
use grid_algebra::arith::z2k::Z2K;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_algebra::poly::{ntt_forward, ntt_inverse, poly_mul_ntt};
use grid_std::UniformRand;
use std::hint::black_box;

const DEGREE: usize = 256;
const SLICE_LEN: usize = 4096;

// ── Ring types ──────────────────────────────────────────────────────

type F12289 = PrimeField<12289>;
type F8380417 = PrimeField<8380417>;
type Z32 = Z2K<32>;
type Z64 = Z2K<64>;
type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, DEGREE>;
type Z32Poly = CyclotomicPolyRing<Z2K<32>, DEGREE>;

// ── Helpers ─────────────────────────────────────────────────────────

fn random_values<R: Ring + UniformRand>(len: usize) -> Vec<R> {
    let mut rng = grid_std::test_rng();
    (0..len).map(|_| R::rand(&mut rng)).collect()
}

fn random_ntt_poly<F: grid_algebra::arith::ring::Field + UniformRand>(len: usize) -> Vec<F> {
    let mut rng = grid_std::test_rng();
    (0..len).map(|_| F::rand(&mut rng)).collect()
}

// ── Ring slice ops ──────────────────────────────────────────────────

fn bench_ring_slice<R: Ring + UniformRand + 'static>(c: &mut Criterion, label: &str) {
    let lhs = random_values::<R>(SLICE_LEN);
    let rhs = random_values::<R>(SLICE_LEN);

    c.bench_with_input(BenchmarkId::new("add", label), &(), |b, _| {
        b.iter(|| {
            let mut values = black_box(lhs.clone());
            R::add_assign_slice(&mut values, black_box(&rhs));
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("sub", label), &(), |b, _| {
        b.iter(|| {
            let mut values = black_box(lhs.clone());
            R::sub_assign_slice(&mut values, black_box(&rhs));
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("mul", label), &(), |b, _| {
        b.iter(|| {
            let mut values = black_box(lhs.clone());
            R::pointwise_mul_assign_slice(&mut values, black_box(&rhs));
            black_box(values);
        });
    });
}

// ── Poly ring ops ───────────────────────────────────────────────────

fn bench_poly<R>(c: &mut Criterion, label: &str)
where
    R: Ring + UniformRand + 'static,
{
    let mut rng = grid_std::test_rng();
    let lhs = R::rand(&mut rng);
    let rhs = R::rand(&mut rng);

    c.bench_with_input(BenchmarkId::new("add", label), &(), |b, _| {
        b.iter(|| black_box(lhs.clone()) + black_box(rhs.clone()));
    });
    c.bench_with_input(BenchmarkId::new("sub", label), &(), |b, _| {
        b.iter(|| black_box(lhs.clone()) - black_box(rhs.clone()));
    });
    c.bench_with_input(BenchmarkId::new("mul", label), &(), |b, _| {
        b.iter(|| black_box(lhs.clone()) * black_box(rhs.clone()));
    });
}

// ── Lattice ops ─────────────────────────────────────────────────────

fn prime_vec(len: usize) -> RingVec<F8380417> {
    let mut rng = grid_std::test_rng();
    RingVec::new((0..len).map(|_| F8380417::rand(&mut rng)).collect())
}

fn prime_mat(rows: usize, cols: usize) -> RingMat<F8380417> {
    let mut rng = grid_std::test_rng();
    RingMat::new(
        rows,
        cols,
        (0..rows * cols).map(|_| F8380417::rand(&mut rng)).collect(),
    )
}

fn rq23_np8_vec(len: usize) -> RingVec<Rq23Np8> {
    let mut rng = grid_std::test_rng();
    RingVec::new((0..len).map(|_| Rq23Np8::rand(&mut rng)).collect())
}

fn rq23_np8_mat(rows: usize, cols: usize) -> RingMat<Rq23Np8> {
    let mut rng = grid_std::test_rng();
    RingMat::new(
        rows,
        cols,
        (0..rows * cols).map(|_| Rq23Np8::rand(&mut rng)).collect(),
    )
}

// ── NTT ops ─────────────────────────────────────────────────────────

fn bench_ntt(c: &mut Criterion) {
    let n = 256usize;
    let poly = random_ntt_poly::<F12289>(n);
    let mut evals = poly.clone();
    ntt_forward(&mut evals).unwrap();
    let lhs = random_ntt_poly::<F12289>(n);
    let rhs = random_ntt_poly::<F12289>(n);

    c.bench_with_input(BenchmarkId::new("forward", n), &(), |b, _| {
        b.iter(|| {
            let mut values = black_box(poly.clone());
            ntt_forward(&mut values).unwrap();
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("inverse", n), &(), |b, _| {
        b.iter(|| {
            let mut values = black_box(evals.clone());
            ntt_inverse(&mut values).unwrap();
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("poly_mul_ntt", n), &(), |b, _| {
        b.iter(|| black_box(poly_mul_ntt(black_box(&lhs), black_box(&rhs)).unwrap()));
    });
}

// ── Runner ──────────────────────────────────────────────────────────

fn bench_all(c: &mut Criterion) {
    // Ring slice ops
    bench_ring_slice::<F12289>(c, "ring/prime12289");
    bench_ring_slice::<F8380417>(c, "ring/prime8380417");
    bench_ring_slice::<Z32>(c, "ring/pow2_32");
    bench_ring_slice::<Z64>(c, "ring/pow2_64");

    // Poly ring ops
    bench_poly::<Rq23Np8>(c, "poly/rq23_np8");
    bench_poly::<Z32Poly>(c, "poly/pow2_32");

    // Lattice ops
    let prime_lhs = prime_vec(4096);
    let prime_rhs = prime_vec(4096);
    let prime_matrix = prime_mat(64, 64);
    let prime_matrix_rhs = prime_mat(64, 64);
    let prime_vec64 = prime_vec(64);

    c.bench_with_input(
        BenchmarkId::new("ringvec_dot", "prime8380417_4096"),
        &(),
        |b, _| {
            b.iter(|| black_box(prime_lhs.dot(black_box(&prime_rhs))));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringvec_add", "prime8380417_4096"),
        &(),
        |b, _| {
            b.iter(|| black_box(prime_lhs.clone() + black_box(prime_rhs.clone())));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringvec_sub", "prime8380417_4096"),
        &(),
        |b, _| {
            b.iter(|| black_box(prime_lhs.clone() - black_box(prime_rhs.clone())));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringmat_mul_vec", "prime8380417_64x64"),
        &(),
        |b, _| {
            b.iter(|| black_box(prime_matrix.mul_vec(black_box(&prime_vec64))));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringmat_add", "prime8380417_64x64"),
        &(),
        |b, _| {
            b.iter(|| black_box(prime_matrix.clone() + black_box(prime_matrix_rhs.clone())));
        },
    );

    let lhs = rq23_np8_vec(4);
    let rhs = rq23_np8_vec(4);
    let matrix = rq23_np8_mat(4, 4);
    let matrix_rhs = rq23_np8_mat(4, 4);

    c.bench_with_input(
        BenchmarkId::new("ringvec_dot", "rq23_np8_4"),
        &(),
        |b, _| {
            b.iter(|| black_box(lhs.dot(black_box(&rhs))));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringvec_add", "rq23_np8_4"),
        &(),
        |b, _| {
            b.iter(|| black_box(lhs.clone() + black_box(rhs.clone())));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringvec_sub", "rq23_np8_4"),
        &(),
        |b, _| {
            b.iter(|| black_box(lhs.clone() - black_box(rhs.clone())));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringmat_mul_vec", "rq23_np8_4x4"),
        &(),
        |b, _| {
            b.iter(|| black_box(matrix.mul_vec(black_box(&rhs))));
        },
    );
    c.bench_with_input(
        BenchmarkId::new("ringmat_add", "rq23_np8_4x4"),
        &(),
        |b, _| {
            b.iter(|| black_box(matrix.clone() + black_box(matrix_rhs.clone())));
        },
    );

    // NTT
    bench_ntt(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
