use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::composite::CompositeRing;
use grid_algebra::arith::large_rns::Rns3V0;
use grid_algebra::arith::ring::Ring;
use grid_algebra::arith::rns::RnsBasis;
use grid_algebra::poly::{ntt_forward, ntt_inverse, poly_mul_ntt};
use grid_std::UniformRand;
use grid_std::rand::RngExt;
use std::hint::black_box;

const RESIDUE_SLICE_LEN: usize = 2048;
const RNS_PROFILE_NAME: &str = "rns3_v0";
const NTT_DEGREE: usize = 256;
const RNS_PROFILE_PRIMES: [u64; 3] = [
    0x0000_1000_01d0_0001,
    0x0000_1000_03b0_0001,
    0x0000_1000_0450_0001,
];

fn random_values(len: usize) -> Vec<Rns3V0> {
    let mut rng = grid_std::test_rng();
    (0..len).map(|_| Rns3V0::rand(&mut rng)).collect()
}

fn dynamic_basis() -> Arc<RnsBasis> {
    Arc::new(RnsBasis::new(RNS_PROFILE_PRIMES.to_vec()))
}

fn random_composite_values(basis: Arc<RnsBasis>, len: usize) -> Vec<CompositeRing> {
    let mut rng = grid_std::test_rng();
    (0..len)
        .map(|_| CompositeRing::from_u64_with_basis(rng.random(), basis.clone()))
        .collect()
}

fn bench_fixed_profile_residue_ops(c: &mut Criterion) {
    let lhs = random_values(RESIDUE_SLICE_LEN);
    let rhs = random_values(RESIDUE_SLICE_LEN);

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_slice_add", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let mut out = lhs.clone();
                Rns3V0::add_assign_slice(&mut out, black_box(&rhs));
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_slice_sub", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let mut out = lhs.clone();
                Rns3V0::sub_assign_slice(&mut out, black_box(&rhs));
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_add", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| *black_box(lhs) + black_box(rhs))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_sub", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| *black_box(lhs) - black_box(rhs))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_mul", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| *black_box(lhs) * black_box(rhs))
                    .collect();
                black_box(out);
            });
        },
    );
}

fn bench_dynamic_interop(c: &mut Criterion) {
    let basis = dynamic_basis();
    let fixed_values = random_values(RESIDUE_SLICE_LEN);
    let composite_values = random_composite_values(basis.clone(), 512);

    c.bench_with_input(
        BenchmarkId::new("dynamic_reconstruct", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = fixed_values
                    .iter()
                    .map(|value| basis.reconstruct_biguint::<3>(black_box(value.residues())))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("composite_serialize_roundtrip", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = composite_values
                    .iter()
                    .map(|value| {
                        let bytes = grid_serialize::CanonicalSerialize::serialize(black_box(value))
                            .expect("serialize composite value");
                        CompositeRing::deserialize_exact_with_basis(&bytes, basis.clone())
                            .expect("deserialize composite value")
                    })
                    .collect();
                black_box(out);
            });
        },
    );
}

fn bench_fixed_profile_ntt(c: &mut Criterion) {
    let poly = random_values(NTT_DEGREE);
    let mut evals = poly.clone();
    ntt_forward(&mut evals).expect("Rns3V0 must support the benchmarked NTT size");
    let rhs = random_values(NTT_DEGREE);

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_ntt_forward", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let mut values = black_box(poly.clone());
                ntt_forward(&mut values).unwrap();
                black_box(values);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_ntt_inverse", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| {
                let mut values = black_box(evals.clone());
                ntt_inverse(&mut values).unwrap();
                black_box(values);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("fixed_profile_poly_mul_ntt", RNS_PROFILE_NAME),
        &RNS_PROFILE_NAME,
        |b, _| {
            b.iter(|| black_box(poly_mul_ntt(black_box(&poly), black_box(&rhs)).unwrap()));
        },
    );
}

fn bench_rns_large(c: &mut Criterion) {
    bench_fixed_profile_residue_ops(c);
    bench_fixed_profile_ntt(c);
    bench_dynamic_interop(c);
}

criterion_group!(benches, bench_rns_large);
criterion_main!(benches);
