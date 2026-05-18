use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::LargeCanonicalRing;
use grid_algebra::arith::bigint::BigUint;
use grid_algebra::arith::large_prime::{Bls12_381Fq, Bn254Fr};
use grid_algebra::arith::ntt::NTTRing;
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{Field, Ring};
use grid_algebra::poly::{ntt_forward, ntt_inverse, poly_mul_ntt};
use grid_std::UniformRand;
use grid_std::rand::RngExt;
use std::hint::black_box;

const COEFF_SLICE_LEN: usize = 1024;
const POLY_DEGREE: usize = 64;
const NTT_DEGREE: usize = 256;

type FWordPrime = PrimeField<9223372036854775783>;

fn random_biguint<const N: usize, R: RngExt + ?Sized>(rng: &mut R) -> BigUint<N> {
    BigUint {
        limbs: core::array::from_fn(|_| rng.random()),
    }
}

fn random_biguint_vec<const N: usize>(len: usize) -> Vec<BigUint<N>> {
    let mut rng = grid_std::test_rng();
    (0..len).map(|_| random_biguint(&mut rng)).collect()
}

fn bench_biguint_ops<const N: usize>(c: &mut Criterion, label: &str) {
    let lhs = random_biguint_vec::<N>(COEFF_SLICE_LEN);
    let rhs = random_biguint_vec::<N>(COEFF_SLICE_LEN);

    c.bench_with_input(BenchmarkId::new("substrate_add", label), &label, |b, _| {
        b.iter(|| {
            let out: Vec<_> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(lhs, rhs)| lhs.add_with_carry(black_box(rhs)).0)
                .collect();
            black_box(out);
        });
    });

    c.bench_with_input(BenchmarkId::new("substrate_sub", label), &label, |b, _| {
        b.iter(|| {
            let out: Vec<_> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(lhs, rhs)| lhs.sub_with_borrow(black_box(rhs)).0)
                .collect();
            black_box(out);
        });
    });

    c.bench_with_input(
        BenchmarkId::new("substrate_sub_if_ge", label),
        &label,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| lhs.sub_if_ge(black_box(rhs)))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("substrate_widening_mul", label),
        &label,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| lhs.widening_mul(black_box(rhs)))
                    .collect();
                black_box(out);
            });
        },
    );
}

fn naive_poly_mul_biguint<const N: usize>(
    lhs: &[BigUint<N>],
    rhs: &[BigUint<N>],
) -> Vec<BigUint<N>> {
    let mut out = vec![BigUint::<N>::ZERO; lhs.len() + rhs.len() - 1];
    for (i, lhs_coeff) in lhs.iter().enumerate() {
        for (j, rhs_coeff) in rhs.iter().enumerate() {
            let (lo, _) = lhs_coeff.widening_mul(rhs_coeff);
            out[i + j] = out[i + j].add_with_carry(&lo).0;
        }
    }
    out
}

fn bench_biguint_poly<const N: usize>(c: &mut Criterion, label: &str) {
    let lhs = random_biguint_vec::<N>(POLY_DEGREE);
    let rhs = random_biguint_vec::<N>(POLY_DEGREE);

    c.bench_with_input(
        BenchmarkId::new("naive_poly_mul_deg64", label),
        &label,
        |b, _| {
            b.iter(|| {
                let out = naive_poly_mul_biguint(black_box(&lhs), black_box(&rhs));
                black_box(out);
            });
        },
    );
}

fn nonzero_word_prime_values(len: usize) -> Vec<FWordPrime> {
    let mut rng = grid_std::test_rng();
    (0..len)
        .map(|_| {
            let value = FWordPrime::rand(&mut rng);
            if value.is_zero() {
                FWordPrime::one()
            } else {
                value
            }
        })
        .collect()
}

fn nonzero_large_prime_values<F, const N: usize>(len: usize) -> Vec<F>
where
    F: Field + LargeCanonicalRing<Canonical = BigUint<N>> + UniformRand,
{
    let mut rng = grid_std::test_rng();
    (0..len)
        .map(|_| {
            let value = F::rand(&mut rng);
            if value.is_zero() { F::one() } else { value }
        })
        .collect()
}

fn bench_large_prime_profile<F, const N: usize>(c: &mut Criterion, label: &str)
where
    F: Field + LargeCanonicalRing<Canonical = BigUint<N>> + UniformRand,
{
    let canonical_inputs = random_biguint_vec::<N>(COEFF_SLICE_LEN);
    let values = nonzero_large_prime_values::<F, N>(COEFF_SLICE_LEN);
    let rhs = nonzero_large_prime_values::<F, N>(COEFF_SLICE_LEN);

    c.bench_with_input(
        BenchmarkId::new("field_slice_add", label),
        &label,
        |b, _| {
            b.iter(|| {
                let mut out = values.clone();
                F::add_assign_slice(&mut out, black_box(&rhs));
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("field_slice_sub", label),
        &label,
        |b, _| {
            b.iter(|| {
                let mut out = values.clone();
                F::sub_assign_slice(&mut out, black_box(&rhs));
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("field_from_canonical_full_width", label),
        &label,
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = canonical_inputs
                    .iter()
                    .map(|value| F::from_canonical(black_box(value)))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(BenchmarkId::new("field_inv", label), &label, |b, _| {
        b.iter(|| {
            let out: Vec<_> = values.iter().map(|value| black_box(value).inv()).collect();
            black_box(out);
        });
    });
}

fn random_ntt_poly<F>(len: usize) -> Vec<F>
where
    F: Field + UniformRand,
{
    let mut rng = grid_std::test_rng();
    (0..len).map(|_| F::rand(&mut rng)).collect()
}

fn bench_large_prime_ntt_profile<F, const N: usize>(c: &mut Criterion, label: &str)
where
    F: Field
        + NTTRing
        + LargeCanonicalRing<Canonical = BigUint<N>>
        + UniformRand
        + Send
        + Sync
        + 'static,
{
    let poly = random_ntt_poly::<F>(NTT_DEGREE);
    let mut evals = poly.clone();
    ntt_forward(&mut evals).expect("large-prime profile must support the benchmarked NTT size");
    let rhs = random_ntt_poly::<F>(NTT_DEGREE);

    c.bench_with_input(BenchmarkId::new("ntt_forward", label), &label, |b, _| {
        b.iter(|| {
            let mut values = black_box(poly.clone());
            ntt_forward(&mut values).unwrap();
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("ntt_inverse", label), &label, |b, _| {
        b.iter(|| {
            let mut values = black_box(evals.clone());
            ntt_inverse(&mut values).unwrap();
            black_box(values);
        });
    });

    c.bench_with_input(BenchmarkId::new("poly_mul_ntt", label), &label, |b, _| {
        b.iter(|| black_box(poly_mul_ntt(black_box(&poly), black_box(&rhs)).unwrap()));
    });
}

fn bench_word_prime_baseline(c: &mut Criterion) {
    let lhs = nonzero_word_prime_values(COEFF_SLICE_LEN);
    let rhs = nonzero_word_prime_values(COEFF_SLICE_LEN);

    c.bench_with_input(
        BenchmarkId::new("word_prime_add", "prime63_baseline"),
        &"prime63_baseline",
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| *lhs + black_box(rhs))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("word_prime_mul", "prime63_baseline"),
        &"prime63_baseline",
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| *lhs * black_box(rhs))
                    .collect();
                black_box(out);
            });
        },
    );

    c.bench_with_input(
        BenchmarkId::new("word_prime_inv", "prime63_baseline"),
        &"prime63_baseline",
        |b, _| {
            b.iter(|| {
                let out: Vec<_> = lhs.iter().map(|value| black_box(value).inv()).collect();
                black_box(out);
            });
        },
    );
}

fn bench_large_prime(c: &mut Criterion) {
    bench_word_prime_baseline(c);
    bench_large_prime_profile::<Bn254Fr, 4>(c, "bn254_fr");
    bench_large_prime_profile::<Bls12_381Fq, 6>(c, "bls12_381_fq");
    bench_large_prime_ntt_profile::<Bn254Fr, 4>(c, "bn254_fr");
    bench_large_prime_ntt_profile::<Bls12_381Fq, 6>(c, "bls12_381_fq");
    bench_biguint_ops::<4>(c, "u256_substrate");
    bench_biguint_ops::<6>(c, "u384_substrate");
    bench_biguint_poly::<4>(c, "u256_substrate");
    bench_biguint_poly::<6>(c, "u384_substrate");
}

criterion_group!(benches, bench_large_prime);
criterion_main!(benches);
