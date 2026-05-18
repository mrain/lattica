//! Native Fibonacci LaBRADOR proof — simple linear recurrence.
//!
//! F(i+1) = F(i) + F(i-1), encoded as linear LaBRADOR constraints.
//!
//! # Security Warning
//!
//! This example is demo-only. The parameters are deliberately weakened for
//! fast local iteration and do not provide cryptographic soundness.
//!
//! Known insecure choices:
//! - `security_bits = 8`; real profiles should target at least 128 bits.
//! - `kappa = kappa1 = kappa2 = 2`; Module-SIS ranks are far below any
//!   Core-SVP/BDGL security threshold.
//! - `q = 12281`, `d = 64` is not a paper-aligned LaBRADOR proof ring
//!   profile, and the modulus is too small for useful witness bounds.
//!
//! Do not use these parameters outside tests or demos. Production profiles
//! need a vetted proof-ring factorization, Module-SIS estimator artifacts,
//! and statement-specific norm/security analysis.

use std::time::Instant;

use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::poly::ring::{CyclotomicPolyRing, PolyRing};
use grid_labrador::crs::CRS;
use grid_labrador::params::{ChallengeProfile, JLProfile, LabradorParams};
use grid_labrador::relation::{
    LabradorStatement, LabradorWitness, QuadraticFunction, verify as verify_relation,
};
use grid_labrador::{prove, verify};
use grid_serialize::CanonicalSerialize;

const Q: u64 = 12281;
type F = PrimeField<Q, u16>;
type Poly = CyclotomicPolyRing<F, 64>;
const N: usize = 64;

fn cp(v: u64) -> Poly {
    let mut p = Poly::zero();
    p.set_coeff(0, F::from_u64(v));
    p
}

fn build_fib(
    target: usize,
    num_parts: usize,
    rank: usize,
) -> (LabradorStatement<Poly>, LabradorWitness<Poly>, u64) {
    let mut fib = vec![0u64, 1];
    for i in 2..=target {
        fib.push((fib[i - 1] + fib[i - 2]) % Q);
    }
    let result = fib[target];
    let num_values = target + 1;
    let zero = cp(0);

    // Pack Fibonacci values into parts: value i -> part (i / rank), entry (i % rank)
    let mut f_funcs = Vec::with_capacity(num_values);

    fn pe(idx: usize, rank: usize) -> (usize, usize) {
        (idx / rank, idx % rank)
    }

    // F(0) = 0
    let (p, e) = pe(0, rank);
    f_funcs.push(QuadraticFunction::from_sparse(
        vec![],
        vec![(p, e, cp(1))],
        Poly::zero(),
    ));

    // F(1) = 1
    let (p, e) = pe(1, rank);
    f_funcs.push(QuadraticFunction::from_sparse(
        vec![],
        vec![(p, e, cp(1))],
        cp(1),
    ));

    // Recurrence: F(i) + F(i-1) - F(i+1) = 0
    for i in 1..target {
        let (pi, ei) = pe(i, rank);
        let (pm, em) = pe(i - 1, rank);
        let (pp, ep) = pe(i + 1, rank);
        f_funcs.push(QuadraticFunction::from_sparse(
            vec![],
            vec![(pi, ei, cp(1)), (pm, em, cp(1)), (pp, ep, cp(Q - 1))],
            Poly::zero(),
        ));
    }

    // Build packed witness: each part holds `rank` Fibonacci values (padded with zeros)
    let witness = LabradorWitness::new(
        (0..num_parts)
            .map(|part_idx| {
                let start = part_idx * rank;
                let end = (start + rank).min(num_values);
                let mut part: Vec<Poly> = Vec::with_capacity(rank);
                for &val in fib.iter().take(end).skip(start) {
                    part.push(cp(val));
                }
                // Pad remaining entries with zeros
                while part.len() < rank {
                    part.push(zero.clone());
                }
                part
            })
            .collect(),
    );

    (
        LabradorStatement {
            f: f_funcs,
            f_prime: vec![],
        },
        witness,
        result,
    )
}

fn main() {
    println!("=== Native LaBRADOR Fibonacci ===\n");

    let target: usize = 1000;
    let num_values = target + 1;

    // 2-level profile: r=23 > r'=22 (2*nu+mu), so witness shrinks across levels
    let nu = 1;
    let mu = 20;
    let num_levels = 2;
    let r = 23;
    let n = num_values.div_ceil(r);

    assert!(
        r > 2 * nu + mu,
        "r={} must exceed r'={} for shrinking",
        r,
        2 * nu + mu
    );

    // Beta: witness L2 norm bound
    let beta = 360_000.0;
    let jl_slack = (128.0_f64 / 30.0).sqrt();
    let beta_prime = beta * jl_slack;

    let params = LabradorParams {
        jl: JLProfile::default(),
        challenge: ChallengeProfile::paper_default(),
        security_bits: 8,
        soundness_error: 0.0,
        l: 1,
        arith_p: 274177,
        n,
        r,
        beta,
        d: N,
        q: Q as f64,
        sigma: 1.0,
        b: 2,
        b1: 16,
        b2: 16,
        t1: 4,
        t2: 4,
        kappa: 2,
        kappa1: 2,
        kappa2: 2,
        gamma: beta * 2.5,
        gamma1_sq: (beta * 3.0) as u128 * (beta * 3.0) as u128,
        gamma2_sq: (beta * 3.0) as u128 * (beta * 3.0) as u128,
        beta_prime,
        nu,
        mu,
        num_levels,
    };
    let (statement, witness, expected) = build_fib(target, params.r, params.n);

    println!("F({}) = {}", target, expected);
    println!(
        "F({}) target — r={}, n={} (packed), garbage_count={}, constraints={}, levels={}, nu={}, mu={}\n",
        target,
        params.r,
        params.n,
        params.r * (params.r + 1) / 2,
        statement.num_f(),
        params.num_levels,
        params.nu,
        params.mu
    );

    verify_relation(&statement, &witness, params.beta).expect("principal relation");
    println!("Principal relation: OK");

    // CRS
    let mut rng = grid_std::test_rng();
    let crs = CRS::random(&mut rng);

    // Prove (Phase 5 API)
    let num_main = params.num_levels - 1;
    let mut pt = grid_transcript::hash::ShakeTranscript::default();
    let t0 = Instant::now();
    let proof = prove(&crs, &params, &statement, &witness, num_main, &mut pt)
        .expect("prove should succeed");
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Proof: {} levels", proof.num_levels());

    // Verify (Phase 5 API)
    let mut vt = grid_transcript::hash::ShakeTranscript::default();
    let t1 = Instant::now();
    verify(&crs, &statement, &proof, &params, &mut vt).expect("verify should succeed");
    let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Proof size (actual serialized bytes)
    let proof_bytes = proof.serialize().expect("serialize proof");
    let proof_kb = proof_bytes.len() as f64 / 1024.0;

    // Size breakdown
    fn fmt(b: usize) -> String {
        if b > 1024 {
            format!("{} KB", b as f64 / 1024.0)
        } else {
            format!("{} B", b)
        }
    }
    for (i, level) in proof.levels.iter().enumerate() {
        let lsize = level.serialize().unwrap().len();
        println!("\nMain level {}: {}", i, fmt(lsize));
        println!("  u1:        {}", fmt(level.u1.serialized_size()));
        println!("  jl_seed:   32 B");
        println!("  jl_retry:  4 B");
        println!("  p:         {}", fmt(level.p.serialize().unwrap().len()));
        println!(
            "  b_dp:      {}",
            fmt(level.b_double_prime.serialize().unwrap().len())
        );
        println!("  u2:        {}", fmt(level.u2.serialized_size()));
    }
    let last = &proof.last;
    let lsize = last.serialize().unwrap().len();
    println!("\nLast level: {}", fmt(lsize));
    println!("  t_vecs:    {}", fmt(last.t_vecs.serialized_size()));
    println!("  jl_seed:   32 B");
    println!("  jl_retry:  4 B");
    println!("  p:         {}", fmt(last.p.serialize().unwrap().len()));
    println!(
        "  b_dp:      {}",
        fmt(last.b_double_prime.serialize().unwrap().len())
    );
    println!(
        "  h_garbage: {}",
        fmt(last.h_garbage.serialize().unwrap().len())
    );
    println!(
        "  g_garbage: {}",
        fmt(last.g_garbage.serialize().unwrap().len())
    );
    println!("  z:         {}", fmt(last.z.serialize().unwrap().len()));
    println!("  z polys:   {}", last.z.len());

    println!("Verification: PASSED");
    println!(
        "\nF({}) = {}  ({}-level LaBRADOR proof)",
        target,
        expected,
        proof.num_levels()
    );
    println!("Proving:     {:.1} ms", prove_ms);
    println!("Verification: {:.1} ms", verify_ms);
    println!(
        "Proof size:   {:.1} KB ({} bytes)",
        proof_kb,
        proof_bytes.len()
    );

    // Loop for profiling tools (flamegraph, perf) — proves 10x without verify
    let num_iters: usize = std::env::var("PROFILE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if num_iters > 1 {
        println!("\n[profiling] looping prove {} times...", num_iters);
        for i in 0..num_iters {
            let mut pt = grid_transcript::hash::ShakeTranscript::default();
            let _proof =
                prove(&crs, &params, &statement, &witness, num_main, &mut pt).expect("prove loop");
            if (i + 1) % 5 == 0 {
                println!("[profiling]   iter {}/{}", i + 1, num_iters);
            }
        }
    }
}
