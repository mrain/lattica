//! Mixed R1CS → R reduction (§6, prose after Figure 5).
//!
//! Combines binary R1CS (for hash/comparison/table-lookups) and arithmetic R1CS
//! (for integer arithmetic) into a single LaBRADOR relation R instance.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::types::RingMat;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_std::rand::RngExt;
use grid_transcript::TranscriptError;

use crate::error::LabradorError;
use crate::reduction::binary_r1cs::{
    BinaryR1CSInstance, BinaryR1CSReduction, build_binary_r1cs_reduction,
    build_binary_r1cs_reduction_transcript,
};
use crate::reduction::r1cs_mod_rns::{
    ArithR1CSInstance, ArithR1CSReduction, build_arith_r1cs_reduction,
    build_arith_r1cs_reduction_transcript, check_divisible_by_x_minus_2, verify_naf_coeffs,
};
use crate::relation::{LabradorStatement, LabradorWitness, QuadraticFunction, verify};

/// Combined binary + arithmetic R1CS instance.
#[derive(Debug, Clone)]
pub struct MixedR1CSInstance<R: IntegerRing<Uint = u64>> {
    pub binary: BinaryR1CSInstance<R>,
    pub arithmetic: ArithR1CSInstance<R>,
}

/// Result of the mixed R1CS → R reduction.
#[derive(Debug, Clone)]
pub struct MixedR1CSReduction<R, const N: usize>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    pub statement: LabradorStatement<CyclotomicPolyRing<R, N>>,
    pub witness: LabradorWitness<CyclotomicPolyRing<R, N>>,
    pub commitment_binary_t: grid_algebra::lattice::types::RingVec<CyclotomicPolyRing<R, N>>,
    pub commitment_arith_t: grid_algebra::lattice::types::RingVec<CyclotomicPolyRing<R, N>>,
    pub commitment_arith_td: grid_algebra::lattice::types::RingVec<CyclotomicPolyRing<R, N>>,
    pub l_binary: usize,
    pub l_arithmetic: usize,
    /// Binary F2-challenge g_values carried from binary reduction.
    /// Verifier must check g_i ≡ 0 (mod 2) per paper §6 Figure 4.
    pub binary_g_values: Vec<i64>,
    /// Arithmetic R1CS modulus M = 2^d + 1 for external g_j(2) = 0 mod M check.
    pub arithmetic_modulus_m: u64,
    /// Binary R1CS: number of constraints (rows of A, B, C matrices)
    pub binary_k: usize,
    /// Binary R1CS: number of variables (columns of A, B, C matrices)
    pub binary_n: usize,
}

/// Remap quadratic function indices by adding an offset, and pad/extend phi vectors.
/// For binary functions: prepend_zeros=0, append_zeros=arith_num_parts.
/// For arithmetic functions: prepend_zeros=binary_num_parts, append_zeros=0.
fn remap_function<R, const N: usize>(
    f: &QuadraticFunction<CyclotomicPolyRing<R, N>>,
    part_offset: usize,
    prepend_zeros: usize,
    append_zeros: usize,
    part_rank: usize,
) -> QuadraticFunction<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing + NegacyclicMulRing<N>,
{
    match f {
        QuadraticFunction::Dense(d) => {
            let zero = CyclotomicPolyRing::<R, N>::zero();

            let ij: Vec<(usize, usize)> =
                d.ij.iter()
                    .map(|&(i, j)| (i + part_offset, j + part_offset))
                    .collect();

            let padded_phi: Vec<Vec<CyclotomicPolyRing<R, N>>> = d
                .phi
                .iter()
                .map(|phi_i| {
                    let mut v = phi_i.clone();
                    while v.len() < part_rank {
                        v.push(zero.clone());
                    }
                    v
                })
                .collect();

            let mut phi: Vec<Vec<CyclotomicPolyRing<R, N>>> =
                Vec::with_capacity(prepend_zeros + padded_phi.len() + append_zeros);
            for _ in 0..prepend_zeros {
                phi.push(vec![zero.clone(); part_rank]);
            }
            phi.extend(padded_phi);
            for _ in 0..append_zeros {
                phi.push(vec![zero.clone(); part_rank]);
            }

            QuadraticFunction::Dense(crate::relation::DenseQuadraticFunction {
                a: d.a.clone(),
                ij,
                phi,
                b: d.b.clone(),
            })
        }
        QuadraticFunction::Sparse(s) => {
            // Remap sparse indices: add part_offset to quadratic part indices and phi part indices
            let ij_a: Vec<_> = s
                .ij_a
                .iter()
                .map(|&(i, j, ref c)| (i + part_offset, j + part_offset, c.clone()))
                .collect();
            let phi: Vec<_> = s
                .phi
                .iter()
                .map(|&(part_idx, entry_idx, ref c)| (part_idx + part_offset, entry_idx, c.clone()))
                .collect();

            QuadraticFunction::Sparse(crate::relation::SparseQuadraticFunction {
                ij_a,
                phi,
                b: s.b.clone(),
            })
        }
    }
}

/// Combine binary and arithmetic reductions into a single LaBRADOR relation.
///
/// Remaps phi indices so binary constraints see parts 0..binary_num_parts
/// and arithmetic constraints see parts binary_num_parts..combined_num_parts.
fn combine_reductions<R, const N: usize>(
    binary_reduction: BinaryR1CSReduction<R, N>,
    arith_reduction: ArithR1CSReduction<R, N>,
) -> MixedR1CSReduction<R, N>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    let binary_num_parts = binary_reduction.witness.num_parts();
    let arith_num_parts = arith_reduction.witness.num_parts();
    let binary_rank = binary_reduction.witness.rank();
    let arith_rank = arith_reduction.witness.rank();
    let combined_rank = binary_rank.max(arith_rank);

    let mut combined_parts: Vec<Vec<CyclotomicPolyRing<R, N>>> =
        Vec::with_capacity(binary_num_parts + arith_num_parts);
    let zero = CyclotomicPolyRing::<R, N>::zero();
    for part in binary_reduction.witness.parts {
        let mut padded = part;
        while padded.len() < combined_rank {
            padded.push(zero.clone());
        }
        combined_parts.push(padded);
    }
    for part in arith_reduction.witness.parts {
        let mut padded = part;
        while padded.len() < combined_rank {
            padded.push(zero.clone());
        }
        combined_parts.push(padded);
    }
    let combined_witness = LabradorWitness::new(combined_parts);

    // Combine F: binary F (prepend=0, append=arith_num_parts) +
    //           arithmetic F (prepend=binary_num_parts, append=0)
    let mut combined_f: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = binary_reduction
        .statement
        .f
        .iter()
        .map(|f| remap_function(f, 0, 0, arith_num_parts, combined_rank))
        .collect();
    for f in arith_reduction.statement.f {
        combined_f.push(remap_function(
            &f,
            binary_num_parts,
            binary_num_parts,
            0,
            combined_rank,
        ));
    }

    // Combine F': same pattern
    let mut combined_f_prime: Vec<QuadraticFunction<CyclotomicPolyRing<R, N>>> = binary_reduction
        .statement
        .f_prime
        .iter()
        .map(|f| remap_function(f, 0, 0, arith_num_parts, combined_rank))
        .collect();
    for f in arith_reduction.statement.f_prime {
        combined_f_prime.push(remap_function(
            &f,
            binary_num_parts,
            binary_num_parts,
            0,
            combined_rank,
        ));
    }

    let statement = LabradorStatement {
        f: combined_f,
        f_prime: combined_f_prime,
    };

    MixedR1CSReduction {
        statement,
        witness: combined_witness,
        commitment_binary_t: binary_reduction.commitment,
        commitment_arith_t: arith_reduction.commitment_t,
        commitment_arith_td: arith_reduction.commitment_td,
        l_binary: binary_reduction.l,
        l_arithmetic: arith_reduction.l,
        binary_g_values: binary_reduction.g_values,
        arithmetic_modulus_m: arith_reduction.m,
        binary_k: binary_reduction.k,
        binary_n: binary_reduction.n,
    }
}

/// Build the mixed R1CS → R reduction.
pub fn build_mixed_r1cs_reduction<R, Rng, const N: usize>(
    instance: &MixedR1CSInstance<R>,
    witness_binary: &[R],
    witness_arith: &[R],
    crs_a_binary: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_a_arith: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_b_arith: &RingMat<CyclotomicPolyRing<R, N>>,
    rng: &mut Rng,
    l_binary: usize,
    l_arithmetic: usize,
) -> Result<MixedR1CSReduction<R, N>, LabradorError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
    Rng: RngExt,
{
    let binary_reduction = build_binary_r1cs_reduction::<R, Rng, N>(
        &instance.binary,
        witness_binary,
        crs_a_binary,
        rng,
        l_binary,
    )?;

    let arith_reduction = build_arith_r1cs_reduction::<R, Rng, N>(
        &instance.arithmetic,
        witness_arith,
        crs_a_arith,
        crs_b_arith,
        rng,
        l_arithmetic,
    )?;

    Ok(combine_reductions(binary_reduction, arith_reduction))
}

/// Build the mixed R1CS → R reduction using transcript for all verifier challenges.
///
/// Same as [`build_mixed_r1cs_reduction`] but derives both binary F2 and arithmetic
/// challenges from the Fiat-Shamir transcript.
pub fn build_mixed_r1cs_reduction_transcript<R, T, const N: usize>(
    instance: &MixedR1CSInstance<R>,
    witness_binary: &[R],
    witness_arith: &[R],
    crs_a_binary: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_a_arith: &RingMat<CyclotomicPolyRing<R, N>>,
    crs_b_arith: &RingMat<CyclotomicPolyRing<R, N>>,
    transcript: &mut T,
    l_binary: usize,
    l_arithmetic: usize,
) -> Result<MixedR1CSReduction<R, N>, TranscriptError>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
    T: grid_transcript::Transcript,
{
    let binary_reduction = build_binary_r1cs_reduction_transcript::<R, T, N>(
        &instance.binary,
        witness_binary,
        crs_a_binary,
        transcript,
        l_binary,
    )?;

    let arith_reduction = build_arith_r1cs_reduction_transcript::<R, T, N>(
        &instance.arithmetic,
        witness_arith,
        crs_a_arith,
        crs_b_arith,
        transcript,
        l_arithmetic,
    )?;

    Ok(combine_reductions(binary_reduction, arith_reduction))
}

/// Verify a mixed R1CS reduction: relation::verify + binary g_i parity + arithmetic checks.
///
/// This is the complete verifier path per paper §6 mixed proof.
/// Runs all external checks from both reductions in parallel:
/// 1. LaBRADOR relation::verify (combined F/F' + norm bound)
/// 2. Binary g_i ≡ 0 (mod 2) parity check (from combined F' b constants)
/// 3. Arithmetic g_j(2) ≡ 0 (mod M) divisibility (from combined F b constants)
/// 4. Arithmetic NAF coefficient bounds on all encoded parts (a,b,c,w,d_i)
///
/// Returns `Ok(())` if all checks pass.
pub fn verify_mixed_r1cs_reduction<R, const N: usize>(
    reduction: &MixedR1CSReduction<R, N>,
    max_norm_bound: f64,
) -> Result<(), String>
where
    R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
{
    // 1. Verify LaBRADOR relation (combined binary + arithmetic F/F' constraints)
    verify(&reduction.statement, &reduction.witness, max_norm_bound)?;

    // 2. Binary g_i parity: extract from combined F' b constants
    // Combined F' layout: [N*(3*binary_k+binary_n) conjugacy][4 binary][1 Hadamard][l_binary F2]
    let binary_f2_start =
        crate::reduction::binary_r1cs::binary_f2_start::<N>(reduction.binary_k, reduction.binary_n);
    let f_prime = &reduction.statement.f_prime;
    if f_prime.len() < binary_f2_start + reduction.l_binary {
        return Err(format!(
            "F' has {} functions, need {} for binary F2 extraction",
            f_prime.len(),
            binary_f2_start + reduction.l_binary
        ));
    }
    let q = R::modulus();
    for i in 0..reduction.l_binary {
        let g_mod = f_prime[binary_f2_start + i].b().coeff(0).to_u64();
        let g_signed: i64 = if g_mod > q / 2 {
            -((q - g_mod) as i64)
        } else {
            g_mod as i64
        };
        if g_signed % 2 != 0 {
            return Err(format!(
                "mixed binary g_values[{}] is odd (g={} from combined F' b constant)",
                i, g_signed
            ));
        }
    }

    // 3. Arithmetic g_j(2) ≡ 0 (mod M): extract from combined F b constants
    // Arithmetic aggregation constraints are in statement.f, not f_prime.
    // Combined F layout: [binary F (Ajtai)][arithmetic F (Ajtai_t + Ajtai_td + aggregation)]
    // The last l_arithmetic F functions are the aggregation constraints (f̃_j - g_j = 0).
    let f = &reduction.statement.f;
    if f.len() < reduction.l_arithmetic {
        return Err(format!(
            "F has {} functions, need at least {} for arithmetic g_j extraction",
            f.len(),
            reduction.l_arithmetic
        ));
    }
    let m = reduction.arithmetic_modulus_m;
    let agg_start = f.len() - reduction.l_arithmetic;
    for j in 0..reduction.l_arithmetic {
        let g_j_poly = &f[agg_start + j].b();
        if !check_divisible_by_x_minus_2(g_j_poly, m) {
            return Err(format!(
                "arithmetic g_{}(2) not divisible by M={} (mod M check failed)",
                j, m
            ));
        }
    }

    // 4. Arithmetic NAF coefficient bounds: check ALL arithmetic parts (a,b,c,w,d_1..d_l)
    // Combined witness layout: [binary 8 parts][arith a][arith b][arith c][arith w][arith d_1..d_l]
    // All NAF-encoded arithmetic parts must have coefficients in {-1, 0, 1}.
    let binary_parts = 8; // a,b,c,w,ã,b̃,c̃,w̃
    let arith_total = 4 + reduction.l_arithmetic; // a,b,c,w + d_1..d_l
    if binary_parts + arith_total > reduction.witness.parts.len() {
        return Err(format!(
            "witness has {} parts, need {} for arithmetic NAF check",
            reduction.witness.parts.len(),
            binary_parts + arith_total
        ));
    }
    for (part_idx, arith_part) in reduction
        .witness
        .parts
        .iter()
        .skip(binary_parts)
        .take(arith_total)
        .enumerate()
    {
        for (poly_idx, poly) in arith_part.iter().enumerate() {
            if verify_naf_coeffs::<R, N>(poly).is_none() {
                return Err(format!(
                    "arithmetic part {} NAF coefficient out of bounds at poly index {}",
                    part_idx, poly_idx
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::CyclotomicPolyRing;

    type F = PrimeField<12289>;

    fn zero_poly() -> CyclotomicPolyRing<F, 8> {
        Ring::zero()
    }

    #[test]
    fn test_mixed_r1cs_reduction_smoke() {
        let k = 1;
        let n = 1;

        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            c_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
        };

        // Arithmetic: A=w, B=w, C=w*w mod M. With witness=[3], a=3, b=3, a∘b=9.
        // C must give c=9, so C=[[3]] (C·w = 3*3 = 9 mod 17).
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            c_r1cs: RingMat::new(k, n, vec![F::from_u64(3)]),
            modulus_m: 257,
        };

        let instance = MixedR1CSInstance { binary, arithmetic };
        let witness_binary = vec![F::from_u64(1)];
        let witness_arith = vec![F::from_u64(3)];

        let l_binary = 1;
        let l_arithmetic = 1;

        let total_rank_bin = 3 * k + n;
        let total_rank_arith = 3 * k + n;

        let crs_a_binary = RingMat::new(2, total_rank_bin, vec![zero_poly(); 2 * total_rank_bin]);
        let crs_a_arith =
            RingMat::new(2, total_rank_arith, vec![zero_poly(); 2 * total_rank_arith]);
        let crs_b_arith =
            RingMat::new(2, l_arithmetic * k, vec![zero_poly(); 2 * l_arithmetic * k]);

        let mut rng = grid_std::test_rng();
        let reduction = build_mixed_r1cs_reduction::<F, _, 8>(
            &instance,
            &witness_binary,
            &witness_arith,
            &crs_a_binary,
            &crs_a_arith,
            &crs_b_arith,
            &mut rng,
            l_binary,
            l_arithmetic,
        )
        .unwrap();

        let binary_parts = 8; // a,b,c,w,ã,b̃,c̃,w̃
        let arith_parts = 4 + l_arithmetic; // a,b,c,w,d_1
        assert_eq!(reduction.witness.num_parts(), binary_parts + arith_parts);

        // All parts share the same rank
        let rank = reduction.witness.rank();
        for (i, part) in reduction.witness.parts.iter().enumerate() {
            assert_eq!(part.len(), rank, "part {} rank mismatch", i);
        }

        // All phi vectors match the part rank
        for (fi, f) in reduction
            .statement
            .f
            .iter()
            .chain(reduction.statement.f_prime.iter())
            .enumerate()
        {
            if let crate::relation::QuadraticFunction::Dense(d) = f {
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    assert_eq!(phi_i.len(), rank, "F[{}] phi[{}] length mismatch", fi, pi);
                }
                // Verify indices are within bounds
                for &(i, j) in &d.ij {
                    assert!(
                        i < reduction.witness.num_parts(),
                        "quad index {} out of bounds",
                        i
                    );
                    assert!(
                        j < reduction.witness.num_parts(),
                        "quad index {} out of bounds",
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_mixed_r1cs_mismatched_ranks() {
        // Binary: k=1, n=1 -> max_rank=1, binary has 8 parts
        // Arithmetic: k=2, n=3 -> max_rank=3, arith has 4+l=5 parts
        let k_bin = 1;
        let n_bin = 1;
        let k_arith = 2;
        let n_arith = 3;
        let l_arithmetic = 1;

        let binary = BinaryR1CSInstance {
            a_r1cs: RingMat::new(k_bin, n_bin, vec![F::from_u64(1)]),
            b_r1cs: RingMat::new(k_bin, n_bin, vec![F::from_u64(1)]),
            c_r1cs: RingMat::new(k_bin, n_bin, vec![F::from_u64(1)]),
        };

        // Arithmetic: construct A,B,C so that a∘b=c.
        // Row 0: a[0]=w[0]=1, b[0]=w[1]=2, c[0]=w[2]=2 (since 1*2=2 mod 17)
        // Row 1: trivial zeros (0*0=0)
        let arithmetic = ArithR1CSInstance {
            a_r1cs: RingMat::new(
                k_arith,
                n_arith,
                vec![
                    F::from_u64(1),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                ],
            ),
            b_r1cs: RingMat::new(
                k_arith,
                n_arith,
                vec![
                    F::from_u64(0),
                    F::from_u64(1),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                ],
            ),
            c_r1cs: RingMat::new(
                k_arith,
                n_arith,
                vec![
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(1),
                    F::from_u64(0),
                    F::from_u64(0),
                    F::from_u64(0),
                ],
            ),
            modulus_m: 257,
        };

        let instance = MixedR1CSInstance { binary, arithmetic };
        let witness_binary = vec![F::from_u64(1)];
        // Witness: w=[1,2,2] where w[0]*w[1]=1*2=2=w[2]
        let witness_arith = vec![F::from_u64(1), F::from_u64(2), F::from_u64(2)];

        let total_rank_bin = 3 * k_bin + n_bin; // 4
        let total_rank_arith = 3 * k_arith + n_arith; // 9

        let crs_a_binary = RingMat::new(2, total_rank_bin, vec![zero_poly(); 2 * total_rank_bin]);
        let crs_a_arith =
            RingMat::new(2, total_rank_arith, vec![zero_poly(); 2 * total_rank_arith]);
        let crs_b_arith = RingMat::new(
            2,
            l_arithmetic * k_arith,
            vec![zero_poly(); 2 * l_arithmetic * k_arith],
        );

        let mut rng = grid_std::test_rng();
        let reduction = build_mixed_r1cs_reduction::<F, _, 8>(
            &instance,
            &witness_binary,
            &witness_arith,
            &crs_a_binary,
            &crs_a_arith,
            &crs_b_arith,
            &mut rng,
            1,
            l_arithmetic,
        )
        .unwrap();

        let binary_parts = 8; // a,b,c,w,ã,b̃,c̃,w̃
        let arith_parts = 4 + l_arithmetic; // a,b,c,w,d_1
        assert_eq!(reduction.witness.num_parts(), binary_parts + arith_parts);

        // All parts share the same combined rank (max of binary_rank=1, arith_rank=3)
        let combined_rank = reduction.witness.rank();
        assert_eq!(combined_rank, 3, "combined rank should be max(1, 3) = 3");
        for (i, part) in reduction.witness.parts.iter().enumerate() {
            assert_eq!(part.len(), combined_rank, "part {} rank mismatch", i);
        }

        // All Dense phi vectors have correct length and count
        // Sparse functions (conjugacy) have individual entries, not full phi vectors
        for (fi, f) in reduction
            .statement
            .f
            .iter()
            .chain(reduction.statement.f_prime.iter())
            .enumerate()
        {
            if let crate::relation::QuadraticFunction::Dense(d) = f {
                assert_eq!(
                    d.phi.len(),
                    binary_parts + arith_parts,
                    "F[{}] phi count mismatch",
                    fi
                );
                for (pi, phi_i) in d.phi.iter().enumerate() {
                    assert_eq!(
                        phi_i.len(),
                        combined_rank,
                        "F[{}] phi[{}] length mismatch",
                        fi,
                        pi
                    );
                }
                for &(i, j) in &d.ij {
                    assert!(i < reduction.witness.num_parts(), "quad index {} OOB", i);
                    assert!(j < reduction.witness.num_parts(), "quad index {} OOB", j);
                }
            }
        }
    }

    #[test]
    fn test_verify_mixed_r1cs_reduction_positive() {
        let k = 1;
        let n = 1;
        let l_binary = 1;
        let l_arithmetic = 1;

        let instance = MixedR1CSInstance {
            binary: BinaryR1CSInstance {
                a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                c_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            },
            arithmetic: ArithR1CSInstance {
                a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                c_r1cs: RingMat::new(k, n, vec![F::from_u64(3)]),
                modulus_m: 257,
            },
        };

        let total_rank_bin = 3 * k + n;
        let total_rank_arith = 3 * k + n;
        let crs_a_binary = RingMat::new(2, total_rank_bin, vec![zero_poly(); 2 * total_rank_bin]);
        let crs_a_arith =
            RingMat::new(2, total_rank_arith, vec![zero_poly(); 2 * total_rank_arith]);
        let crs_b_arith =
            RingMat::new(2, l_arithmetic * k, vec![zero_poly(); 2 * l_arithmetic * k]);

        let mut rng = grid_std::test_rng();
        let reduction = build_mixed_r1cs_reduction::<F, _, 8>(
            &instance,
            &[F::from_u64(1)],
            &[F::from_u64(3)],
            &crs_a_binary,
            &crs_a_arith,
            &crs_b_arith,
            &mut rng,
            l_binary,
            l_arithmetic,
        )
        .unwrap();

        assert!(
            verify_mixed_r1cs_reduction::<F, 8>(&reduction, 1e6).is_ok(),
            "verify_mixed_r1cs_reduction should pass for honest reduction"
        );
    }

    #[test]
    fn test_verify_mixed_r1cs_reduction_reject_divisibility() {
        let k = 1;
        let n = 1;
        let l_binary = 1;
        let l_arithmetic = 1;

        let instance = MixedR1CSInstance {
            binary: BinaryR1CSInstance {
                a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                c_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
            },
            arithmetic: ArithR1CSInstance {
                a_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                b_r1cs: RingMat::new(k, n, vec![F::from_u64(1)]),
                c_r1cs: RingMat::new(k, n, vec![F::from_u64(3)]),
                modulus_m: 257,
            },
        };

        let total_rank_bin = 3 * k + n;
        let total_rank_arith = 3 * k + n;
        let crs_a_binary = RingMat::new(2, total_rank_bin, vec![zero_poly(); 2 * total_rank_bin]);
        let crs_a_arith =
            RingMat::new(2, total_rank_arith, vec![zero_poly(); 2 * total_rank_arith]);
        let crs_b_arith =
            RingMat::new(2, l_arithmetic * k, vec![zero_poly(); 2 * l_arithmetic * k]);

        let mut rng = grid_std::test_rng();
        let mut reduction = build_mixed_r1cs_reduction::<F, _, 8>(
            &instance,
            &[F::from_u64(1)],
            &[F::from_u64(3)],
            &crs_a_binary,
            &crs_a_arith,
            &crs_b_arith,
            &mut rng,
            l_binary,
            l_arithmetic,
        )
        .unwrap();

        // Isolate the g_j(2) mod M divisibility branch by constructing a relation-valid
        // F equation where b equals an NAF-encoded witness polynomial, but b(2) ≠ 0 mod M.
        // Paper §5: F proves g_j was computed correctly from the witness; external check
        // g_j(2) = 0 mod M catches invalid arithmetic witnesses (a*b ≠ c mod M).
        //
        // Strategy: replace the last aggregation F with a trivial identity:
        // phi[arith_w] = [1, 0, 0, ...], b = witness[arith_w][0]
        // F equation: 1 * w_poly - w_poly = 0 (passes relation::verify)
        // But w_poly encodes field element 3, so w_poly(2) = 3 ≠ 0 mod 257 (fails divisibility)
        let f = &mut reduction.statement.f;
        let binary_parts = 8;
        let arith_w_part = binary_parts + 3; // arithmetic w is part index 3 within arith
        if let Some(last_f) = f.last_mut() {
            let dense = last_f.expect_dense_mut();
            let num_parts = dense.phi.len();
            let rank = dense.phi[0].len();
            // Clear quadratic terms
            dense.ij.clear();
            dense.a.clear();
            // Zero all phi, then set phi[arith_w][0] = 1
            dense.phi = vec![vec![zero_poly(); rank]; num_parts];
            dense.phi[arith_w_part][0] = CyclotomicPolyRing::<F, 8>::one();
            // Set b = witness polynomial (NAF encoding of 3, so b(2) = 3 ≠ 0 mod 257)
            let w_poly = &reduction.witness.parts[arith_w_part][0];
            dense.b = w_poly.clone();
        }

        let result = verify_mixed_r1cs_reduction::<F, 8>(&reduction, 1e6);
        assert!(
            result.is_err(),
            "verify_mixed_r1cs_reduction should reject via g_j(2) mod M: {:?}",
            result
        );
        // Confirm rejection came from the divisibility check, not relation::verify
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("not divisible by M"),
            "Expected divisibility error, got: {}",
            err_msg
        );
        // F equation passes (w_poly - w_poly = 0), but g_j(2) = 3 ≠ 0 mod 257.
        // This exercises the external divisibility branch (paper §5 Figure 5).
    }
}
