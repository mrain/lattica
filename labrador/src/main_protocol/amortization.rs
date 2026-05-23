//! Amortization (§5.2 step 16-17).
//!
//! Amortize r witness openings into one: z = Σ c_i · s_i.
//! z ∈ R_q^n (vector of rank n).

use alloc::vec;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::arith::ring::Ring;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_transcript::Transcript;

use crate::challenges::FoldingChallengeSampler;
use crate::error::LabradorError;
use crate::params::LabradorParams;
use crate::relation::LabradorWitness;

/// Result of amortization.
#[derive(Debug, Clone)]
pub struct AmortizationData<R, const N: usize>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    /// Amortized witness opening: z = Σ c_i · s_i, a vector of rank n.
    pub z: Vec<CyclotomicPolyRing<R, N>>,
    /// Folding challenges c_1..c_r.
    pub challenges: Vec<CyclotomicPolyRing<R, N>>,
}

/// Sample amortization challenges only (no z computation).
///
/// Returns the folding challenges c_1..c_r without computing z = Σ c_i·s_i.
/// Callers that need z should compute it separately via [`super::garbage::compute_amortized_witness`].
pub fn sample_amortization_challenges<R, const N: usize, T>(
    num_parts: usize,
    params: &LabradorParams,
    transcript: &mut T,
    level: usize,
) -> Result<Vec<CyclotomicPolyRing<R, N>>, LabradorError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: Transcript,
{
    let sampler = FoldingChallengeSampler::new(params.challenge.clone());
    let mut domain = [0u8; 8];
    domain[..4].copy_from_slice(&(level as u32).to_le_bytes());

    let mut challenges = Vec::with_capacity(num_parts);
    for i in 0..num_parts {
        domain[4..].copy_from_slice(&(i as u32).to_le_bytes());
        transcript.append_bytes(b"labrador_amortize_domain", &domain)?;
        let c_i = sampler.sample_transcript(transcript, b"labrador_amortize")?;
        challenges.push(c_i);
    }

    Ok(challenges)
}

/// Sample amortization challenges and compute amortized witness z = Σ c_i · s_i.
pub fn compute_amortized_witness<R, const N: usize, T>(
    witness: &LabradorWitness<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
    transcript: &mut T,
    level: usize,
) -> Result<AmortizationData<R, N>, LabradorError>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
    T: Transcript,
{
    let r = witness.num_parts();
    let n = witness.rank();
    let challenges = sample_amortization_challenges(r, params, transcript, level)?;

    // z[k] = Σ_i c_i · s_i[k]
    let mut z = vec![CyclotomicPolyRing::<R, N>::zero(); n];
    for (c_i, s_i) in challenges.iter().zip(witness.parts.iter()) {
        for (z_k, s_k) in z.iter_mut().zip(s_i.iter()) {
            *z_k += c_i * s_k;
        }
    }

    Ok(AmortizationData { z, challenges })
}

#[cfg(test)]
mod tests {
    use super::*;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::PolyRing;
    use grid_transcript::TranscriptError;

    type F = PrimeField<12289>;

    /// Fake transcript for testing amortization (no real transcript needed).
    struct TestTranscript {
        call_count: usize,
    }

    impl Transcript for TestTranscript {
        fn append_preframed_bytes(&mut self, _bytes: &[u8]) -> Result<(), TranscriptError> {
            Ok(())
        }

        fn challenge_bytes(
            &mut self,
            _label: &'static [u8],
            out_len: usize,
        ) -> Result<Vec<u8>, TranscriptError> {
            self.call_count += 1;
            let val = (self.call_count as u64).to_le_bytes();
            let mut out = Vec::with_capacity(out_len);
            for i in 0..out_len {
                out.push(val[i % val.len()]);
            }
            Ok(out)
        }
    }

    fn fake_params() -> LabradorParams {
        use crate::params::{ChallengeProfile, JLProfile};
        LabradorParams {
            jl: JLProfile::default(),
            challenge: ChallengeProfile::paper_default(),
            security_bits: 8,
            soundness_error: 0.0,
            l: 1,
            arith_p: 274177,
            n: 4,
            r: 3,
            beta: 100.0,
            d: 64,
            q: 12289.0,
            sigma: 1.0,
            b: 2,
            b1: 16,
            b2: 16,
            t1: 4,
            t2: 4,
            kappa: 2,
            kappa1: 2,
            kappa2: 2,
            gamma: 100.0,
            gamma1_sq: 10_000,
            gamma2_sq: 10_000,
            beta_prime: 100.0,
            nu: 1,
            mu: 1,
            num_levels: 1,
        }
    }

    fn make_witness(r: usize, n: usize) -> LabradorWitness<CyclotomicPolyRing<F, 64>> {
        LabradorWitness {
            parts: (0..r)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let mut p = CyclotomicPolyRing::<F, 64>::zero();
                            p.set_coeff(0, F::from_u64((i * n + j + 1) as u64));
                            p
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn test_amortization_zero_witness() {
        let params = fake_params();
        let zero_witness = LabradorWitness {
            parts: (0..params.r)
                .map(|_| {
                    (0..params.n)
                        .map(|_| CyclotomicPolyRing::<F, 64>::zero())
                        .collect()
                })
                .collect(),
        };

        let mut transcript = TestTranscript { call_count: 0 };
        let result =
            compute_amortized_witness::<F, 64, _>(&zero_witness, &params, &mut transcript, 0);
        assert!(result.is_ok());
        let data = result.unwrap();

        assert_eq!(data.challenges.len(), params.r);
        assert_eq!(data.z.len(), params.n);
        for z_k in &data.z {
            assert!(z_k.is_zero(), "z should be zero for zero witness");
        }
    }

    #[test]
    fn test_amortization_shapes_and_z_formula() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);

        let mut transcript = TestTranscript { call_count: 0 };
        let result = compute_amortized_witness::<F, 64, _>(&witness, &params, &mut transcript, 0);
        assert!(result.is_ok());
        let data = result.unwrap();

        assert_eq!(data.challenges.len(), params.r);
        assert_eq!(data.z.len(), params.n);

        for k in 0..params.n {
            let mut expected = CyclotomicPolyRing::<F, 64>::zero();
            for i in 0..params.r {
                expected += data.challenges[i].clone() * witness.parts[i][k].clone();
            }
            assert_eq!(data.z[k], expected, "z[{}] mismatch", k);
        }
    }

    #[test]
    fn test_amortization_challenges_unique() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);

        let mut transcript = TestTranscript { call_count: 0 };
        let result = compute_amortized_witness::<F, 64, _>(&witness, &params, &mut transcript, 0);
        let data = result.unwrap();

        // All challenges should be distinct (TestTranscript returns incrementing values)
        for i in 0..data.challenges.len() {
            for j in (i + 1)..data.challenges.len() {
                assert_ne!(
                    data.challenges[i], data.challenges[j],
                    "challenges[{}] should differ from challenges[{}]",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_sample_challenges_matches_full_amortization() {
        let params = fake_params();
        let witness = make_witness(params.r, params.n);

        // Sample challenges only
        let mut transcript1 = TestTranscript { call_count: 0 };
        let sampled =
            sample_amortization_challenges::<F, 64, _>(params.r, &params, &mut transcript1, 0)
                .unwrap();

        // Full amortization
        let mut transcript2 = TestTranscript { call_count: 0 };
        let full =
            compute_amortized_witness::<F, 64, _>(&witness, &params, &mut transcript2, 0).unwrap();

        assert_eq!(sampled.len(), full.challenges.len());
        for (i, (s, f)) in sampled.iter().zip(full.challenges.iter()).enumerate() {
            assert_eq!(s, f, "challenge[{}] mismatch", i);
        }
    }
}
