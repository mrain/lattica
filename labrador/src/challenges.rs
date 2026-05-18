//! Challenge sampler for LaBRADOR (§2).
//!
//! Draws polynomials `c ∈ R_q` with a fixed coefficient weight distribution
//! (e.g., 23 zeros, 31 ±1s, 10 ±2s for d=64), then rejects if
//! `||c||_op > T` (operator norm threshold).
//!
//! Operator norm: `||c||_op = max_k |c(ζ^k)|` where ζ^k are the roots of X^d+1.
//! This equals the L∞ norm of the DFT of the coefficient vector over the
//! complex embeddings.
//!
//! Key property (LS18 Cor 1.2): for any distinct `c₁, c₂ ∈ C`,
//! `c₁ - c₂` is invertible in `R_q`, enabling witness extraction.

use core::array::from_fn;

use chacha20::{ChaCha20, cipher::KeyIvInit, cipher::StreamCipher};
use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};
use grid_std::rand::RngExt;
use grid_transcript::Transcript;

use crate::error::LabradorError;
use crate::params::ChallengeProfile;

/// Folding challenge sampler configured with a [`ChallengeProfile`].
///
/// Samples polynomials with a fixed coefficient weight distribution
/// (e.g., 23 zeros, 31 ±1s, 10 ±2s for d=64), then rejects if
/// `||c||_op > T` (operator norm threshold).
///
/// These "folding" challenges are used in the amortization step (§5.2) to fold
/// `r` witness openings `s_1..s_r` into one: `z = Σ c_i * s_i`.
/// The fixed-weight distribution ensures:
/// - `c_1 - c_2` is invertible for distinct `c_1, c_2` (soundness extraction)
/// - `||c||_op ≤ T` bounds norm blowup of the amortized opening
///
/// Generic over the ring degree `N`. The coefficient ring `R` is specified
/// at call time on `sample()`.
///
/// The sampler precomputes roots of X^N+1 for fast operator norm evaluation,
/// avoiding per-sample trig calls.
#[derive(Clone)]
pub struct FoldingChallengeSampler<const N: usize> {
    profile: ChallengeProfile,
    /// Precomputed magnitude array: sorted {0, 1, 2} values to fill the polynomial.
    magnitudes: [i8; N],
    /// First N/2 roots of X^N+1: (cos, sin) pairs. Conjugate symmetry cuts eval in half.
    roots: alloc::vec::Vec<(f64, f64)>,
}

impl<const N: usize> core::fmt::Debug for FoldingChallengeSampler<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FoldingChallengeSampler")
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

impl<const N: usize> FoldingChallengeSampler<N> {
    /// Create a sampler from a challenge profile.
    ///
    /// # Panics
    /// Panics if `N` is not a power of two, `profile.shape.degree() != N`,
    /// or `profile.T` is too low to accept any polynomial from this shape.
    ///
    /// On `std`, performs a bounded liveness probe (1000 attempts) to verify
    /// that at least one polynomial from this shape passes the operator norm
    /// threshold. If the probe fails, the sampler would loop forever on
    /// `sample()` calls.
    pub fn new(profile: ChallengeProfile) -> Self {
        // N must be a power of two ≥ 2 (X^N+1 is cyclotomic only for power-of-two N ≥ 2).
        assert!(
            N > 1 && (N & (N - 1)) == 0,
            "ring degree N ({}) must be a power of two ≥ 2",
            N
        );

        assert_eq!(
            profile.shape.degree(),
            N,
            "challenge shape degree ({}) must equal ring degree ({})",
            profile.shape.degree(),
            N
        );

        // Sanity: T must be at least the L2 norm of the coefficients,
        // since |c(ζ)| ≥ |c[0]| for some root ζ. For a shape with all
        // coefficients non-zero, the minimum possible operator norm is
        // the sum of absolute values divided by √N (achieved by a
        // constant-magnitude DFT). Conservatively require T ≥ √τ.
        let min_t = grid_std::sqrt(profile.tau());
        assert!(
            profile.T >= min_t,
            "operator norm threshold T ({}) is too low for this shape (min possible ≈ {:.1} = √τ); sampling would never accept",
            profile.T,
            min_t,
        );

        // Build the magnitude array: zeros first, then ones, then twos.
        // Array is zero-initialized; only fill ones and twos.
        let mut magnitudes = [0i8; N];
        let shape = &profile.shape;
        let mut idx = shape.zeros;

        for _ in 0..shape.ones {
            magnitudes[idx] = 1;
            idx += 1;
        }
        for _ in 0..shape.twos {
            magnitudes[idx] = 2;
            idx += 1;
        }

        // Precompute first N/2 roots of X^N+1 (conjugate symmetry).
        let pi_over_n = core::f64::consts::PI / N as f64;
        let half = N / 2;
        let mut roots = alloc::vec::Vec::with_capacity(half);
        for k in 0..half {
            let angle = (2.0 * k as f64 + 1.0) * pi_over_n;
            roots.push((grid_std::cos(angle), grid_std::sin(angle)));
        }

        // Liveness probe (std only): verify at least one polynomial from
        // this shape passes the threshold. For the paper profile (d=64, T=15),
        // ~6 attempts suffice on average. If probe fails, sampling would
        // loop forever.
        #[cfg(feature = "std")]
        {
            let mut probe_rng = grid_std::test_rng();
            let mut found = false;
            for _ in 0..1000 {
                let mut coeffs = magnitudes;
                fisher_yates_shuffle(&mut coeffs, &mut probe_rng);
                for c in &mut coeffs {
                    if *c != 0 && probe_rng.random() {
                        *c = -*c;
                    }
                }
                if evaluate_op_norm(&coeffs, &roots) <= profile.T {
                    found = true;
                    break;
                }
            }
            if !found {
                panic!(
                    "folding challenge sampler liveness probe failed after \
                     1000 attempts for shape {{{:?}}} and T={:.1}. \
                     Sampling would loop indefinitely. Raise T or adjust the shape.",
                    profile.shape, profile.T
                );
            }
        }

        Self {
            profile,
            magnitudes,
            roots,
        }
    }

    /// Evaluate operator norm: max_k |Σ_j c[j] · (ζ^k)^j|.
    /// Uses precomputed roots and conjugate symmetry (only first N/2).
    #[inline]
    fn operator_norm(&self, coeffs: &[i8; N]) -> f64 {
        evaluate_op_norm(coeffs, &self.roots)
    }

    /// Return the profile this sampler uses.
    pub fn profile(&self) -> &ChallengeProfile {
        &self.profile
    }

    /// Sample a single folding challenge polynomial from an RNG.
    ///
    /// Repeatedly draws fixed-weight polynomials until one satisfies
    /// `||c||_op ≤ T`. The paper reports ~6 attempts on average for d=64, T=15.
    ///
    /// # Panics
    /// Panics if no polynomial from this shape satisfies the threshold
    /// (should be caught by the liveness probe in `new()`).
    pub fn sample<R, Rng>(&self, rng: &mut Rng) -> CyclotomicPolyRing<R, N>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        Rng: RngExt,
    {
        self.try_sample(rng, 1000)
            .expect("folding challenge sampler exhausted — profile may be infeasible")
    }

    /// Bounded variant of [`Self::sample`].
    ///
    /// Returns `None` if `max_attempts` draws are exhausted without finding
    /// a polynomial that satisfies `||c||_op ≤ T`. Use this in the main
    /// protocol to avoid unbounded loops on adversarial or misconfigured profiles.
    pub fn try_sample<R, Rng>(
        &self,
        rng: &mut Rng,
        max_attempts: usize,
    ) -> Option<CyclotomicPolyRing<R, N>>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        Rng: RngExt,
    {
        let threshold = self.profile.T;
        for _ in 0..max_attempts {
            let (coeffs_i8, coeffs_ring) = self.draw_once::<R, Rng>(rng);
            if self.operator_norm(&coeffs_i8) <= threshold {
                return Some(coeffs_ring);
            }
        }
        None
    }

    /// Sample a single folding challenge polynomial from a transcript.
    ///
    /// Derives a 32-byte seed from the transcript, uses it to seed a
    /// ChaCha20 RNG for the Fisher-Yates shuffle + rejection sampling loop.
    /// The transcript label is used for domain separation.
    /// On rejection, the nonce is incremented (not re-squeezed) so the
    /// transcript advances by exactly one step regardless of rejection count.
    pub fn sample_transcript<R, T>(
        &self,
        transcript: &mut T,
        label: &'static [u8],
    ) -> Result<CyclotomicPolyRing<R, N>, LabradorError>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        T: Transcript,
    {
        self.try_sample_transcript(transcript, label, 1000)
    }

    /// Bounded variant of [`Self::sample_transcript`].
    ///
    /// Returns `Err(LabradorError::SamplerExhausted)` if `max_attempts` draws
    /// are exhausted. Returns `Err(LabradorError::Transcript(..))` if the
    /// transcript itself fails.
    pub fn try_sample_transcript<R, T>(
        &self,
        transcript: &mut T,
        label: &'static [u8],
        max_attempts: usize,
    ) -> Result<CyclotomicPolyRing<R, N>, LabradorError>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        T: Transcript,
    {
        let threshold = self.profile.T;
        // Squeeze once — transcript advances exactly one step.
        let seed_bytes = transcript.challenge_bytes(label, 32)?;
        let key = <[u8; 32]>::try_from(seed_bytes.as_slice())
            .expect("challenge_bytes returns exactly 32 bytes");

        // Increment nonce on each rejection to get independent keystreams.
        let mut nonce_counter = 0u32;
        for _ in 0..max_attempts {
            let mut nonce = [0u8; 12];
            nonce[..4].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);

            let mut rng = ChaCha20::new(&key.into(), &nonce.into());
            let (coeffs_i8, coeffs_ring) = self.draw_once_chacha::<R>(&mut rng);
            if self.operator_norm(&coeffs_i8) <= threshold {
                return Ok(coeffs_ring);
            }
        }
        Err(LabradorError::SamplerExhausted)
    }

    /// Sample `count` folding challenge polynomials from an RNG.
    pub fn sample_batch<R, Rng>(
        &self,
        rng: &mut Rng,
        count: usize,
    ) -> alloc::vec::Vec<CyclotomicPolyRing<R, N>>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        Rng: RngExt,
    {
        let mut challenges = alloc::vec::Vec::with_capacity(count);
        for _ in 0..count {
            challenges.push(self.sample(rng));
        }
        challenges
    }

    /// Sample `count` folding challenge polynomials from a transcript.
    pub fn sample_batch_transcript<R, T>(
        &self,
        transcript: &mut T,
        label: &'static [u8],
        count: usize,
    ) -> Result<alloc::vec::Vec<CyclotomicPolyRing<R, N>>, LabradorError>
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        T: Transcript,
    {
        let mut challenges = alloc::vec::Vec::with_capacity(count);
        for _ in 0..count {
            challenges.push(self.sample_transcript(transcript, label)?);
        }
        Ok(challenges)
    }

    /// Draw a single fixed-weight polynomial from an RNG.
    /// Returns both the signed i8 coefficients (for operator norm check)
    /// and the ring-element polynomial.
    fn draw_once<R, Rng>(&self, rng: &mut Rng) -> ([i8; N], CyclotomicPolyRing<R, N>)
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
        Rng: RngExt,
    {
        // Copy magnitudes and randomize: shuffle positions, assign signs.
        let mut coeffs = self.magnitudes;
        fisher_yates_shuffle(&mut coeffs, rng);

        // Apply random signs (0 stays 0, ±1 and ±2 get uniform sign).
        for c in &mut coeffs {
            if *c != 0 && rng.random() {
                *c = -*c;
            }
        }

        let signed_coeffs = coeffs;
        let ring_coeffs: [R; N] = from_fn(|i| encode_signed::<R>(coeffs[i] as i64));

        (signed_coeffs, CyclotomicPolyRing::from_array(ring_coeffs))
    }

    /// Draw a single fixed-weight polynomial from a ChaCha20 stream cipher.
    /// Used internally by `sample_transcript` to bridge transcript → RNG → sampler.
    fn draw_once_chacha<R>(&self, cipher: &mut ChaCha20) -> ([i8; N], CyclotomicPolyRing<R, N>)
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
    {
        // Copy magnitudes and randomize: shuffle positions, assign signs.
        let mut coeffs = self.magnitudes;
        fisher_yates_shuffle_chacha(&mut coeffs, cipher);

        // Apply random signs (0 stays 0, ±1 and ±2 get uniform sign).
        for c in &mut coeffs {
            if *c != 0 && random_bool_chacha(cipher) {
                *c = -*c;
            }
        }

        let signed_coeffs = coeffs;
        let ring_coeffs: [R; N] = from_fn(|i| encode_signed::<R>(coeffs[i] as i64));

        (signed_coeffs, CyclotomicPolyRing::from_array(ring_coeffs))
    }
}

/// Encode a signed integer into a ring element.
fn encode_signed<R: IntegerRing<Uint = u64>>(v: i64) -> R {
    if v >= 0 {
        R::from_u64(v as u64)
    } else {
        let abs = (-v) as u64;
        let modulus = R::modulus();
        debug_assert_ne!(modulus, 0, "IntegerRing modulus must be non-zero");
        let reduced = abs % modulus;
        if reduced == 0 {
            R::zero()
        } else {
            R::from_u64(modulus - reduced)
        }
    }
}

/// Fisher-Yates shuffle for in-place permutation.
fn fisher_yates_shuffle<T: RngExt>(arr: &mut [i8], rng: &mut T) {
    let len = arr.len();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        arr.swap(i, j);
    }
}

/// Fisher-Yates shuffle using ChaCha20 stream cipher with rejection sampling.
///
/// Uses unbiased rejection sampling to select a uniform index in `[0..=i]`,
/// avoiding modulo bias. Draws 4 bytes from the cipher, interprets as u32,
/// and rejects values above the largest multiple of `(i+1)` that fits in u32.
fn fisher_yates_shuffle_chacha(arr: &mut [i8], cipher: &mut ChaCha20) {
    let len = arr.len();
    for i in (1..len).rev() {
        let limit = (i + 1) as u32;
        let threshold = u32::MAX - (u32::MAX % limit);
        let mut val = u32::MAX;
        while val >= threshold {
            let mut buf = [0u8; 4];
            cipher.apply_keystream(&mut buf);
            val = u32::from_le_bytes(buf);
        }
        let j = (val % limit) as usize;
        arr.swap(i, j);
    }
}

/// Draw a single random bit from ChaCha20 stream cipher.
fn random_bool_chacha(cipher: &mut ChaCha20) -> bool {
    let mut buf = [0u8; 1];
    cipher.apply_keystream(&mut buf);
    buf[0] & 1 == 1
}

/// Evaluate operator norm of a polynomial given precomputed roots.
fn evaluate_op_norm<const N: usize>(coeffs: &[i8; N], roots: &[(f64, f64)]) -> f64 {
    // NOTE: This function relies on f64 cos/sin/hypot from libm. Cross-platform
    // differences in libm rounding (glibc vs musl, x86 FMA vs strict IEEE-754)
    // could produce different last-bit results. The threshold T is set with
    // sufficient margin (T=15, typical norms ~12-14) to absorb drift. If this
    // sampler runs on mixed platforms (e.g., prover on x86, verifier on ARM),
    // ensure both sides agree on the same libm or increase T conservatively.
    let mut max_val = 0.0f64;
    for &(root_re, root_im) in roots {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let mut power_re = 1.0f64;
        let mut power_im = 0.0f64;
        for &c in coeffs {
            let c = c as f64;
            if c != 0.0 {
                re += c * power_re;
                im += c * power_im;
            }
            let new_re = power_re * root_re - power_im * root_im;
            let new_im = power_re * root_im + power_im * root_re;
            power_re = new_re;
            power_im = new_im;
        }
        let mag = grid_std::hypot(re, im);
        if mag > max_val {
            max_val = mag;
        }
    }
    max_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ChallengeProfile, ChallengeShape};

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::poly::ring::PolyRing;

    type F = PrimeField<12289>;

    #[test]
    fn test_d64_shape_tau() {
        let profile = ChallengeProfile::paper_default();
        // tau = 31·1² + 10·2² = 71
        assert_eq!(profile.shape.tau(), 71.0);
        assert_eq!(profile.shape.degree(), 64);
    }

    #[test]
    fn test_d64_sample_l2_norm() {
        let profile = ChallengeProfile::paper_default();
        let sampler = FoldingChallengeSampler::<64>::new(profile);

        let mut rng = grid_std::test_rng();
        let poly: CyclotomicPolyRing<F, 64> = sampler.sample(&mut rng);

        // Verify squared L2 norm equals tau=71.
        let mut sq_sum: i64 = 0;
        for i in 0..64 {
            let raw = poly.coeff(i).to_u64() as i64;
            let centered = if raw > 12289 / 2 { raw - 12289 } else { raw };
            sq_sum += centered * centered;
        }
        assert_eq!(sq_sum, 71, "squared L2 norm must equal tau");
    }

    #[test]
    fn test_d64_operator_norm_within_threshold() {
        let profile = ChallengeProfile::paper_default();
        let sampler = FoldingChallengeSampler::<64>::new(profile.clone());

        let mut rng = grid_std::test_rng();
        let poly: CyclotomicPolyRing<F, 64> = sampler.sample(&mut rng);

        // Convert poly to i8 for operator norm check.
        let coeffs = poly_to_i8::<F, 64>(&poly);
        let op = sampler.operator_norm(&coeffs);
        assert!(
            op <= profile.T,
            "operator norm {} exceeds threshold {}",
            op,
            profile.T
        );
    }

    #[test]
    fn test_operator_norm_known_polys() {
        let cases = [
            // c = 1 (constant) has ||c||_op = 1.
            (
                ChallengeShape {
                    zeros: 3,
                    ones: 1,
                    twos: 0,
                },
                [1, 0, 0, 0],
                1.0,
            ),
            // c = 1 - X³ in Z[X]/(X⁴+1): coeffs = [1, 0, 0, -1]
            // |c(root_k)| = |1 - root_k³|, max over k=0..3
            // Expected max: sqrt(2 + sqrt(2)) ≈ 1.848
            (
                ChallengeShape {
                    zeros: 2,
                    ones: 2,
                    twos: 0,
                },
                [1, 0, 0, -1],
                grid_std::sqrt(2.0 + grid_std::sqrt(2.0)),
            ),
        ];

        for (shape, coeffs, expected) in &cases {
            let sampler = FoldingChallengeSampler::<4>::new(ChallengeProfile {
                shape: shape.clone(),
                T: 10.0,
                space_bits: 8,
            });
            let op = sampler.operator_norm(coeffs);
            assert!(
                (op - expected).abs() < 1e-10,
                "shape={:?} coeffs={:?} expected {:.10}, got {:.10}",
                shape,
                coeffs,
                expected,
                op
            );
        }
    }

    #[test]
    fn test_sampler_produces_distinct_challenges() {
        let profile = ChallengeProfile {
            shape: ChallengeShape {
                zeros: 1,
                ones: 2,
                twos: 1,
            },
            T: 10.0,
            space_bits: 8,
        };
        let sampler = FoldingChallengeSampler::<4>::new(profile);

        let mut rng = grid_std::test_rng();
        let challenges: alloc::vec::Vec<_> =
            (0..20).map(|_| sampler.sample::<F, _>(&mut rng)).collect();

        // Count distinct polynomials.
        let mut distinct = 0;
        for i in 0..challenges.len() {
            let mut is_new = true;
            for j in 0..i {
                if challenges[i] == challenges[j] {
                    is_new = false;
                    break;
                }
            }
            if is_new {
                distinct += 1;
            }
        }
        assert!(
            distinct >= 15,
            "expected at least 15 distinct challenges out of 20, got {}",
            distinct
        );
    }

    #[test]
    fn test_batch_sampling() {
        let profile = ChallengeProfile::paper_default();
        let sampler = FoldingChallengeSampler::<64>::new(profile.clone());

        let mut rng = grid_std::test_rng();
        let batch = sampler.sample_batch::<F, _>(&mut rng, 5);
        assert_eq!(batch.len(), 5);

        // All should be within threshold.
        for poly in &batch {
            let coeffs = poly_to_i8::<F, 64>(poly);
            let op = sampler.operator_norm(&coeffs);
            assert!(op <= profile.T);
        }
    }

    #[test]
    fn test_coefficient_distribution() {
        // Over many samples, coefficient magnitudes should approach
        // the profile distribution: 23 zeros, 31 ones, 10 twos.
        let profile = ChallengeProfile::paper_default();
        let sampler = FoldingChallengeSampler::<64>::new(profile);

        let mut rng = grid_std::test_rng();
        let n_samples = 100;

        let mut total_zeros = 0usize;
        let mut total_ones = 0usize;
        let mut total_twos = 0usize;

        for _ in 0..n_samples {
            let poly: CyclotomicPolyRing<F, 64> = sampler.sample(&mut rng);
            for i in 0..64 {
                let raw = poly.coeff(i).to_u64() as i64;
                let centered = if raw > 12289 / 2 { raw - 12289 } else { raw };
                let abs = centered.abs();
                if abs == 0 {
                    total_zeros += 1;
                } else if abs == 1 {
                    total_ones += 1;
                } else if abs == 2 {
                    total_twos += 1;
                }
            }
        }

        // Each sample has exactly 23 zeros, 31 ones, 10 twos.
        assert_eq!(total_zeros, 23 * n_samples, "zero count mismatch");
        assert_eq!(total_ones, 31 * n_samples, "one count mismatch");
        assert_eq!(total_twos, 10 * n_samples, "two count mismatch");
    }

    /// Convert a polynomial to signed i8 coefficients (centered representatives).
    fn poly_to_i8<R, const N: usize>(poly: &CyclotomicPolyRing<R, N>) -> [i8; N]
    where
        R: IntegerRing<Uint = u64> + NegacyclicMulRing<N>,
    {
        let modulus = R::modulus();
        debug_assert_ne!(modulus, 0, "IntegerRing modulus must be non-zero");
        let half = (modulus as i64) / 2;
        from_fn(|i| {
            let raw = poly.coeff(i).to_u64() as i64;
            let centered = if raw > half {
                raw - modulus as i64
            } else {
                raw
            };
            centered as i8
        })
    }
}
