//! Johnson-Lindenstrauss projection matrix for LaBRADOR.
//!
//! Seed-based JL matrix `Π ∈ {-1, 0, 1}^{rows × cols}` where entries are derived
//! on-demand via per-part ChaCha20 streaming. One key schedule per part, sequential
//! scatter-accumulate into a `[u8; 64]` stack buffer — zero heap allocation per column.
//!
//! Entry distribution (Lemma 4.1, GHL21): `Pr[+1] = 1/4, Pr[-1] = 1/4, Pr[0] = 1/2`.
//! The 2-bit encoding maps `0b00→+1, 0b01→-1, 0b10→0, 0b11→0` (4 equally-likely codepoints).
//!
//! **Streaming construction:** `key = matrix.seed`, `nonce = part_idx(LE u32) || 0(96 bits)`.
//! Zero SHA3 calls, one key schedule per part (not per column), collision-free by design.

use alloc::vec;
use alloc::vec::Vec;

use chacha20::{
    ChaCha20,
    cipher::{KeyIvInit, StreamCipher},
};
use grid_algebra::arith::ring::IntegerRing;
use grid_std::rand::CryptoRng;

use crate::params::JLProfile;

/// Maximum JL retry count accepted by both prover and verifier.
/// The honest prover caps at this value; the verifier rejects proofs exceeding it.
pub const JL_MAX_RETRY: u32 = 100;

/// Seed-based Johnson-Lindenstrauss projection matrix `Π ∈ {-1, 0, 1}^{rows × cols}`.
///
/// Entries derived via per-part ChaCha20 streaming: one key schedule per part,
/// sequential 64-byte stack buffer per column. Zero heap allocation during projection.
#[derive(Debug, Clone)]
pub struct JLMatrix {
    /// ChaCha20 key for the projection matrix.
    seed: [u8; 32],
    /// Output dimension (256 from paper).
    rows: usize,
    /// Input dimension (n · d scalar coefficients per witness part).
    cols: usize,
}

impl JLMatrix {
    /// Sample a new JL projection matrix from the given profile and column count.
    ///
    /// The seed is drawn from the provided RNG. The column count should be `n * d`
    /// (witness rank × ring degree) for a single witness part.
    /// Panics if `cols == 0` (degenerate projection) or profile is invalid.
    pub fn sample<Rng: grid_std::rand::Rng + CryptoRng>(
        profile: &JLProfile,
        cols: usize,
        rng: &mut Rng,
    ) -> Self {
        assert!(cols > 0, "JL projection columns must be > 0 (n * d)");
        assert!(profile.validate(), "JL profile must be structurally valid");
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self {
            seed,
            rows: profile.rows,
            cols,
        }
    }

    /// Return the number of rows (output dimension).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the number of columns (input dimension).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Return the raw seed for this matrix.
    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    /// Construct from a known seed for Fiat-Shamir determinism.
    ///
    /// Used by the prover and verifier to reconstruct the same JL matrix
    /// from a transcript-derived seed, without needing an RNG.
    /// Panics if `cols == 0` or profile is invalid.
    pub fn from_seed(profile: &JLProfile, cols: usize, seed: [u8; 32]) -> Self {
        assert!(cols > 0, "JL projection columns must be > 0 (n * d)");
        assert!(profile.validate(), "JL profile must be structurally valid");
        Self {
            seed,
            rows: profile.rows,
            cols,
        }
    }

    /// Extract JL rows at polynomial granularity for aggregation.
    ///
    /// Paper §5.2 requires each JL entry `π_i^(m)[k]` to be a full polynomial in R_q,
    /// where the k-th coefficient of `π_i^(m)` is the vector of n JL scalars:
    ///
    ///   `π_i^(m)[k] = [ Π[m, part_i, 0*d+k], Π[m, part_i, 1*d+k], ..., Π[m, part_i, (n-1)*d+k] ]`
    ///
    /// The JL matrix stores flattened witness coefficients:
    ///   col j*d + k → k-th coefficient of the j-th polynomial within part_i
    ///
    /// Returns `jl_rows_poly[m][i][k]` where:
    /// - m: JL row index (0..rows)
    /// - i: witness part index (0..num_parts)
    /// - k: polynomial coefficient position (0..d)
    ///
    /// Each element `jl_rows_poly[m][i][k]` is a `Vec<i8>` of length `n = cols / d`,
    /// containing the JL scalar entries for the k-th coefficient of π_i^(m).
    ///
    /// The caller must apply σ₋₁ conjugation (ring automorphism) to each polynomial
    /// before using it in aggregation. Callers should use `crate::relation::conjugation`.
    pub fn extract_jl_rows_poly(&self, num_parts: usize, d: usize) -> Vec<Vec<Vec<Vec<i8>>>> {
        assert!(d > 0, "ring degree d must be > 0 (got {})", d);
        assert!(
            self.cols.is_multiple_of(d),
            "cols ({}) must be divisible by ring degree d ({})",
            self.cols,
            d
        );
        let n = self.cols / d; // witness rank (number of polynomials per part)
        let packed_len = self.rows.div_ceil(4);
        // Shape: [rows][parts][d][n]
        let mut result = vec![vec![vec![vec![0i8; n]; d]; num_parts]; self.rows];

        #[allow(clippy::needless_range_loop)]
        for part_idx in 0..num_parts {
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[0..4].copy_from_slice(&(part_idx as u32).to_le_bytes());
            let mut cipher = ChaCha20::new(&self.seed.into(), &nonce_bytes.into());

            let mut col_packed = [0u8; 64];
            for col in 0..self.cols {
                col_packed.fill(0);
                cipher.apply_keystream(&mut col_packed[..packed_len]);

                // col = j * d + k → j is poly index, k is coeff within poly
                let poly_idx = col / d;
                let coeff_idx = col % d;

                for row in 0..self.rows {
                    let byte_idx = row / 4;
                    let nibble = (row % 4) as u8;
                    let val = (col_packed[byte_idx] >> (nibble * 2)) & 0x03;
                    let entry = match val {
                        0b00 => 1i8,
                        0b01 => -1i8,
                        _ => 0i8,
                    };
                    // π_i^(m)[k][j] = Π[m, part_i, j*d + k]
                    result[row][part_idx][coeff_idx][poly_idx] = entry;
                }
            }
        }

        result
    }

    /// Compute the JL projection `p = Π · w mod q`.
    ///
    /// Equivalent to `project_multi(&[&w])` — the witness is treated as part index 0.
    /// `w` is a slice of `n * d` scalar ring coefficients. Returns a vector of
    /// `rows` ring elements (256 by default).
    pub fn project<R>(&self, w: &[R]) -> Vec<R>
    where
        R: IntegerRing<Canonical = u64>,
    {
        self.project_multi(&[w])
    }

    /// Compute the JL projection `p = Π · w mod q` into a pre-allocated output buffer.
    ///
    /// Equivalent to writing the `project_multi(&[&w])` result into `out`.
    /// Avoids allocation when the caller already has a buffer.
    pub fn project_into<R>(&self, w: &[R], out: &mut [R])
    where
        R: IntegerRing<Canonical = u64>,
    {
        self.project_multi_into(&[w], out);
    }

    /// Project multiple witness parts and sum their projections.
    ///
    /// Given witness parts `w_1, ..., w_k` (each of length `cols`), computes
    /// `p = Σ_j Π^{(j)} · w_j` where each `Π^{(j)}` is a matrix with the same
    /// seed but offset row derivation. This matches the LaBRADOR construction
    /// where `p = Σ_i Π_i · s_i`.
    ///
    /// The per-part matrices are derived via ChaCha20 streaming:
    /// `key = seed`, `nonce = part_idx(LE u32) || 0`, columns are keystream offsets.
    pub fn project_multi<R>(&self, parts: &[&[R]]) -> Vec<R>
    where
        R: IntegerRing<Canonical = u64>,
    {
        let mut p = vec![R::zero(); self.rows];
        self.project_multi_into(parts, &mut p);
        p
    }

    /// Internal: project multiple parts into a pre-allocated output buffer.
    ///
    /// Per-part streaming: one ChaCha20 key schedule per part, sequential scatter-accumulate
    /// into a `[u8; 64]` stack buffer. Columns processed in order — no seeking, no heap alloc.
    fn project_multi_into<R>(&self, parts: &[&[R]], out: &mut [R])
    where
        R: IntegerRing<Canonical = u64>,
    {
        for (i, part) in parts.iter().enumerate() {
            assert_eq!(
                part.len(),
                self.cols,
                "part {i} length mismatch: expected {}, got {}",
                self.cols,
                part.len()
            );
        }
        assert_eq!(out.len(), self.rows, "output buffer length mismatch");

        for elem in out.iter_mut() {
            *elem = R::zero();
        }

        // Stack buffer: 256 rows / 4 entries-per-byte = 64 bytes per column.
        let packed_len = self.rows.div_ceil(4);
        let mut col_packed = [0u8; 64];

        for (part_idx, part) in parts.iter().enumerate() {
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[0..4].copy_from_slice(&(part_idx as u32).to_le_bytes());
            let mut cipher = ChaCha20::new(&self.seed.into(), &nonce_bytes.into());

            // Sequential stream: 64 bytes per column, columns processed in order
            #[allow(clippy::needless_range_loop)]
            for col in 0..self.cols {
                let coeff = &part[col];

                // Reset buffer before XOR: apply_keystream XORs into existing data
                col_packed.fill(0);
                cipher.apply_keystream(&mut col_packed[..packed_len]);

                // Scatter-accumulate into output
                for row in 0..self.rows {
                    let byte_idx = row / 4;
                    let nibble = (row % 4) as u8;
                    let val = (col_packed[byte_idx] >> (nibble * 2)) & 0x03;
                    match val {
                        0b00 => out[row] += coeff,
                        0b01 => out[row] -= coeff,
                        _ => {}
                    }
                }
            }
        }
    }

    /// Combined projection + extraction in a single pass.
    ///
    /// Returns `(p, jl_rows_raw_flat)` where:
    /// - `p` is the scalar projection `Π · w` (length `rows`)
    /// - `jl_rows_raw_flat` is the raw JL entries in flat layout:
    ///   `[m * parts * d * n + part_idx * d * n + coeff_idx * n + poly_idx]`
    ///
    /// Uses batched keystream (one allocation per part) and unrolled
    /// 4-row-per-byte inner loop to eliminate division/modulo overhead.
    pub fn project_and_extract_jl_rows<R>(
        &self,
        parts: &[&[R]],
        num_parts: usize,
        d: usize,
    ) -> (Vec<R>, Vec<i8>)
    where
        R: IntegerRing<Canonical = u64>,
    {
        assert!(d > 0, "ring degree d must be > 0 (got {})", d);
        assert!(
            self.cols.is_multiple_of(d),
            "cols ({}) must be divisible by ring degree d ({})",
            self.cols,
            d
        );
        assert!(
            parts.len() == num_parts,
            "parts count mismatch: got {}, expected {}",
            parts.len(),
            num_parts
        );
        for (i, part) in parts.iter().enumerate() {
            assert_eq!(
                part.len(),
                self.cols,
                "part {i} length mismatch: expected {}, got {}",
                self.cols,
                part.len()
            );
        }

        let n = self.cols / d;
        let flat_len = self.rows * num_parts * d * n;

        Self::project_and_extract_seq(self, parts, num_parts, d, n, flat_len)
    }

    /// Extract JL rows as a flat `Vec<i8>` without building 4-level Vecs.
    ///
    /// Flat layout: `[m * num_parts * d * n + part_idx * d * n + coeff_idx * n + poly_idx]`
    ///
    /// Uses batched keystream (one allocation per part) and unrolled
    /// 4-row-per-byte inner loop.
    pub fn extract_jl_rows_flat(&self, num_parts: usize, d: usize) -> Vec<i8> {
        assert!(d > 0, "ring degree d must be > 0 (got {})", d);
        assert!(
            self.cols.is_multiple_of(d),
            "cols ({}) must be divisible by ring degree d ({})",
            self.cols,
            d
        );
        let n = self.cols / d;
        let flat_len = self.rows * num_parts * d * n;
        let packed_len = self.rows.div_ceil(4);

        let mut flat = vec![0i8; flat_len];
        for part_idx in 0..num_parts {
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[0..4].copy_from_slice(&(part_idx as u32).to_le_bytes());
            let mut cipher = ChaCha20::new(&self.seed.into(), &nonce_bytes.into());
            let total_bytes = self.cols * packed_len;
            let mut all_keystream = vec![0u8; total_bytes];
            cipher.apply_keystream(&mut all_keystream);

            for col in 0..self.cols {
                let poly_idx = col / d;
                let coeff_idx = col % d;
                let ks_base = col * packed_len;
                for byte_idx in 0..packed_len {
                    let byte = all_keystream[ks_base + byte_idx];
                    let row_base = byte_idx * 4;
                    let part_flat_base = (part_idx * d + coeff_idx) * n + poly_idx;
                    let val0 = byte & 0x03;
                    if row_base < self.rows {
                        let flat_idx = row_base * num_parts * d * n + part_flat_base;
                        flat[flat_idx] = match val0 {
                            0b00 => 1,
                            0b01 => -1,
                            _ => 0,
                        };
                    }
                    let val1 = (byte >> 2) & 0x03;
                    let row1 = row_base + 1;
                    if row1 < self.rows {
                        let flat_idx = row1 * num_parts * d * n + part_flat_base;
                        flat[flat_idx] = match val1 {
                            0b00 => 1,
                            0b01 => -1,
                            _ => 0,
                        };
                    }
                    let val2 = (byte >> 4) & 0x03;
                    let row2 = row_base + 2;
                    if row2 < self.rows {
                        let flat_idx = row2 * num_parts * d * n + part_flat_base;
                        flat[flat_idx] = match val2 {
                            0b00 => 1,
                            0b01 => -1,
                            _ => 0,
                        };
                    }
                    let val3 = (byte >> 6) & 0x03;
                    let row3 = row_base + 3;
                    if row3 < self.rows {
                        let flat_idx = row3 * num_parts * d * n + part_flat_base;
                        flat[flat_idx] = match val3 {
                            0b00 => 1,
                            0b01 => -1,
                            _ => 0,
                        };
                    }
                }
            }
        }
        flat
    }
}

/// Sequential implementation of combined projection + extraction.
impl JLMatrix {
    fn project_and_extract_seq<R>(
        &self,
        parts: &[&[R]],
        num_parts: usize,
        d: usize,
        n: usize,
        flat_len: usize,
    ) -> (Vec<R>, Vec<i8>)
    where
        R: IntegerRing<Canonical = u64>,
    {
        let mut p = vec![R::zero(); self.rows];
        let mut flat = vec![0i8; flat_len];
        let packed_len = self.rows.div_ceil(4);

        for (part_idx, part) in parts.iter().enumerate() {
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[0..4].copy_from_slice(&(part_idx as u32).to_le_bytes());
            let mut cipher = ChaCha20::new(&self.seed.into(), &nonce_bytes.into());

            // Batch keystream: generate all columns at once
            let total_bytes = self.cols * packed_len;
            let mut all_keystream = vec![0u8; total_bytes];
            cipher.apply_keystream(&mut all_keystream);

            for col in 0..self.cols {
                let coeff = &part[col];
                let poly_idx = col / d;
                let coeff_idx = col % d;
                let ks_base = col * packed_len;

                // Unrolled 4-row-per-byte inner loop
                for byte_idx in 0..packed_len {
                    let byte = all_keystream[ks_base + byte_idx];
                    let row_base = byte_idx * 4;
                    let part_flat_base = (part_idx * d + coeff_idx) * n + poly_idx;

                    // Nibble 0: row = row_base
                    let val0 = byte & 0x03;
                    if row_base < self.rows {
                        let flat_idx = row_base * num_parts * d * n + part_flat_base;
                        match val0 {
                            0b00 => {
                                p[row_base] += coeff;
                                flat[flat_idx] = 1;
                            }
                            0b01 => {
                                p[row_base] -= coeff;
                                flat[flat_idx] = -1;
                            }
                            _ => {
                                flat[flat_idx] = 0;
                            }
                        }
                    }

                    // Nibble 1: row = row_base + 1
                    let val1 = (byte >> 2) & 0x03;
                    let row1 = row_base + 1;
                    if row1 < self.rows {
                        let flat_idx = row1 * num_parts * d * n + part_flat_base;
                        match val1 {
                            0b00 => {
                                p[row1] += coeff;
                                flat[flat_idx] = 1;
                            }
                            0b01 => {
                                p[row1] -= coeff;
                                flat[flat_idx] = -1;
                            }
                            _ => {
                                flat[flat_idx] = 0;
                            }
                        }
                    }

                    // Nibble 2: row = row_base + 2
                    let val2 = (byte >> 4) & 0x03;
                    let row2 = row_base + 2;
                    if row2 < self.rows {
                        let flat_idx = row2 * num_parts * d * n + part_flat_base;
                        match val2 {
                            0b00 => {
                                p[row2] += coeff;
                                flat[flat_idx] = 1;
                            }
                            0b01 => {
                                p[row2] -= coeff;
                                flat[flat_idx] = -1;
                            }
                            _ => {
                                flat[flat_idx] = 0;
                            }
                        }
                    }

                    // Nibble 3: row = row_base + 3
                    let val3 = (byte >> 6) & 0x03;
                    let row3 = row_base + 3;
                    if row3 < self.rows {
                        let flat_idx = row3 * num_parts * d * n + part_flat_base;
                        match val3 {
                            0b00 => {
                                p[row3] += coeff;
                                flat[flat_idx] = 1;
                            }
                            0b01 => {
                                p[row3] -= coeff;
                                flat[flat_idx] = -1;
                            }
                            _ => {
                                flat[flat_idx] = 0;
                            }
                        }
                    }
                }
            }
        }

        (p, flat)
    }
}

/// Compute the L2 norm of a vector in `Z_q^k` as a `f64`.
///
/// Each coefficient is mapped to its centered representative in `[-q/2, q/2]`
/// before squaring.
///
/// Uses u128 accumulation (exact before f64 conversion) with overflow guards.
/// For q > 2^32 the running sum may overflow — in that case f64::INFINITY is
/// returned, causing conservative (always-reject) behavior.
pub fn l2_norm<R: IntegerRing<Canonical = u64>>(v: &[R]) -> f64 {
    crate::main_protocol::squared_l2_norm(v)
        .map(grid_std::sqrt)
        .unwrap_or(f64::INFINITY)
}

/// Verify that `||p|| ≤ verify_factor · β`.
///
/// Returns `true` if the projected vector passes the norm threshold.
/// Returns `false` for any invalid profile, non-finite or non-positive beta,
/// or projection length mismatch.
///
/// Uses squared-norm comparison to avoid `sqrt` on the norm side:
/// `||p||² ≤ (verify_factor · β)²`.
pub fn verify_norm<R: IntegerRing<Canonical = u64>>(
    profile: &JLProfile,
    p: &[R],
    beta: f64,
) -> bool {
    if !profile.validate() || !beta.is_finite() || beta <= 0.0 {
        return false;
    }
    if p.len() != profile.rows {
        return false;
    }
    let norm_sq = crate::main_protocol::squared_l2_norm(p).unwrap_or(f64::INFINITY);
    let threshold_sq = (profile.verify_factor * beta) * (profile.verify_factor * beta);
    norm_sq <= threshold_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_std::rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn test_beta() -> f64 {
        1000.0
    }

    #[test]
    fn test_matrix_dimensions() {
        let profile = JLProfile::paper_default();
        let cols = 128;
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        assert_eq!(matrix.rows(), 256);
        assert_eq!(matrix.cols(), 128);
    }

    #[test]
    fn test_column_nonzero_count() {
        type Fr = PrimeField<4_294_967_291u64>;

        let profile = JLProfile::paper_default();
        let cols = 256;
        let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        // Project a 1-hot witness at each column position → the projection result
        // is exactly that column of Π (as ring elements). Count nonzeros.
        for col in 0..cols {
            let mut w = vec![Fr::zero(); cols];
            w[col] = Fr::one();
            let p = matrix.project(&w);

            let nonzero_count = p.iter().filter(|e| !e.is_zero()).count() as u32;
            assert!(
                nonzero_count >= profile.rows as u32 / 3
                    && nonzero_count <= 2 * profile.rows as u32 / 3,
                "Col {col}: {nonzero_count} nonzeros out of {} (expected ~{})",
                profile.rows,
                profile.rows / 2,
            );
        }
    }

    #[test]
    fn test_project_witness() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let cols = 64;

        let cases = [
            (
                "zero_witness",
                [2u8; 32],
                (0..cols).map(|_| Fr::zero()).collect::<Vec<_>>(),
                true,
            ),
            (
                "known_witness",
                [3u8; 32],
                (0..cols)
                    .map(|i| Fr::from_u64((i + 1) as u64))
                    .collect::<Vec<_>>(),
                false,
            ),
        ];

        for (name, seed, w, expect_zero) in &cases {
            let mut rng = ChaCha20Rng::from_seed(*seed);
            let matrix = JLMatrix::sample(&profile, cols, &mut rng);
            let p = matrix.project(w);

            assert_eq!(p.len(), 256, "{}: output length mismatch", name);

            if *expect_zero {
                for elem in &p {
                    assert!(
                        elem.is_zero(),
                        "{}: projection of zero vector should be zero",
                        name
                    );
                }
            } else {
                let norm = l2_norm(&p);
                assert!(
                    norm > 0.0,
                    "{}: projection of non-zero witness should be non-zero",
                    name
                );
            }
        }
    }

    #[test]
    fn test_verify_norm() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let beta = test_beta();
        let q = Fr::modulus();

        let cases = [
            (
                "empty",
                Vec::new(),
                false,
                "empty projection should be rejected",
            ),
            (
                "short",
                vec![Fr::from_u64(1); 10],
                false,
                "short projection should be rejected",
            ),
            (
                "large_vector",
                vec![Fr::from_u64(q / 2); profile.rows],
                false,
                "large vector should fail norm check",
            ),
            (
                "small_vector",
                vec![Fr::from_u64(1); profile.rows],
                true,
                "small vector should pass norm check with large beta",
            ),
        ];

        for (name, p, expect_pass, msg) in &cases {
            let result = verify_norm(&profile, p, beta);
            assert_eq!(result, *expect_pass, "{}: {}", name, msg);
        }
    }

    #[test]
    fn test_deterministic_projection() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let cols = 64;

        let seed = [7u8; 32];
        let mut rng1 = ChaCha20Rng::from_seed(seed);
        let mut rng2 = ChaCha20Rng::from_seed(seed);
        let matrix1 = JLMatrix::sample(&profile, cols, &mut rng1);
        let matrix2 = JLMatrix::sample(&profile, cols, &mut rng2);

        let w: Vec<Fr> = (0..cols)
            .map(|i| Fr::from_u64((i * 7 + 3) as u64))
            .collect();

        let p1 = matrix1.project(&w);
        let p2 = matrix2.project(&w);

        assert_eq!(p1, p2, "same seed should produce same projection");
    }

    #[test]
    fn test_l2_norm_centered_representatives() {
        type Fr = PrimeField<997u64>;

        let one = Fr::from_u64(1);
        let minus_one = Fr::from_u64(996); // q - 1

        let v_pos = vec![one];
        let v_neg = vec![minus_one];

        let norm_pos = l2_norm(&v_pos);
        let norm_neg = l2_norm(&v_neg);

        assert!(
            (norm_pos - norm_neg).abs() < 1e-10,
            "||1|| should equal ||-1||, got {} vs {}",
            norm_pos,
            norm_neg
        );
        assert!(
            (norm_pos - 1.0).abs() < 1e-10,
            "||1|| should be 1.0, got {norm_pos}"
        );
    }

    #[test]
    fn test_project_into_buffer() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let cols = 32;
        let mut rng = ChaCha20Rng::from_seed([10u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        let w: Vec<Fr> = (0..cols).map(|i| Fr::from_u64((i + 1) as u64)).collect();

        let p_alloc = matrix.project(&w);
        let mut p_buffer = vec![Fr::zero(); 256];
        matrix.project_into(&w, &mut p_buffer);

        assert_eq!(p_alloc, p_buffer, "project_into should match project");
    }

    #[test]
    fn test_multi_project_linearity() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let cols = 16;
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        let w1: Vec<Fr> = (0..cols).map(|i| Fr::from_u64((i + 1) as u64)).collect();
        let w2: Vec<Fr> = (0..cols)
            .map(|i| Fr::from_u64((i * 3 + 2) as u64))
            .collect();
        let zeros = vec![Fr::zero(); cols];

        let p_both = matrix.project_multi(&[&w1, &w2]);
        let p_w1 = matrix.project_multi(&[&w1, &zeros]);
        let p_w2 = matrix.project_multi(&[&zeros, &w2]);
        let p_sum: Vec<Fr> = p_w1.iter().zip(p_w2.iter()).map(|(a, b)| *a + *b).collect();

        assert_eq!(p_both, p_sum, "project_multi should be linear in each part");
    }

    #[test]
    fn test_seed_differs_per_part_position() {
        type Fr = PrimeField<4_294_967_291u64>;
        let profile = JLProfile::paper_default();
        let cols = 16;
        let mut rng = ChaCha20Rng::from_seed([12u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        let w: Vec<Fr> = (0..cols).map(|i| Fr::from_u64((i + 1) as u64)).collect();

        let p_pos0 = matrix.project_multi(&[&w]);
        let zeros = vec![Fr::zero(); cols];
        let p_pos1 = matrix.project_multi(&[&zeros, &w]);

        assert_ne!(
            p_pos0, p_pos1,
            "same witness at different part positions should produce different projections"
        );
    }

    #[test]
    fn test_jl_projection_round_trip() {
        type Fr = PrimeField<4_294_967_291u64>;

        let profile = JLProfile::paper_default();
        // n=4 polys per part, d=4 ring degree, r=3 parts → cols=16
        let n = 4;
        let d = 4;
        let cols = n * d;
        let num_parts = 3;
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        // Build random witness parts (each part: n polys of d coefficients)
        let witness_parts: Vec<Vec<Fr>> = (0..num_parts)
            .map(|_| {
                (0..cols)
                    .map(|i| Fr::from_u64(((i * 137 + 42) as u64) % 100))
                    .collect()
            })
            .collect();

        // Compute JL projection (scalar output per row)
        let p = matrix.project_multi(
            &witness_parts
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>(),
        );

        // Extract JL rows at polynomial granularity for scalar round-trip check.
        // jl_rows_raw[m][i][k][j]: row m, part i, coeff k, poly index j.
        let jl_rows_raw = matrix.extract_jl_rows_poly(num_parts, d);
        let q = Fr::modulus();

        // Verify: p[m] from project_multi equals manual Σ_i Π[m,i,col]·w_i[col]
        // using jl_rows_raw entries where jl_rows_raw[m][i][k][j], k=col%d, j=col/d.
        for m in 0..profile.rows {
            let mut manual_pm: i128 = 0;
            for (i, part_coeffs) in witness_parts.iter().enumerate() {
                for (col, coeff) in part_coeffs.iter().enumerate() {
                    let k = col % d;
                    let j = col / d;
                    let entry = jl_rows_raw[m][i][k][j] as i128;
                    let coeff_val = coeff.to_u64() as i128;
                    manual_pm += entry * coeff_val;
                }
            }
            let manual_pm_mod = ((manual_pm % q as i128) + q as i128) % q as i128;
            assert_eq!(
                p[m].to_u64() as i128,
                manual_pm_mod,
                "Row {}: project_multi gave {}, manual Σ gave {}",
                m,
                p[m].to_u64(),
                manual_pm_mod
            );
        }
    }

    /// Verify the polynomial-granularity path: build Rq polynomials from extract_jl_rows_poly,
    /// apply conjugation, multiply by witness polynomials, check constant term == project_multi.
    #[test]
    fn test_jl_poly_conjugation_round_trip() {
        type Fr = PrimeField<4_294_967_291u64>;
        type Rq = CyclotomicPolyRing<Fr, 4>;

        use grid_algebra::poly::ring::{CyclotomicPolyRing, PolyRing};

        let profile = JLProfile::paper_default();
        let d = 4; // ring degree (matches Rq)
        let num_parts = 2;
        let n = 2; // witness rank (polys per part)
        let cols = n * d; // 8 scalars per part

        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        // Build witness as polynomials: each part has n polynomials of d coefficients
        let witness_polys: Vec<Vec<Rq>> = (0..num_parts)
            .map(|pi| {
                (0..n)
                    .map(|j| {
                        let coeffs: Vec<Fr> = (0..d)
                            .map(|k| {
                                let scalar_idx = j * d + k;
                                Fr::from_u64(((pi * 997 + scalar_idx * 311 + 42) as u64) % 100)
                            })
                            .collect();
                        Rq::try_from_coeffs(&coeffs).expect("d matches ring degree")
                    })
                    .collect()
            })
            .collect();

        // Also build flattened scalar witness for project_multi comparison
        let witness_scalar: Vec<Vec<Fr>> = witness_polys
            .iter()
            .map(|part_polys| {
                let mut scalars = Vec::with_capacity(cols);
                for poly in part_polys {
                    for c in poly.coeffs() {
                        scalars.push(*c);
                    }
                }
                scalars
            })
            .collect();

        // Compute scalar projection via project_multi
        let p = matrix.project_multi(
            &witness_scalar
                .iter()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>(),
        );

        // Extract JL rows at polynomial granularity
        let jl_rows_raw = matrix.extract_jl_rows_poly(num_parts, d);
        let q = Fr::modulus();

        // For each JL row m, build Rq polynomials from extract_jl_rows_poly,
        // apply conjugation, compute inner product with witness polys, check ct == p[m]
        for m in 0..profile.rows {
            // Compute Σ_i Σ_j ⟨σ₋₁(π_i[m][j]), s_i[j]⟩ then take constant term
            let mut sum_poly = Rq::zero();

            for i in 0..num_parts {
                for j in 0..n {
                    // Build polynomial π_i[m][j] from jl_rows_raw[m][i][k][j] over k=0..d
                    let coeffs: Vec<Fr> = (0..d)
                        .map(|k| {
                            let v = jl_rows_raw[m][i][k][j];
                            if v < 0 {
                                Fr::from_u64(q.wrapping_sub((-v) as u64))
                            } else {
                                Fr::from_u64(v as u64)
                            }
                        })
                        .collect();
                    let pi_poly = Rq::try_from_coeffs(&coeffs).expect("d matches ring degree");

                    // Apply conjugation (σ₋₁)
                    let pi_conj = crate::relation::conjugation(&pi_poly);

                    // Multiply conjugated projection poly by witness poly, accumulate
                    sum_poly += pi_conj * &witness_polys[i][j];
                }
            }

            // Constant term of the accumulated polynomial should match p[m]
            let ct = sum_poly.coeff(0);
            assert_eq!(
                ct.to_u64(),
                p[m].to_u64(),
                "Row {}: poly+conjugation path gave ct={}, project_multi gave {}",
                m,
                ct.to_u64(),
                p[m].to_u64()
            );
        }
    }

    #[test]
    fn test_extract_jl_rows_poly_shape_and_consistency() {
        let profile = JLProfile::paper_default();
        // n=4 witness rank (polys per part), d=4 ring degree → 16 scalars per part
        let n = 4;
        let d = 4;
        let cols = n * d;
        let num_parts = 3;
        let mut rng = ChaCha20Rng::from_seed([99u8; 32]);
        let matrix = JLMatrix::sample(&profile, cols, &mut rng);

        // Extract at polynomial granularity
        let poly_rows = matrix.extract_jl_rows_poly(num_parts, d);

        // Check shape: [rows][parts][d][n]
        assert_eq!(poly_rows.len(), profile.rows);
        assert_eq!(poly_rows[0].len(), num_parts);
        assert_eq!(poly_rows[0][0].len(), d);
        assert_eq!(poly_rows[0][0][0].len(), n);

        // All entries are in {-1, 0, 1}
        for rows_m in &poly_rows {
            for parts_i in rows_m {
                for coeffs_k in parts_i {
                    for val in coeffs_k {
                        assert!(
                            *val == -1 || *val == 0 || *val == 1,
                            "entry {} not in {{-1,0,1}}",
                            val
                        );
                    }
                }
            }
        }

        // Cross-check: manually derive one entry from ChaCha20 keystream
        // For part 0, col 0 (poly_idx=0, coeff_idx=0), row 0:
        let nonce_bytes: [u8; 12] = [0u8; 12];
        let mut cipher = ChaCha20::new(&matrix.seed.into(), &nonce_bytes.into());
        let mut col_packed = [0u8; 64];
        col_packed.fill(0);
        cipher.apply_keystream(&mut col_packed[..profile.rows.div_ceil(4)]);
        let val = col_packed[0] & 0x03;
        let expected = match val {
            0b00 => 1i8,
            0b01 => -1i8,
            _ => 0i8,
        };
        assert_eq!(
            poly_rows[0][0][0][0], expected,
            "manual ChaCha20 check failed"
        );
    }
}
