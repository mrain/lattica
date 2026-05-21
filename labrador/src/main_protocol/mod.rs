//! Shared types and helpers for the LaBRADOR main protocol (§5.2).
//!
//! Provides balanced decomposition, reconstruction, garbage indexing,
//! and transcript convenience helpers used across all sub-modules.

pub mod aggregation;
pub mod amortization;
pub mod garbage;
pub mod step_prover;
pub mod step_verifier;
pub mod verify_equations;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use grid_algebra::arith::ring::IntegerRing;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

use crate::crs::CRS;
use crate::params::LabradorParams;
use crate::relation::{LabradorStatement, QuadraticFunction};

/// Absorb public inputs into the transcript before any protocol message.
///
/// The Fiat-Shamir transformation requires all public input (statement,
/// system parameters, CRS) to be bound to the transcript before any
/// challenge is derived. This function absorbs:
///
/// 1. Protocol version + domain separator (`b"labrador_public"`)
/// 2. CRS seed (32 bytes, labeled `b"labrador_master_crs"`)
/// 3. Full statement (F and F' functions, all fields)
/// 4. Complete params struct via CanonicalSerialize
///
/// Both prover and verifier must call this at the very start of the protocol.
pub fn absorb_public_input<R, const N: usize, T>(
    transcript: &mut T,
    crs: &CRS,
    statement: &LabradorStatement<CyclotomicPolyRing<R, N>>,
    params: &LabradorParams,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + CanonicalSerialize + CanonicalDeserialize,
    CyclotomicPolyRing<R, N>: CanonicalSerialize,
    T: grid_transcript::Transcript,
{
    // Protocol version + domain separator
    transcript
        .append_bytes(b"labrador_public", b"v1")
        .map_err(|e| alloc::format!("failed to absorb domain separator: {:?}", e))?;

    // CRS seed (distinct from per-level challenge_bytes labels)
    transcript
        .append_bytes(b"labrador_master_crs", &crs.seed)
        .map_err(|e| alloc::format!("failed to absorb CRS seed: {:?}", e))?;

    // Statement: absorb count of functions, then each function's data
    let num_f = statement.f.len();
    let num_f_prime = statement.f_prime.len();
    transcript
        .append_bytes(b"labrador_stmt_num_f", &(num_f as u64).to_le_bytes())
        .map_err(|e| alloc::format!("failed to absorb num_f: {:?}", e))?;
    transcript
        .append_bytes(
            b"labrador_stmt_num_f_prime",
            &(num_f_prime as u64).to_le_bytes(),
        )
        .map_err(|e| alloc::format!("failed to absorb num_f_prime: {:?}", e))?;

    // Absorb each F function
    for (idx, func) in statement.f.iter().enumerate() {
        absorb_quadratic_func(transcript, func, idx)?;
    }

    // Absorb each F' function
    for (idx, func) in statement.f_prime.iter().enumerate() {
        absorb_quadratic_func(transcript, func, idx)?;
    }

    // Complete params struct (all fields via CanonicalSerialize)
    transcript
        .append_serializable(b"labrador_params", params)
        .map_err(|e| alloc::format!("failed to absorb params: {:?}", e))?;

    Ok(())
}

/// Absorb a single quadratic function into the transcript.
fn absorb_quadratic_func<R, const N: usize, T>(
    transcript: &mut T,
    func: &QuadraticFunction<CyclotomicPolyRing<R, N>>,
    idx: usize,
) -> Result<(), String>
where
    R: IntegerRing<Canonical = u64> + CanonicalSerialize + CanonicalDeserialize,
    CyclotomicPolyRing<R, N>: CanonicalSerialize,
    T: grid_transcript::Transcript,
{
    match func {
        QuadraticFunction::Dense(d) => {
            // Absorb count of quadratic terms
            transcript
                .append_bytes(b"labrador_func_dense", &(d.a.len() as u64).to_le_bytes())
                .map_err(|e| alloc::format!("failed to absorb dense func: {:?}", e))?;

            // Quadratic coefficients a[i]
            for (qi, coeff) in d.a.iter().enumerate() {
                transcript
                    .append_serializable(b"labrador_func_a", coeff)
                    .map_err(|e| {
                        alloc::format!("failed to absorb func[{}] a[{}]: {:?}", idx, qi, e)
                    })?;
            }

            // Index pairs ij[k] = (i, j) — full u32 to avoid truncation for r > 255
            for (pi, &(i, j)) in d.ij.iter().enumerate() {
                let bytes = [(i as u32).to_le_bytes(), (j as u32).to_le_bytes()];
                transcript
                    .append_bytes(b"labrador_func_ij", &bytes.concat())
                    .map_err(|e| {
                        alloc::format!("failed to absorb func[{}] ij[{}]: {:?}", idx, pi, e)
                    })?;
            }

            // Linear terms (phi[part][entry])
            for phi_i in d.phi.iter() {
                for coeff in phi_i.iter() {
                    transcript
                        .append_serializable(b"labrador_func_phi", coeff)
                        .map_err(|e| alloc::format!("failed to absorb phi coeff: {:?}", e))?;
                }
            }

            // Constant term b
            transcript
                .append_serializable(b"labrador_func_b", &d.b)
                .map_err(|e| alloc::format!("failed to absorb func[{}] b: {:?}", idx, e))?;
        }
        QuadraticFunction::Sparse(s) => {
            // Absorb count of quadratic terms
            transcript
                .append_bytes(
                    b"labrador_func_sparse",
                    &(s.ij_a.len() as u64).to_le_bytes(),
                )
                .map_err(|e| alloc::format!("failed to absorb sparse func: {:?}", e))?;

            // Quadratic terms (i, j, coeff) — full u32 to avoid truncation
            for &(i, j, ref coeff) in &s.ij_a {
                let bytes = [(i as u32).to_le_bytes(), (j as u32).to_le_bytes()];
                transcript
                    .append_bytes(b"labrador_func_ij", &bytes.concat())
                    .map_err(|e| alloc::format!("failed to absorb func[{}] ij: {:?}", idx, e))?;
                transcript
                    .append_serializable(b"labrador_func_a", coeff)
                    .map_err(|e| alloc::format!("failed to absorb func[{}] a: {:?}", idx, e))?;
            }

            // Sparse linear terms (part, entry, coeff) — full u32 to avoid truncation
            for &(part, entry, ref coeff) in &s.phi {
                let bytes = [(part as u32).to_le_bytes(), (entry as u32).to_le_bytes()];
                transcript
                    .append_bytes(b"labrador_func_phi_ij", &bytes.concat())
                    .map_err(|e| {
                        alloc::format!("failed to absorb func[{}] phi_ij: {:?}", idx, e)
                    })?;
                transcript
                    .append_serializable(b"labrador_func_phi_a", coeff)
                    .map_err(|e| alloc::format!("failed to absorb func[{}] phi_a: {:?}", idx, e))?;
            }

            // Constant term b
            transcript
                .append_serializable(b"labrador_func_b", &s.b)
                .map_err(|e| alloc::format!("failed to absorb func[{}] b: {:?}", idx, e))?;
        }
    }

    Ok(())
}

/// Compute the squared L2 norm of a vector of ring elements with overflow guard.
///
/// Maps each coefficient to its centered representative in `[-q/2, q/2)`, squares,
/// and accumulates in u128 (exact before f64 conversion). Returns `Ok(norm_sq)`
/// on success, or `Err` if the running sum overflows u128 — in which case the
/// norm definitely exceeds any reasonable threshold.
///
/// For paper-size q ≈ 2^32 the u128 accumulation is exact; the final f64
/// conversion may lose low-order bits for very large norms. For larger q,
/// overflow triggers conservative rejection.
pub fn squared_l2_norm<'a, R: IntegerRing<Canonical = u64>>(
    elems: impl AsRef<[R]> + 'a,
) -> Result<f64, &'static str> {
    let q = R::modulus();
    let q128 = q as u128;
    let half_q = q / 2;

    let mut sum_sq: u128 = 0;
    for elem in elems.as_ref() {
        let val = elem.to_u64() as u128;
        let abs_centered = if val > half_q as u128 {
            q128 - val
        } else {
            val
        };
        let sq = abs_centered
            .checked_mul(abs_centered)
            .ok_or("norm overflow: square exceeds u128")?;
        sum_sq = sum_sq
            .checked_add(sq)
            .ok_or("norm overflow: sum exceeds u128")?;
    }
    Ok(sum_sq as f64)
}

/// Compute the centered L2 squared norm of a vector, returning exact u128 result.
///
/// Same algorithm as [`squared_l2_norm`] but preserves the exact u128 value
/// instead of converting to f64. Use this for security-critical threshold
/// comparisons to avoid floating-point rounding at the ULP boundary.
pub fn squared_l2_norm_u128<R: IntegerRing<Canonical = u64>>(
    elems: impl AsRef<[R]>,
) -> Result<u128, &'static str> {
    let q = R::modulus();
    let q128 = q as u128;
    let half_q = q / 2;

    let mut sum_sq: u128 = 0;
    for elem in elems.as_ref() {
        let val = elem.to_u64() as u128;
        let abs_centered = if val > half_q as u128 {
            q128 - val
        } else {
            val
        };
        let sq = abs_centered
            .checked_mul(abs_centered)
            .ok_or("norm overflow: square exceeds u128")?;
        sum_sq = sum_sq
            .checked_add(sq)
            .ok_or("norm overflow: sum exceeds u128")?;
    }
    Ok(sum_sq)
}

/// Balanced (centered) radix decomposition of a single coefficient.
///
/// Given coefficient `c ∈ Z_q` and base `b`, produces `num_limbs` limbs
/// such that `c ≡ Σ limb[k] · b^k (mod q)` and each `limb[k] ∈ [-b/2, b/2]`.
///
/// **Signed carry algorithm:** center the coefficient in [-q/2, q/2], then
/// iteratively extract `limb = centered_mod(carry, base)` as i64 in [-b/2, b/2],
/// encode into Z_q as unsigned `(limb % b + b) % b`, then update
/// `carry = (carry - limb) / base`. No modular inverse involved.
fn decompose_coeff_balanced(c_u64: u64, base: u64, num_limbs: usize, q: u64) -> Vec<i64> {
    let half_q = q as i128 / 2;
    let mut carry = c_u64 as i128;
    if carry > half_q {
        carry -= q as i128;
    }

    let base_i128 = base as i128;
    let mut limbs = Vec::with_capacity(num_limbs);

    for _ in 0..num_limbs {
        let limb = centered_mod(carry, base_i128);
        limbs.push(limb as i64);
        carry = (carry - limb) / base_i128;
    }

    limbs
}

/// Centered modulo: returns `v mod base` in range `[-base/2, base/2]` as i128.
#[inline]
fn centered_mod(v: i128, base: i128) -> i128 {
    let half = base / 2;
    let r = v % base;
    if r > half {
        r - base
    } else if r < -half {
        r + base
    } else {
        r
    }
}

/// Balanced decomposition of a polynomial.
///
/// Returns `num_limbs` limb polynomials, each with centered coefficients in [-b/2, b/2].
/// Each limb coefficient is stored as an unsigned ring element in [0, q):
/// negative centered value `v ∈ [-b/2, -1]` is encoded as `q - |v|` (≡ v mod q).
pub fn decompose_poly_balanced<Rq>(poly: &Rq, base: u64, num_limbs: usize) -> Vec<Rq>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    let q = Rq::Coeff::modulus();
    let n = Rq::degree();

    let mut limbs: Vec<Rq> = Vec::with_capacity(num_limbs);
    for _ in 0..num_limbs {
        limbs.push(Rq::zero());
    }

    for i in 0..n {
        let c = poly.coeff(i).to_u64();
        let signed_limbs = decompose_coeff_balanced(c, base, num_limbs, q);
        for (k, &limb_signed) in signed_limbs.iter().enumerate() {
            let limb_u64 = if limb_signed >= 0 {
                limb_signed as u64
            } else {
                q.wrapping_sub((-limb_signed) as u64)
            };
            limbs[k].set_coeff(i, Rq::Coeff::from_u64(limb_u64));
        }
    }

    limbs
}

/// Thin alias: reconstruct a single polynomial from limbs + base.
/// Generic over `Rq: PolyRing` so callers with abstract Rq can use it without conversion.
///
/// Each limb coefficient is stored unsigned in [0, q). Decode to centered
/// i64, then accumulate `Σ limb_signed[k] · base^k` as i128, finally
/// reduce mod q and encode unsigned.
pub fn reconstruct_from_limbs<Rq>(limbs: &[Rq], base: u64) -> Rq
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    let q = Rq::Coeff::modulus();
    let n = Rq::degree();
    let q_i128 = q as i128;
    let base_i128 = base as i128;
    let coeffs: Vec<Rq::Coeff> = (0..n)
        .map(|i| {
            let mut acc = 0i128;
            let mut base_pow = 1i128;
            for limb_poly in limbs.iter() {
                let limb_u64 = limb_poly.coeff(i).to_u64();
                let half_base = base_i128 / 2;
                let limb_signed = if limb_u64 as i128 > half_base {
                    (limb_u64 as i128) - q_i128
                } else {
                    limb_u64 as i128
                };
                acc = (acc + limb_signed * base_pow).rem_euclid(q_i128);
                base_pow = (base_pow * base_i128).rem_euclid(q_i128);
            }
            Rq::Coeff::from_u64(acc as u64)
        })
        .collect();
    Rq::try_from_coeffs(&coeffs).expect("reconstruct_from_limbs: length matches degree")
}

/// Decomposed representation of a batch of polynomials.
///
/// `polys[j]` is decomposed into `num_limbs` limb polynomials.
/// The flat vector stores: `[limb_0_of_poly_0, limb_1_of_poly_0, ..., limb_0_of_poly_1, ...]`
///
/// Index: element `(poly_idx, limb_idx)` → `flat[poly_idx * num_limbs + limb_idx]`.
///
/// Each limb polynomial has centered coefficients in `[-base/2, base/2]`.
#[derive(Debug, Clone)]
pub struct DecomposedPolys<Rq> {
    pub flat: Vec<Rq>,
    pub num_polys: usize,
    pub num_limbs: usize,
    pub base: u64,
}

impl<Rq> CanonicalSerialize for DecomposedPolys<Rq>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    fn serialized_size(&self) -> usize {
        // Use saturating arithmetic so malformed in-memory values never panic.
        // serialize_into() performs strict validation and returns InvalidData.
        let range = self.base.saturating_add(1);
        let bits = if range <= 1 {
            1
        } else {
            (64usize.saturating_sub((range - 1).leading_zeros() as usize)).min(57)
        };
        let n = <Rq as PolyRing>::degree();
        let total_coeffs = self.flat.len().saturating_mul(n);
        let total_bits = total_coeffs.saturating_mul(bits);
        16usize.saturating_add(total_bits.div_ceil(8))
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        // Guard: base must be positive (matches deserialization check)
        if self.base == 0 {
            return Err(SerializationError::InvalidData(
                "base must be positive".into(),
            ));
        }

        let num_polys_u32 = u32::try_from(self.num_polys)
            .map_err(|_| SerializationError::InvalidData("num_polys exceeds u32".into()))?;
        let num_limbs_u32 = u32::try_from(self.num_limbs)
            .map_err(|_| SerializationError::InvalidData("num_limbs exceeds u32".into()))?;
        let expected_flat = self.num_polys.checked_mul(self.num_limbs).ok_or_else(|| {
            SerializationError::InvalidData("num_polys * num_limbs overflow".into())
        })?;
        if self.flat.len() != expected_flat {
            return Err(SerializationError::InvalidData(format!(
                "flat.len({}) != num_polys({}) * num_limbs({})",
                self.flat.len(),
                self.num_polys,
                self.num_limbs
            )));
        }
        buf.extend_from_slice(&num_polys_u32.to_le_bytes());
        buf.extend_from_slice(&num_limbs_u32.to_le_bytes());
        buf.extend_from_slice(&self.base.to_le_bytes());

        let q = <Rq::Coeff as IntegerRing>::modulus();
        let half_base = self.base / 2;

        // Guard: base/2 must be < q (otherwise threshold = q - base/2 wraps)
        if half_base >= q {
            return Err(SerializationError::InvalidData(format!(
                "base/2 ({}) >= q ({})",
                half_base, q
            )));
        }

        // Balanced decomposition limbs: [-base/2, base/2] → compact [0, base+1)
        // Offset = base/2, threshold = q - base/2
        let offset = half_base;
        let threshold = q.wrapping_sub(offset);

        // Guard: base+1 must not overflow
        let range = self
            .base
            .checked_add(1)
            .ok_or_else(|| SerializationError::InvalidData("base+1 overflow".into()))?;
        let range_u64 = range;
        let range_u128 = range as u128;
        let bits = if range_u128 <= 1 {
            1
        } else {
            (128 - (range_u128 - 1).leading_zeros()) as usize
        };
        // Guard: bits must fit in u64 accumulator (7 remaining bits + bits <= 64)
        if bits > 57 {
            return Err(SerializationError::InvalidData(format!(
                "bits ({}) exceeds u64 packer capacity (57)",
                bits
            )));
        }
        let n = <Rq as PolyRing>::degree();
        let total_coeffs = self
            .flat
            .len()
            .checked_mul(n)
            .ok_or_else(|| SerializationError::InvalidData("total_coeffs overflow".into()))?;
        let total_bytes = (total_coeffs
            .checked_mul(bits)
            .ok_or_else(|| SerializationError::InvalidData("total bits overflow".into()))?)
        .div_ceil(8);

        // Collect all compact values first, rejecting out-of-range coefficients
        let mut compact = Vec::with_capacity(total_coeffs);
        for limb_poly in &self.flat {
            for coeff in limb_poly.coeffs() {
                let val = coeff.to_u64();
                let c = if val < threshold {
                    val.wrapping_add(offset)
                } else {
                    val.wrapping_sub(threshold)
                };
                if c >= range_u64 {
                    return Err(SerializationError::InvalidData(format!(
                        "coefficient compact value {} out of range [0, {})",
                        c, range_u64
                    )));
                }
                compact.push(c);
            }
        }

        // Bit-pack coefficients into bytes
        let mut packed = Vec::with_capacity(total_bytes);
        let mut bitbuf: u64 = 0;
        let mut bitbuf_bits: u32 = 0; // bits accumulated in bitbuf
        for &c in &compact {
            bitbuf = (bitbuf << bits) | c;
            bitbuf_bits += bits as u32;
            while bitbuf_bits >= 8 {
                let shift = bitbuf_bits - 8;
                packed.push(((bitbuf >> shift) & 0xFF) as u8);
                bitbuf_bits -= 8;
            }
        }
        if bitbuf_bits > 0 {
            packed.push(((bitbuf << (8 - bitbuf_bits)) & 0xFF) as u8);
        }

        buf.extend_from_slice(&packed);
        Ok(())
    }
}

impl<Rq> CanonicalDeserialize for DecomposedPolys<Rq>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 16 {
            return Err(SerializationError::UnexpectedEnd);
        }

        let num_polys = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let num_limbs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let base = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);

        // Validate base before any arithmetic that depends on it
        if base == 0 {
            return Err(SerializationError::InvalidData(
                "base must be positive".into(),
            ));
        }
        let range = base
            .checked_add(1)
            .ok_or_else(|| SerializationError::InvalidData("base + 1 overflows u64".into()))?;
        let bits = if range <= 1 {
            1
        } else {
            (64 - (range - 1).leading_zeros()) as usize
        };
        if bits > 57 {
            return Err(SerializationError::InvalidData(format!(
                "base {} too large for u64 packer (bits={})",
                base, bits
            )));
        }

        let q = <Rq::Coeff as IntegerRing>::modulus();
        let half_base = base / 2;
        if half_base >= q {
            return Err(SerializationError::InvalidData(format!(
                "base/2 ({}) >= modulus ({})",
                half_base, q
            )));
        }

        let n = <Rq as PolyRing>::degree();
        let total_polys = num_polys
            .checked_mul(num_limbs)
            .ok_or_else(|| SerializationError::InvalidData("total_polys overflow".into()))?;
        let total_coeffs = total_polys
            .checked_mul(n)
            .ok_or_else(|| SerializationError::InvalidData("total_coeffs overflow".into()))?;

        let total_bits = total_coeffs
            .checked_mul(bits)
            .ok_or_else(|| SerializationError::InvalidData("total bits overflow".into()))?;
        let total_bytes = total_bits.div_ceil(8);

        if data.len() < 16 + total_bytes {
            return Err(SerializationError::UnexpectedEnd);
        }

        let offset = half_base;
        let payload = &data[16..16 + total_bytes];

        // Bit-unpack coefficients with canonical checks
        let mut compact = Vec::with_capacity(total_coeffs);
        let mut buf: u64 = 0;
        let mut buf_bits: u32 = 0;
        let mut pidx = 0;
        for _ in 0..total_coeffs {
            while buf_bits < bits as u32 {
                buf = (buf << 8) | (payload[pidx] as u64);
                buf_bits += 8;
                pidx += 1;
            }
            let shift = buf_bits - bits as u32;
            let val = (buf >> shift) & ((1u64 << bits) - 1);
            // Reject out-of-alphabet values (non-canonical encoding)
            if val >= range {
                return Err(SerializationError::InvalidData(format!(
                    "coefficient value {} out of range [0, {})",
                    val, range
                )));
            }
            buf &= (1u64 << shift) - 1;
            buf_bits = shift;
            compact.push(val);
        }
        // Reject nonzero unused padding bits after the last coefficient
        if buf != 0 {
            return Err(SerializationError::InvalidData(
                "nonzero padding bits in packed data".into(),
            ));
        }

        // Reconstruct polynomials using inverse branch (no wraparound)
        let mut flat = Vec::with_capacity(total_polys);
        let mut cidx = 0;
        for _ in 0..total_polys {
            let mut coeffs = Vec::with_capacity(n);
            for _ in 0..n {
                let c = compact[cidx];
                cidx += 1;
                let val = if c >= offset {
                    c - offset
                } else {
                    q - (offset - c)
                };
                coeffs.push(<Rq::Coeff as IntegerRing>::from_u64(val));
            }
            flat.push(<Rq as PolyRing>::try_from_coeffs(&coeffs).map_err(|e| {
                SerializationError::InvalidData(format!("invalid poly coeffs: {:?}", e))
            })?);
        }

        Ok((
            DecomposedPolys {
                flat,
                num_polys,
                num_limbs,
                base,
            },
            16 + total_bytes,
        ))
    }
}

impl<Rq> DecomposedPolys<Rq>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    /// Reconstruct original polynomials from balanced limbs.
    /// Returns `num_polys` polynomials of the same Rq type.
    pub fn reconstruct(&self) -> Vec<Rq> {
        (0..self.num_polys)
            .map(|j| {
                let start = j * self.num_limbs;
                let limbs: Vec<Rq> = (0..self.num_limbs)
                    .map(|k| self.flat[start + k].clone())
                    .collect();
                reconstruct_from_limbs(&limbs, self.base)
            })
            .collect()
    }
}

/// Upper-triangular index: (i,j) with i ≤ j maps to a unique index in [0, r(r+1)/2).
/// Row i has entries (i,i), (i,i+1), ..., (i, r-1).
#[inline]
pub fn garbage_index(r: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < r);
    // Row i starts at index Σ_{k=0}^{i-1}(r-k) = r·i - i·(i-1)/2.
    // Off-diagonal offset within row: (j - i).
    // Use wrapping_sub/mul to avoid debug-mode panic when i=0 (0-1 would overflow).
    i.wrapping_mul(r) - i.wrapping_sub(1).wrapping_mul(i) / 2 + (j - i)
}

/// Inverse of `garbage_index`: given row count `r` and flat index `idx`, recover `(i, j)`.
///
/// Solves the quadratic `row_start(i) = r*i - i*(i-1)/2 <= idx` in O(1) via
/// the closed-form `i = floor((2r+1 - sqrt((2r+1)^2 - 8*idx)) / 2)`.
/// Uses ceiling of isqrt because `i` is inversely related to `sqrt(D)`:
/// flooring would produce an `i` that's too large.
#[inline]
pub fn garbage_index_inv(r: usize, idx: usize) -> (usize, usize) {
    debug_assert!(idx < garbage_count(r));
    let d = (2 * r + 1) as u128;
    let disc = d * d - 8 * (idx as u128);
    let s = disc.isqrt();
    let s_ceil = if s * s < disc { s + 1 } else { s };
    let i = ((d - s_ceil) / 2) as usize;
    let start_i = i.wrapping_mul(r) - i.wrapping_sub(1).wrapping_mul(i) / 2;
    let offset = idx - start_i;
    let j = i + offset;
    (i, j)
}

/// Number of upper-triangular entries for r vectors: r(r+1)/2.
#[inline]
pub fn garbage_count(r: usize) -> usize {
    r * (r + 1) / 2
}

/// Reconstruct t_vecs from decomposed limbs.
///
/// The verifier uses this to recover each t_i from the opening,
/// rather than trusting t_vecs directly.
pub fn reconstruct_t_vecs<Rq>(
    decomposed: &DecomposedPolys<Rq>,
    r: usize,
    kappa: usize,
    t1: usize,
) -> Vec<RingVec<Rq>>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    let mut t_vecs = Vec::with_capacity(r);
    for wi in 0..r {
        let mut t_i = Vec::with_capacity(kappa);
        for ki in 0..kappa {
            let poly_idx = wi * kappa + ki;
            let limb_start = poly_idx * t1;
            let limbs: Vec<_> = (0..t1)
                .map(|k| decomposed.flat[limb_start + k].clone())
                .collect();
            let reconstructed = reconstruct_from_limbs(&limbs, decomposed.base);
            t_i.push(reconstructed);
        }
        t_vecs.push(RingVec::new(t_i));
    }
    t_vecs
}

/// Balanced-decompose a batch of polynomials.
/// Used by garbage.rs and inner_commit.rs.
pub fn decompose_polys<Rq>(polys: &[Rq], base: u64, num_limbs: usize) -> DecomposedPolys<Rq>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
{
    let mut flat = Vec::with_capacity(polys.len() * num_limbs);
    for poly in polys {
        let limbs = decompose_poly_balanced(poly, base, num_limbs);
        flat.extend(limbs);
    }
    DecomposedPolys {
        flat,
        num_polys: polys.len(),
        num_limbs,
        base,
    }
}

/// Squeeze a polynomial (N ring coefficients) from the transcript.
///
/// Collects coefficients with `?` — does NOT panic on transcript failure.
/// Generic over the ring type `Rq` so the caller controls the concrete poly type.
pub fn challenge_poly<Rq, T>(transcript: &mut T, label: &'static [u8]) -> Result<Rq, T::Error>
where
    Rq: PolyRing,
    Rq::Coeff: IntegerRing<Canonical = u64>,
    T: Transcript,
{
    let n = Rq::degree();
    let mut coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        coeffs.push(transcript.challenge_scalar(label)?);
    }
    Ok(Rq::try_from_coeffs(&coeffs)
        .expect("challenge_poly builds exactly Rq::degree() coefficients"))
}

/// Flat JL rows: single contiguous allocation replacing `Vec<Vec<Vec<Rq>>>`.
///
/// Layout: `flat[(m * num_parts + part_idx) * num_polys + poly_idx]`
/// where `m` = JL row (0..num_rows), `part_idx` = witness part (0..num_parts),
/// `poly_idx` = polynomial index (0..num_polys).
#[derive(Debug, Clone)]
pub struct JlRowsFlat<Rq> {
    data: Vec<Rq>,
    num_parts: usize,
    num_polys: usize,
}

impl<Rq: Clone> JlRowsFlat<Rq> {
    /// Create from a flat vector. Panics if len doesn't match shape.
    pub fn new(data: Vec<Rq>, num_rows: usize, num_parts: usize, num_polys: usize) -> Self {
        assert_eq!(
            data.len(),
            num_rows * num_parts * num_polys,
            "JlRowsFlat data length mismatch"
        );
        Self {
            data,
            num_parts,
            num_polys,
        }
    }

    /// Number of JL rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.data.len() / (self.num_parts * self.num_polys)
    }

    /// Number of witness parts.
    #[inline]
    pub fn num_parts(&self) -> usize {
        self.num_parts
    }

    /// Number of polynomials per part.
    #[inline]
    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    /// Access element `[m][part_idx][poly_idx]`.
    #[inline]
    pub fn get(&self, m: usize, part_idx: usize, poly_idx: usize) -> &Rq {
        &self.data[(m * self.num_parts + part_idx) * self.num_polys + poly_idx]
    }

    /// Access mutable element `[m][part_idx][poly_idx]`.
    #[inline]
    pub fn get_mut(&mut self, m: usize, part_idx: usize, poly_idx: usize) -> &mut Rq {
        &mut self.data[(m * self.num_parts + part_idx) * self.num_polys + poly_idx]
    }

    /// Slice for JL row `m` (all parts and polynomials).
    #[inline]
    pub fn row(&self, m: usize) -> &[Rq] {
        let start = m * self.num_parts * self.num_polys;
        &self.data[start..start + self.num_parts * self.num_polys]
    }
}

/// Flat version: converts raw JL rows to conjugated polynomials in a single allocation.
#[allow(clippy::needless_range_loop)]
pub fn jl_rows_to_conjugated_polys_flat<R, const N: usize>(
    jl_rows_raw: &[Vec<Vec<Vec<i8>>>],
    q: u64,
) -> JlRowsFlat<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let num_rows = jl_rows_raw.len();
    let num_parts = jl_rows_raw[0].len();
    assert_eq!(
        jl_rows_raw[0][0].len(),
        N,
        "JL per_part must have exactly N rows"
    );
    let num_polys = jl_rows_raw[0][0][0].len();

    let total = num_rows * num_parts * num_polys;
    let mut data = Vec::with_capacity(total);

    for parts in jl_rows_raw {
        for per_part in parts {
            for j in 0..num_polys {
                let coeffs: [R; N] = core::array::from_fn(|k| {
                    let src = if k == 0 { 0 } else { N - k };
                    let v = per_part[src][j];
                    let u = if v < 0 {
                        q.wrapping_sub((-v) as u64)
                    } else {
                        v as u64
                    };
                    let val = if k == 0 || u == 0 {
                        u
                    } else {
                        q.wrapping_sub(u)
                    };
                    R::from_u64(val)
                });
                data.push(CyclotomicPolyRing::from_array(coeffs));
            }
        }
    }

    JlRowsFlat::new(data, num_rows, num_parts, num_polys)
}

/// Converts flat raw JL entries (from [`crate::jl::JLMatrix::project_and_extract_jl_rows`])
/// to conjugated polynomials in a single `JlRowsFlat` allocation.
///
/// Input flat layout: `[m * num_parts * N * n + part_idx * N * n + coeff_idx * n + poly_idx]`
/// Output layout: `JlRowsFlat` with `[m][part][poly]` → `CyclotomicPolyRing<R, N>`.
pub fn jl_rows_flat_to_conjugated_polys<R, const N: usize>(
    flat: &[i8],
    num_rows: usize,
    num_parts: usize,
    n: usize,
    q: u64,
) -> JlRowsFlat<CyclotomicPolyRing<R, N>>
where
    R: IntegerRing<Canonical = u64> + NegacyclicMulRing<N>,
{
    let total = num_rows * num_parts * n;
    let mut data = Vec::with_capacity(total);

    for m in 0..num_rows {
        for part_idx in 0..num_parts {
            for poly_idx in 0..n {
                let coeffs: [R; N] = core::array::from_fn(|k| {
                    let src = if k == 0 { 0 } else { N - k };
                    let flat_idx = m * num_parts * N * n + part_idx * N * n + src * n + poly_idx;
                    let v = flat[flat_idx];
                    let u = if v < 0 {
                        q.wrapping_sub((-v) as u64)
                    } else {
                        v as u64
                    };
                    let val = if k == 0 || u == 0 {
                        u
                    } else {
                        q.wrapping_sub(u)
                    };
                    R::from_u64(val)
                });
                data.push(CyclotomicPolyRing::from_array(coeffs));
            }
        }
    }

    JlRowsFlat::new(data, num_rows, num_parts, n)
}

/// Minimal transcript trait for challenge_poly and other helpers.
///
/// Re-exported from `grid-transcript` when available; this is a local
/// alias for the function signatures used in this module.
pub trait Transcript {
    type Error;
    fn append_bytes(&mut self, label: &'static [u8], data: &[u8]) -> Result<(), Self::Error>;
    fn challenge_scalar<R>(&mut self, label: &'static [u8]) -> Result<R, Self::Error>
    where
        R: IntegerRing<Canonical = u64>;
}

// Blanket impl: any `grid_transcript::Transcript` also satisfies our local trait.
impl<T: grid_transcript::Transcript> Transcript for T {
    type Error = grid_transcript::TranscriptError;

    fn append_bytes(&mut self, label: &'static [u8], data: &[u8]) -> Result<(), Self::Error> {
        <Self as grid_transcript::Transcript>::append_bytes(self, label, data)
    }

    fn challenge_scalar<R>(&mut self, label: &'static [u8]) -> Result<R, Self::Error>
    where
        R: IntegerRing<Canonical = u64>,
    {
        <Self as grid_transcript::Transcript>::challenge_scalar(self, label)
    }
}

/// Modular addition: (a + b) mod m.
#[cfg(test)]
pub(super) fn add_mod(a: u64, b: u64, m: u64) -> u64 {
    a.wrapping_add(b) % m
}

/// Modular multiplication: (a * b) mod m.
#[cfg(test)]
pub(super) fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (a as u128 * b as u128 % m as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::Ring;
    use grid_algebra::poly::ring::CyclotomicPolyRing;

    type F12289 = PrimeField<12289>;
    type Poly64 = CyclotomicPolyRing<F12289, 64>;

    fn from_u64(v: u64) -> Poly64 {
        Poly64::try_from_coeffs(&vec![F12289::from_u64(v); 64]).expect("from_u64")
    }

    #[test]
    fn test_garbage_index() {
        // Table-driven point checks: (r, i, j) -> expected index
        let cases: &[(usize, usize, usize, usize)] = &[
            // r=1
            (1, 0, 0, 0),
            // r=3
            (3, 0, 0, 0),
            (3, 0, 1, 1),
            (3, 0, 2, 2),
            (3, 1, 1, 3),
            (3, 1, 2, 4),
            (3, 2, 2, 5),
            // r=4
            (4, 0, 0, 0),
            (4, 0, 3, 3),
            (4, 3, 3, 9),
            // r=5 edge: i=0 (wrapping_sub safety)
            (5, 0, 0, 0),
            (5, 0, 4, 4),
        ];
        for &(r, i, j, expected) in cases {
            assert_eq!(
                garbage_index(r, i, j),
                expected,
                "garbage_index({},{},{}) expected {}",
                r,
                i,
                j,
                expected
            );
        }

        // Exhaustive: all indices for r=5 must be in range
        let count = garbage_count(5);
        assert_eq!(count, 15);
        for i in 0..5 {
            for j in i..5 {
                let idx = garbage_index(5, i, j);
                assert!(
                    idx < count,
                    "index ({},{})={} should be < {}",
                    i,
                    j,
                    idx,
                    count
                );
            }
        }
    }

    #[test]
    fn test_garbage_index_inv_roundtrip() {
        // Exhaustive roundtrip for r=1..10: garbage_index_inv(garbage_index(i,j)) == (i,j)
        for r in 1..=10 {
            for i in 0..r {
                for j in i..r {
                    let idx = garbage_index(r, i, j);
                    let (ri, rj) = garbage_index_inv(r, idx);
                    assert_eq!(
                        (ri, rj),
                        (i, j),
                        "roundtrip failed for r={}, i={}, j={} -> idx={} -> ({},{})",
                        r,
                        i,
                        j,
                        idx,
                        ri,
                        rj
                    );
                }
            }
        }
        // Also test sequential inverse for larger r
        for r in [50, 100, 200] {
            let count = garbage_count(r);
            for idx in 0..count {
                let (ri, rj) = garbage_index_inv(r, idx);
                assert!(
                    ri <= rj && rj < r,
                    "out of range: r={}, idx={} -> ({},{})",
                    r,
                    idx,
                    ri,
                    rj
                );
                assert_eq!(
                    garbage_index(r, ri, rj),
                    idx,
                    "inverse mismatch: r={}, idx={}",
                    r,
                    idx
                );
            }
        }
    }

    #[test]
    fn test_garbage_count() {
        assert_eq!(garbage_count(1), 1);
        assert_eq!(garbage_count(2), 3);
        assert_eq!(garbage_count(3), 6);
        assert_eq!(garbage_count(4), 10);
        assert_eq!(garbage_count(10), 55);
    }

    #[test]
    fn test_centered_mod() {
        assert_eq!(centered_mod(0, 10), 0);
        assert_eq!(centered_mod(3, 10), 3);
        assert_eq!(centered_mod(5, 10), 5);
        assert_eq!(centered_mod(6, 10), -4);
        assert_eq!(centered_mod(7, 10), -3);
        assert_eq!(centered_mod(9, 10), -1);
        assert_eq!(centered_mod(-1, 10), -1);
        assert_eq!(centered_mod(-4, 10), -4);
        assert_eq!(centered_mod(-5, 10), -5);
        assert_eq!(centered_mod(-6, 10), 4);
        assert_eq!(centered_mod(-7, 10), 3);

        // Positive and negative with same base
        assert_eq!(centered_mod(12, 10), 2);
        assert_eq!(centered_mod(-12, 10), -2);
    }

    #[test]
    fn test_decompose_coeff_balanced() {
        // (value, base, num_limbs, q) -> expected limbs
        let cases: &[(u64, u64, usize, u64, Vec<i64>)] = &[
            // base=8, q=12289
            (5, 8, 4, 12289, vec![-3, 1, 0, 0]),
            (8, 8, 4, 12289, vec![0, 1, 0, 0]),
            (3, 8, 3, 12289, vec![3, 0, 0]),
            (12, 8, 3, 12289, vec![4, 1, 0]),
            (20, 8, 3, 12289, vec![4, 2, 0]),
            // Large value with base=256
            (12000, 256, 3, 12289, vec![-33, -1, 0]),
        ];

        for &(value, base, num_limbs, q, ref expected) in cases {
            let limbs = decompose_coeff_balanced(value, base, num_limbs, q);
            assert_eq!(
                limbs, *expected,
                "decompose({},{},{}) mismatch",
                value, base, q
            );

            // Verify all limbs in centered range
            let half_base = base as i64 / 2;
            for &l in &limbs {
                assert!(
                    l.abs() <= half_base,
                    "limb {} out of [-{}, {}] for value={}",
                    l,
                    half_base,
                    half_base,
                    value
                );
            }

            // Verify reconstruction
            let mut val = 0i128;
            let mut pow = 1i128;
            for &l in &limbs {
                val += l as i128 * pow;
                pow *= base as i128;
            }
            let val_mod = ((val % q as i128) + q as i128) % q as i128;
            assert_eq!(
                val_mod as u64, value,
                "reconstruction failed for value {}",
                value
            );
        }
    }

    #[test]
    fn test_decompose_reconstruct_roundtrip() {
        let q = 12289u64;
        let base = 8u64;
        let num_limbs = 8usize;

        // Test all small values
        for v in 0..256u64 {
            let limbs = decompose_coeff_balanced(v, base, num_limbs, q);
            let mut val = 0i128;
            let mut pow = 1i128;
            for &l in &limbs {
                val += l as i128 * pow;
                pow *= base as i128;
            }
            let val_mod = ((val % q as i128) + q as i128) % q as i128;
            assert_eq!(val_mod as u64, v, "roundtrip failed for value {}", v);
        }

        // Test near-q values
        for v in &[q - 10, q - 1, q / 2, q / 2 + 1, 0, 1, q - 1] {
            let limbs = decompose_coeff_balanced(*v, base, num_limbs, q);
            let mut val = 0i128;
            let mut pow = 1i128;
            for &l in &limbs {
                val += l as i128 * pow;
                pow *= base as i128;
            }
            let val_mod = ((val % q as i128) + q as i128) % q as i128;
            assert_eq!(val_mod as u64, *v, "roundtrip failed for value {}", v);
        }
    }

    #[test]
    fn test_poly_decompose_reconstruct_roundtrip() {
        let base = 8u64;

        // Zero polynomial
        {
            let zero = Poly64::zero();
            let limbs = decompose_poly_balanced(&zero, base, 4);
            assert_eq!(limbs.len(), 4);
            for limb in &limbs {
                assert!(limb.is_zero(), "limb of zero poly should be zero");
            }
            let reconstructed = reconstruct_from_limbs(&limbs, base);
            assert!(reconstructed.is_zero());
        }

        // Polynomial with various coefficient values
        let num_limbs = 8usize;
        let mut poly = Poly64::zero();
        for i in 0..64 {
            let c = (i * 193 + 7) % 12289;
            poly.set_coeff(i, F12289::from_u64(c as u64));
        }

        let limbs = decompose_poly_balanced(&poly, base, num_limbs);
        assert_eq!(limbs.len(), num_limbs);

        // Each limb coefficient should be in centered range [-base/2, base/2]
        for limb_poly in &limbs {
            for i in 0..64 {
                let v = limb_poly.coeff(i).to_u64();
                let q = F12289::modulus();
                // Decode to centered value
                let centered = if v as i128 > (base as i128 / 2) {
                    (v as i128) - q as i128
                } else {
                    v as i128
                };
                assert!(
                    centered.abs() <= base as i128 / 2,
                    "limb coefficient out of centered range: {} at poly {:?}, coeff {}",
                    v,
                    limb_poly,
                    i
                );
            }
        }

        // Reconstruct and compare
        let reconstructed = reconstruct_from_limbs(&limbs, base);
        for i in 0..64 {
            assert_eq!(
                reconstructed.coeff(i),
                poly.coeff(i),
                "coefficient {} mismatch after roundtrip",
                i
            );
        }
    }

    #[test]
    fn test_decomposed_polys_reconstruct() {
        let base = 8u64;
        let num_limbs = 4usize;

        // Build 3 polynomials
        let polys: Vec<Poly64> = (0..3)
            .map(|j| {
                let mut p = Poly64::zero();
                for i in 0..64 {
                    let c = (j * 331 + i * 17) % 12289;
                    p.set_coeff(i, F12289::from_u64(c as u64));
                }
                p
            })
            .collect();

        let decomposed = decompose_polys(&polys, base, num_limbs);
        assert_eq!(decomposed.num_polys, 3);
        assert_eq!(decomposed.num_limbs, num_limbs);
        assert_eq!(decomposed.flat.len(), 3 * num_limbs);
        assert_eq!(decomposed.base, base);

        // Reconstruct all polynomials
        let reconstructed = decomposed.reconstruct();
        assert_eq!(reconstructed.len(), 3);
        for (j, (orig, recon)) in polys.iter().zip(reconstructed.iter()).enumerate() {
            for i in 0..64 {
                assert_eq!(
                    recon.coeff(i),
                    orig.coeff(i),
                    "poly {} coeff {} mismatch",
                    j,
                    i
                );
            }
        }
    }

    #[test]
    fn test_decomposed_polys_ser_roundtrip() {
        use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

        for &base in &[8u64, 16u64, 256u64] {
            let num_limbs = 4usize;
            let polys: Vec<Poly64> = (0..3)
                .map(|j| {
                    let mut p = Poly64::zero();
                    for i in 0..64 {
                        let c = (j * 331 + i * 17) % 12289;
                        p.set_coeff(i, F12289::from_u64(c as u64));
                    }
                    p
                })
                .collect();

            let decomposed = decompose_polys(&polys, base, num_limbs);
            let serialized = decomposed.serialize().expect("serialize");
            let (decomposed2, consumed) =
                <DecomposedPolys<Poly64> as CanonicalDeserialize>::deserialize(&serialized)
                    .expect("deserialize");
            assert_eq!(consumed, serialized.len());
            assert_eq!(decomposed.flat.len(), decomposed2.flat.len());
            for (i, (a, b)) in decomposed
                .flat
                .iter()
                .zip(decomposed2.flat.iter())
                .enumerate()
            {
                for j in 0..64 {
                    assert_eq!(
                        a.coeff(j),
                        b.coeff(j),
                        "base={} flat[{}] coeff {} mismatch",
                        base,
                        i,
                        j
                    );
                }
            }
            let reconstructed = decomposed2.reconstruct();
            for (j, (orig, recon)) in polys.iter().zip(reconstructed.iter()).enumerate() {
                for i in 0..64 {
                    assert_eq!(
                        recon.coeff(i),
                        orig.coeff(i),
                        "base={} poly {} coeff {} mismatch after roundtrip",
                        base,
                        j,
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_reconstruct_polys_alias() {
        let base = 8u64;
        let num_limbs = 4usize;
        let poly = from_u64(42);
        let polys = vec![poly.clone(), poly.clone()];

        let decomposed = decompose_polys(&polys, base, num_limbs);
        let reconstructed = decomposed.reconstruct();
        assert_eq!(reconstructed.len(), 2);
        for recon in &reconstructed {
            assert_eq!(recon.coeff(0).to_u64(), 42);
        }
    }

    #[test]
    fn test_mul_mod() {
        assert_eq!(mul_mod(3, 4, 12289), 12);
        assert_eq!(mul_mod(1000, 1000, 12289), (1000u128 * 1000 % 12289) as u64);
        // Near-overflow: large values
        assert_eq!(
            mul_mod(12288, 12288, 12289),
            (12288u128 * 12288 % 12289) as u64
        );
    }

    #[test]
    fn test_add_mod() {
        assert_eq!(add_mod(3, 4, 12289), 7);
        assert_eq!(add_mod(12288, 5, 12289), 4);
        assert_eq!(add_mod(0, 0, 12289), 0);
    }

    #[test]
    fn test_decompose_coeff_balanced_bounds() {
        let q = 12289u64;
        let base = 256u64;
        let half_base = base as i128 / 2;

        // Verify all limbs are in [-base/2, base/2] for random-ish values
        for v in (0..q).step_by(37) {
            let limbs = decompose_coeff_balanced(v, base, 8, q);
            for &l in &limbs {
                assert!(
                    l >= -half_base as i64 && l <= half_base as i64,
                    "limb {} out of bounds for value {}, base={}",
                    l,
                    v,
                    base
                );
            }
        }
    }
}
