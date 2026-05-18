//! Number Theoretic Transform helpers for polynomial rings.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};

use dashmap::DashMap;
use once_cell::sync::Lazy;

use crate::arith::ntt::{self, NTTRing, NttError, NttPlan, cached_ntt_plan};
use crate::arith::ring::Ring;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    type_id: TypeId,
    len: usize,
}

type ErasedTwistedPlan = Arc<dyn Any + Send + Sync>;

static TWISTED_PLAN_CACHE: Lazy<DashMap<PlanCacheKey, ErasedTwistedPlan>> =
    Lazy::new(|| DashMap::with_capacity(ntt::plan_cache_capacity()));

/// Reusable stage data for a twisted negacyclic NTT of a fixed size over a fixed coefficient ring.
pub struct TwistedNttPlan<R: Ring> {
    len: usize,
    twists: Box<[R]>,
    untwists: Box<[R]>,
    ntt_plan: Arc<NttPlan<R>>,
}

impl<R: Ring> TwistedNttPlan<R> {
    /// Returns the transform length this plan was built for.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when this plan was built for an empty transform.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn validate_len(&self, len: usize) -> Result<(), NttError> {
        if self.len != len {
            return Err(NttError::LengthMismatch {
                left: self.len,
                right: len,
            });
        }
        Ok(())
    }

    /// Transform coefficient-domain values into the twisted evaluation domain in place.
    pub fn forward_in_place(&self, coeffs: &mut [R]) -> Result<(), NttError>
    where
        R: NTTRing,
    {
        self.validate_len(coeffs.len())?;
        self.forward_in_place_trusted(coeffs);
        Ok(())
    }

    /// Transform twisted-domain values back into the coefficient domain in place.
    pub fn inverse_in_place(&self, evals: &mut [R]) -> Result<(), NttError>
    where
        R: NTTRing,
    {
        self.validate_len(evals.len())?;
        self.inverse_in_place_trusted(evals);
        Ok(())
    }

    #[inline(always)]
    pub fn forward_in_place_trusted(&self, coeffs: &mut [R])
    where
        R: NTTRing,
    {
        debug_assert_eq!(
            self.len,
            coeffs.len(),
            "trusted twisted NTT forward length mismatch"
        );
        R::pointwise_mul_assign_slice(coeffs, self.twists.as_ref());
        R::ntt_forward_with_plan(coeffs, self.ntt_plan.as_ref())
            .expect("trusted twisted NTT forward length mismatch");
    }

    #[inline(always)]
    pub(crate) fn inverse_in_place_trusted(&self, evals: &mut [R])
    where
        R: NTTRing,
    {
        debug_assert_eq!(
            self.len,
            evals.len(),
            "trusted twisted NTT inverse length mismatch"
        );
        // `ntt_inverse_with_plan` only fails on plan/length mismatch, which the trusted callers
        // establish once up front.
        R::ntt_inverse_with_plan(evals, self.ntt_plan.as_ref())
            .expect("trusted twisted NTT inverse length mismatch");
        R::pointwise_mul_assign_slice(evals, self.untwists.as_ref());
    }

    /// Multiply two coefficient-domain polynomials modulo `X^n + 1` in place.
    ///
    /// On success, `lhs` is overwritten with the negacyclic product while `rhs` is used as
    /// temporary workspace.
    pub fn multiply_in_place(&self, lhs: &mut [R], rhs: &mut [R]) -> Result<(), NttError>
    where
        R: NTTRing,
    {
        if lhs.len() != rhs.len() {
            return Err(NttError::LengthMismatch {
                left: lhs.len(),
                right: rhs.len(),
            });
        }
        self.validate_len(lhs.len())?;
        self.forward_in_place(lhs)?;
        self.forward_in_place(rhs)?;

        R::pointwise_mul_assign_slice(lhs, rhs);
        self.inverse_in_place(lhs)
    }
}

fn powers<R: Ring>(base: &R, n: usize) -> Vec<R> {
    let mut out = Vec::with_capacity(n);
    let mut current = R::one();
    for _ in 0..n {
        out.push(current.clone());
        current *= base.clone();
    }
    out
}

impl<R: Ring + NTTRing + Send + Sync + 'static> TwistedNttPlan<R> {
    fn build(len: usize) -> Result<Self, NttError> {
        let ntt_plan = cached_ntt_plan::<R>(len)?;
        let twice_len = len
            .checked_mul(2)
            .ok_or(NttError::UnsupportedTwist { len })?;
        let psi = R::root_of_unity(twice_len).ok_or(NttError::UnsupportedTwist { len })?;
        let inv_psi = R::inv_root_of_unity(twice_len).ok_or(NttError::UnsupportedTwist { len })?;
        Ok(Self {
            len,
            twists: powers(&psi, len).into_boxed_slice(),
            untwists: powers(&inv_psi, len).into_boxed_slice(),
            ntt_plan,
        })
    }
}

fn typed_cached_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    plan: &Arc<dyn Any + Send + Sync>,
) -> Arc<TwistedNttPlan<R>> {
    Arc::downcast::<TwistedNttPlan<R>>(Arc::clone(plan))
        .unwrap_or_else(|_| unreachable!("plan cache key must match plan type"))
}

pub fn cached_twisted_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    len: usize,
) -> Result<Arc<TwistedNttPlan<R>>, NttError> {
    let key = PlanCacheKey {
        type_id: TypeId::of::<R>(),
        len,
    };
    if let Some(plan) = TWISTED_PLAN_CACHE.get(&key) {
        return Ok(typed_cached_plan::<R>(&plan));
    }

    let plan = Arc::new(TwistedNttPlan::<R>::build(len)?);
    let erased: ErasedTwistedPlan = plan.clone();
    TWISTED_PLAN_CACHE.entry(key).insert_entry(erased);
    Ok(plan)
}

/// Returns a cached plain NTT plan for the requested ring and transform size.
pub fn ntt_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    len: usize,
) -> Result<Arc<NttPlan<R>>, NttError> {
    cached_ntt_plan::<R>(len)
}

/// Returns a cached twisted negacyclic NTT plan for the requested ring and transform size.
pub fn twisted_ntt_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    len: usize,
) -> Result<Arc<TwistedNttPlan<R>>, NttError> {
    cached_twisted_plan::<R>(len)
}

/// Performs the forward NTT in-place.
pub fn ntt_forward<R: Ring + NTTRing>(coeffs: &mut [R]) -> Result<(), NttError> {
    R::ntt_forward_in_place(coeffs)
}

/// Performs the forward NTT in-place using a precomputed plan.
pub fn ntt_forward_with_plan<R: Ring + NTTRing>(
    coeffs: &mut [R],
    plan: &NttPlan<R>,
) -> Result<(), NttError> {
    R::ntt_forward_with_plan(coeffs, plan)
}

/// Performs the inverse NTT in-place.
pub fn ntt_inverse<R: Ring + NTTRing>(evals: &mut [R]) -> Result<(), NttError> {
    R::ntt_inverse_in_place(evals)
}

/// Performs the inverse NTT in-place using a precomputed plan.
pub fn ntt_inverse_with_plan<R: Ring + NTTRing>(
    evals: &mut [R],
    plan: &NttPlan<R>,
) -> Result<(), NttError> {
    R::ntt_inverse_with_plan(evals, plan)
}

/// Transform coefficient-domain values into the twisted evaluation domain in place.
pub fn twisted_ntt_forward_in_place<R: Ring + NTTRing + Send + Sync + 'static>(
    coeffs: &mut [R],
) -> Result<(), NttError> {
    let plan = cached_twisted_plan::<R>(coeffs.len())?;
    plan.forward_in_place(coeffs)
}

/// Transform coefficient-domain values into the twisted evaluation domain in place using a
/// precomputed plan.
pub fn twisted_ntt_forward_in_place_with_plan<R: Ring + NTTRing>(
    coeffs: &mut [R],
    plan: &TwistedNttPlan<R>,
) -> Result<(), NttError> {
    plan.forward_in_place(coeffs)
}

/// Transform coefficient-domain values into the twisted evaluation domain.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn twisted_ntt_forward<R: Ring + NTTRing + Send + Sync + 'static>(
    coeffs: &[R],
) -> Result<Vec<R>, NttError> {
    let plan = cached_twisted_plan::<R>(coeffs.len())?;
    let mut evals = coeffs.to_vec();
    plan.forward_in_place(&mut evals)?;
    Ok(evals)
}

/// Transform twisted-domain evaluations back into the coefficient domain in place.
pub fn twisted_ntt_inverse_in_place<R: Ring + NTTRing + Send + Sync + 'static>(
    evals: &mut [R],
) -> Result<(), NttError> {
    let plan = cached_twisted_plan::<R>(evals.len())?;
    plan.inverse_in_place(evals)
}

/// Transform twisted-domain evaluations back into the coefficient domain in place using a
/// precomputed plan.
pub fn twisted_ntt_inverse_in_place_with_plan<R: Ring + NTTRing>(
    evals: &mut [R],
    plan: &TwistedNttPlan<R>,
) -> Result<(), NttError> {
    plan.inverse_in_place(evals)
}

/// Transform twisted-domain evaluations back into coefficient-domain values.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn twisted_ntt_inverse<R: Ring + NTTRing + Send + Sync + 'static>(
    evals: &[R],
) -> Result<Vec<R>, NttError> {
    let plan = cached_twisted_plan::<R>(evals.len())?;
    let mut coeffs = evals.to_vec();
    plan.inverse_in_place(&mut coeffs)?;
    Ok(coeffs)
}

/// Multiply two coefficient-domain polynomials modulo `X^n + 1` in place using a precomputed
/// twisted NTT plan.
pub fn poly_mul_ntt_in_place_with_plan<R: Ring + NTTRing>(
    lhs: &mut [R],
    rhs: &mut [R],
    plan: &TwistedNttPlan<R>,
) -> Result<(), NttError> {
    plan.multiply_in_place(lhs, rhs)
}

/// Multiply two coefficient-domain polynomials modulo `X^n + 1` in place using a twisted NTT.
///
/// On success, `lhs` is overwritten with the negacyclic product while `rhs` is used as temporary
/// workspace.
pub fn poly_mul_ntt_in_place<R: Ring + NTTRing + Send + Sync + 'static>(
    lhs: &mut [R],
    rhs: &mut [R],
) -> Result<(), NttError> {
    if lhs.len() != rhs.len() {
        return Err(NttError::LengthMismatch {
            left: lhs.len(),
            right: rhs.len(),
        });
    }
    let plan = cached_twisted_plan::<R>(lhs.len())?;
    poly_mul_ntt_in_place_with_plan(lhs, rhs, plan.as_ref())
}

/// Multiply two coefficient-domain polynomials modulo `X^n + 1` using a twisted NTT.
pub fn poly_mul_ntt<R: Ring + NTTRing + Send + Sync + 'static>(
    a: &[R],
    b: &[R],
) -> Result<Vec<R>, NttError> {
    if a.len() != b.len() {
        return Err(NttError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    let mut lhs = a.to_vec();
    let mut rhs = b.to_vec();
    poly_mul_ntt_in_place(&mut lhs, &mut rhs)?;
    Ok(lhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::large_prime::{Bls12_381Fq, Bls12_381Fr, Bn254Fq, Bn254Fr};
    use crate::arith::large_rns::Rns3V0;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::{IntegerRing, Ring};
    use crate::poly::ring::{CyclotomicPolyRing, PolyRing};
    use grid_std::UniformRand;

    type F17 = PrimeField<17>;
    type F12289 = PrimeField<12289>;
    type F184683593729 = PrimeField<184683593729>;
    type FGoldilocks = PrimeField<{ crate::arith::prime::GOLDILOCKS_MODULUS }>;

    fn round_trip<R: Ring + NTTRing + grid_std::UniformRand>(n: usize) {
        let mut rng = grid_std::test_rng();
        let original: Vec<R> = (0..n).map(|_| R::rand(&mut rng)).collect();
        let mut transformed = original.clone();
        ntt_forward(&mut transformed).unwrap();
        ntt_inverse(&mut transformed).unwrap();
        assert_eq!(transformed, original);
    }

    fn naive_negacyclic<R: Ring>(a: &[R], b: &[R]) -> Vec<R> {
        let n = a.len();
        let mut out = vec![R::zero(); n];
        for i in 0..n {
            for j in 0..n {
                let prod = a[i].clone() * &b[j];
                if i + j < n {
                    out[i + j] += prod;
                } else {
                    out[i + j - n] -= prod;
                }
            }
        }
        out
    }

    #[test]
    fn test_ntt_round_trip_f17_n8() {
        round_trip::<F17>(8);
    }

    #[test]
    fn test_ntt_round_trip_f12289_n256() {
        round_trip::<F12289>(256);
    }

    #[test]
    fn test_ntt_round_trip_f184683593729_n256() {
        round_trip::<F184683593729>(256);
    }

    #[test]
    fn test_ntt_round_trip_goldilocks_n512() {
        round_trip::<FGoldilocks>(512);
    }

    #[test]
    fn test_ntt_round_trip_bn254_fr_n256() {
        round_trip::<Bn254Fr>(256);
    }

    #[test]
    fn test_ntt_round_trip_bls12_381_fr_n256() {
        round_trip::<Bls12_381Fr>(256);
    }

    #[test]
    fn test_ntt_round_trip_rns3_v0_n2() {
        let values = vec![Rns3V0::from_u64(1), Rns3V0::from_u64(7)];
        let mut transformed = values.clone();
        ntt_forward(&mut transformed).unwrap();
        ntt_inverse(&mut transformed).unwrap();
        assert_eq!(transformed, values);
    }

    #[test]
    fn test_ntt_round_trip_rns3_v0_n256() {
        round_trip::<Rns3V0>(256);
    }

    #[test]
    fn test_ntt_with_cached_plan_round_trip_bn254_fr() {
        let mut rng = grid_std::test_rng();
        let original: Vec<Bn254Fr> = (0..256).map(|_| Bn254Fr::rand(&mut rng)).collect();
        let plan = ntt_plan::<Bn254Fr>(original.len()).unwrap();
        assert_eq!(plan.len(), original.len());
        assert!(!plan.is_empty());

        let mut transformed = original.clone();
        ntt_forward_with_plan(&mut transformed, plan.as_ref()).unwrap();
        ntt_inverse_with_plan(&mut transformed, plan.as_ref()).unwrap();
        assert_eq!(transformed, original);
    }

    #[test]
    fn test_large_prime_ntt_profile_limits() {
        assert_eq!(Bn254Fr::max_ntt_size(), 1usize << 28);
        assert!(Bn254Fr::supports_ntt(256));
        assert!(!Bn254Fr::supports_ntt(1usize << 29));

        assert_eq!(Bls12_381Fr::max_ntt_size(), 1usize << 32);
        assert!(Bls12_381Fr::supports_ntt(256));
        assert!(!Bls12_381Fr::supports_ntt(3));

        assert_eq!(Bn254Fq::max_ntt_size(), 2);
        assert!(Bn254Fq::supports_ntt(2));
        assert!(!Bn254Fq::supports_ntt(4));

        assert_eq!(Bls12_381Fq::max_ntt_size(), 2);
        assert!(Bls12_381Fq::supports_ntt(2));
        assert!(!Bls12_381Fq::supports_ntt(4));

        assert_eq!(Rns3V0::max_ntt_size(), 1usize << 20);
        assert!(Rns3V0::supports_ntt(512));
        assert!(Rns3V0::supports_ntt(1usize << 16));
        assert!(!Rns3V0::supports_ntt(1usize << 21));
    }

    #[test]
    fn test_twisted_ntt_round_trip_small() {
        let values = vec![
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
            F17::from_u64(8),
        ];
        let evals = twisted_ntt_forward(&values).unwrap();
        let coeffs = twisted_ntt_inverse(&evals).unwrap();
        assert_eq!(coeffs, values);
    }

    #[test]
    fn test_twisted_ntt_in_place_with_cached_plan_round_trip_small() {
        let mut values = vec![
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(5),
            F17::from_u64(6),
            F17::from_u64(7),
            F17::from_u64(8),
        ];
        let original = values.clone();
        let plan = twisted_ntt_plan::<F17>(values.len()).unwrap();
        assert_eq!(plan.len(), values.len());
        assert!(!plan.is_empty());

        twisted_ntt_forward_in_place_with_plan(&mut values, plan.as_ref()).unwrap();
        twisted_ntt_inverse_in_place_with_plan(&mut values, plan.as_ref()).unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn test_poly_mul_ntt_matches_naive_small() {
        let a = vec![
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(2),
        ];
        let b = vec![
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(9),
            F17::from_u64(2),
            F17::from_u64(6),
        ];
        assert_eq!(poly_mul_ntt(&a, &b).unwrap(), naive_negacyclic(&a, &b));
    }

    #[test]
    fn test_poly_mul_ntt_in_place_matches_naive_small() {
        let mut a = vec![
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(2),
        ];
        let mut b = vec![
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(9),
            F17::from_u64(2),
            F17::from_u64(6),
        ];
        let expected = naive_negacyclic(&a, &b);
        poly_mul_ntt_in_place(&mut a, &mut b).unwrap();
        assert_eq!(a, expected);
    }

    #[test]
    fn test_poly_mul_ntt_in_place_with_plan_matches_naive_small() {
        let mut a = vec![
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(2),
        ];
        let mut b = vec![
            F17::from_u64(3),
            F17::from_u64(1),
            F17::from_u64(4),
            F17::from_u64(1),
            F17::from_u64(5),
            F17::from_u64(9),
            F17::from_u64(2),
            F17::from_u64(6),
        ];
        let expected = naive_negacyclic(&a, &b);
        let plan = twisted_ntt_plan::<F17>(a.len()).unwrap();
        poly_mul_ntt_in_place_with_plan(&mut a, &mut b, plan.as_ref()).unwrap();
        assert_eq!(a, expected);
    }

    #[test]
    fn test_poly_mul_ntt_matches_naive_kyber_sizes() {
        let mut rng = grid_std::test_rng();
        for &n in &[16usize, 64, 256] {
            let a: Vec<F12289> = (0..n).map(|_| F12289::rand(&mut rng)).collect();
            let b: Vec<F12289> = (0..n).map(|_| F12289::rand(&mut rng)).collect();
            assert_eq!(poly_mul_ntt(&a, &b).unwrap(), naive_negacyclic(&a, &b));
        }
    }

    #[test]
    fn test_poly_mul_ntt_matches_naive_goldilocks_n256() {
        let mut rng = grid_std::test_rng();
        let a: Vec<FGoldilocks> = (0..256).map(|_| FGoldilocks::rand(&mut rng)).collect();
        let b: Vec<FGoldilocks> = (0..256).map(|_| FGoldilocks::rand(&mut rng)).collect();
        assert_eq!(poly_mul_ntt(&a, &b).unwrap(), naive_negacyclic(&a, &b));
    }

    #[test]
    fn test_poly_mul_ntt_matches_naive_bn254_fr_n64() {
        let mut rng = grid_std::test_rng();
        let a: Vec<Bn254Fr> = (0..64).map(|_| Bn254Fr::rand(&mut rng)).collect();
        let b: Vec<Bn254Fr> = (0..64).map(|_| Bn254Fr::rand(&mut rng)).collect();
        assert_eq!(poly_mul_ntt(&a, &b).unwrap(), naive_negacyclic(&a, &b));
    }

    #[test]
    fn test_poly_mul_ntt_matches_naive_rns3_v0_n64() {
        let mut rng = grid_std::test_rng();
        let a: Vec<Rns3V0> = (0..64).map(|_| Rns3V0::rand(&mut rng)).collect();
        let b: Vec<Rns3V0> = (0..64).map(|_| Rns3V0::rand(&mut rng)).collect();
        assert_eq!(poly_mul_ntt(&a, &b).unwrap(), naive_negacyclic(&a, &b));
    }

    #[test]
    fn test_mul_with_ntt_helper() {
        type Poly8 = CyclotomicPolyRing<F17, 8>;
        let a = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(2),
            F17::from_u64(3),
            F17::from_u64(4),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        let b = Poly8::from_array([
            F17::from_u64(1),
            F17::from_u64(1),
            F17::from_u64(1),
            F17::from_u64(1),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
            F17::from_u64(0),
        ]);
        let ntt = crate::poly::ring::mul_with_ntt(&a, &b).unwrap();
        let naive = Poly8::neg_cyclic_mul(&a, &b);
        assert_eq!(ntt, naive);
    }

    #[test]
    fn test_mul_with_ntt_helper_goldilocks() {
        type Poly256 = CyclotomicPolyRing<FGoldilocks, 256>;

        let mut rng = grid_std::test_rng();
        let lhs = Poly256::from_array(core::array::from_fn(|_| FGoldilocks::rand(&mut rng)));
        let rhs = Poly256::from_array(core::array::from_fn(|_| FGoldilocks::rand(&mut rng)));

        let ntt = crate::poly::ring::mul_with_ntt(&lhs, &rhs).unwrap();
        let naive_coeffs = naive_negacyclic(lhs.coeffs(), rhs.coeffs());
        let expected = Poly256::from_array(
            <Vec<FGoldilocks> as TryInto<[FGoldilocks; 256]>>::try_into(naive_coeffs).unwrap(),
        );

        assert_eq!(ntt, expected);
    }

    #[test]
    fn test_mul_with_ntt_helper_rns3_v0() {
        type Poly64 = CyclotomicPolyRing<Rns3V0, 64>;

        let mut rng = grid_std::test_rng();
        let lhs = Poly64::from_array(core::array::from_fn(|_| Rns3V0::rand(&mut rng)));
        let rhs = Poly64::from_array(core::array::from_fn(|_| Rns3V0::rand(&mut rng)));

        let ntt = crate::poly::ring::mul_with_ntt(&lhs, &rhs).unwrap();
        let naive_coeffs = naive_negacyclic(lhs.coeffs(), rhs.coeffs());
        let expected = Poly64::from_array(
            <Vec<Rns3V0> as TryInto<[Rns3V0; 64]>>::try_into(naive_coeffs).unwrap(),
        );

        assert_eq!(ntt, expected);
    }

    #[test]
    fn test_poly_mul_ntt_rejects_length_mismatch() {
        let a = vec![F17::from_u64(1); 8];
        let b = vec![F17::from_u64(1); 4];
        assert_eq!(
            poly_mul_ntt(&a, &b),
            Err(NttError::LengthMismatch { left: 8, right: 4 })
        );
    }

    #[test]
    fn test_ntt_forward_rejects_non_power_of_two_length() {
        let mut values = vec![F17::from_u64(1); 3];
        assert_eq!(
            ntt_forward(&mut values),
            Err(NttError::LengthNotPowerOfTwo { len: 3 })
        );
    }

    #[test]
    fn test_ntt_forward_rejects_unsupported_size() {
        let mut values = vec![F17::from_u64(1); 32];
        assert_eq!(
            ntt_forward(&mut values),
            Err(NttError::UnsupportedSize { len: 32 })
        );
    }

    #[test]
    fn test_poly_mul_ntt_rejects_unsupported_twist() {
        let a = vec![F17::from_u64(1); 16];
        let b = vec![F17::from_u64(1); 16];
        assert_eq!(
            poly_mul_ntt(&a, &b),
            Err(NttError::UnsupportedTwist { len: 16 })
        );
    }
}
