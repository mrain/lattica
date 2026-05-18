//! `NTTRing` trait — marks coefficient rings whose modulus supports the Number Theoretic Transform.
//!
//! The trait is defined here in `arith/`; the actual NTT **algorithm** lives in `poly/ntt.rs`.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};

use dashmap::DashMap;
use once_cell::sync::Lazy;
#[cfg(feature = "std")]
use std::env;

use super::ring::{IntegerRing, Ring};

/// Default capacity for the NTT plan cache.
///
/// In practice only a small number of distinct (ring type, NTT size) pairs are used
/// per program run (typically < 20). Override via the `GRID_NTT_PLAN_CACHE_CAPACITY`
/// environment variable.
pub const DEFAULT_NTT_PLAN_CACHE_CAPACITY: usize = 16;

pub(crate) fn plan_cache_capacity() -> usize {
    #[cfg(feature = "std")]
    {
        env::var("GRID_NTT_PLAN_CACHE_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_NTT_PLAN_CACHE_CAPACITY)
    }
    #[cfg(not(feature = "std"))]
    {
        DEFAULT_NTT_PLAN_CACHE_CAPACITY
    }
}

/// Errors raised by NTT helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NttError {
    /// Polynomial/vector lengths do not match.
    LengthMismatch { left: usize, right: usize },
    /// The provided length is not a power of two.
    LengthNotPowerOfTwo { len: usize },
    /// The coefficient ring does not support the requested NTT size.
    UnsupportedSize { len: usize },
    /// The coefficient ring does not support the required `2n`-th twist.
    UnsupportedTwist { len: usize },
}

/// Reusable stage data for an NTT of a fixed size over a fixed coefficient ring.
pub struct NttPlan<R: Ring> {
    len: usize,
    forward_stage_twiddles: Vec<Box<[R]>>,
    inverse_stage_twiddles: Vec<Box<[R]>>,
    inverse_scale: R,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    type_id: TypeId,
    len: usize,
}

type ErasedNttPlan = Arc<dyn Any + Send + Sync>;

static NTT_PLAN_CACHE: Lazy<DashMap<PlanCacheKey, ErasedNttPlan>> =
    Lazy::new(|| DashMap::with_capacity(plan_cache_capacity()));
impl<R: Ring> NttPlan<R> {
    /// Returns the transform length this plan was built for.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when this plan was built for an empty transform.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn forward_stage_twiddles(&self) -> &[Box<[R]>] {
        &self.forward_stage_twiddles
    }

    pub(crate) fn inverse_stage_twiddles(&self) -> &[Box<[R]>] {
        &self.inverse_stage_twiddles
    }

    pub(crate) fn inverse_scale(&self) -> &R {
        &self.inverse_scale
    }

    pub(crate) fn validate_len(&self, len: usize) -> Result<(), NttError> {
        if self.len != len {
            return Err(NttError::LengthMismatch {
                left: self.len,
                right: len,
            });
        }
        Ok(())
    }
}

fn ring_pow<R: Ring>(base: &R, mut exp: usize) -> R {
    let mut base = base.clone();
    let mut result = R::one();
    while exp > 0 {
        if exp & 1 == 1 {
            result *= &base;
        }
        base = base.square();
        exp >>= 1;
    }
    result
}

fn typed_cached_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    plan: &Arc<dyn Any + Send + Sync>,
) -> Arc<NttPlan<R>> {
    Arc::downcast::<NttPlan<R>>(Arc::clone(plan))
        .unwrap_or_else(|_| unreachable!("NTT plan cache key must match plan type"))
}

impl<R: Ring + NTTRing> NttPlan<R> {
    pub(crate) fn build(len: usize) -> Result<Self, NttError> {
        validate_power_of_two_len(len)?;
        let root = R::root_of_unity(len).ok_or(NttError::UnsupportedSize { len })?;
        let inv_root = R::inv_root_of_unity(len).ok_or(NttError::UnsupportedSize { len })?;

        let mut forward_stage_twiddles = Vec::new();
        let mut stage_len = 2usize;
        while stage_len <= len {
            let half = stage_len / 2;
            let w_len = ring_pow(&root, len / stage_len);
            forward_stage_twiddles.push(stage_twiddles(&w_len, half).into_boxed_slice());
            stage_len <<= 1;
        }

        let mut inverse_stage_twiddles = Vec::new();
        let mut stage_len = len;
        while stage_len > 1 {
            let half = stage_len / 2;
            let w_len = ring_pow(&inv_root, len / stage_len);
            inverse_stage_twiddles.push(stage_twiddles(&w_len, half).into_boxed_slice());
            stage_len >>= 1;
        }

        Ok(Self {
            len,
            forward_stage_twiddles,
            inverse_stage_twiddles,
            inverse_scale: R::inverse_ntt_scale(len).ok_or(NttError::UnsupportedSize { len })?,
        })
    }
}

pub(crate) fn cached_ntt_plan<R: Ring + NTTRing + Send + Sync + 'static>(
    len: usize,
) -> Result<Arc<NttPlan<R>>, NttError> {
    let key = PlanCacheKey {
        type_id: TypeId::of::<R>(),
        len,
    };
    if let Some(plan) = NTT_PLAN_CACHE.get(&key) {
        return Ok(typed_cached_plan::<R>(&plan));
    }

    let plan = Arc::new(NttPlan::<R>::build(len)?);
    let erased: ErasedNttPlan = plan.clone();
    NTT_PLAN_CACHE.entry(key).insert_entry(erased);
    Ok(plan)
}

pub(crate) fn validate_power_of_two_len(len: usize) -> Result<(), NttError> {
    if !len.is_power_of_two() {
        return Err(NttError::LengthNotPowerOfTwo { len });
    }
    Ok(())
}

pub(crate) fn bit_reverse_permute<R>(values: &mut [R]) {
    let n = values.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
}

fn stage_twiddles<R: Ring>(step: &R, half: usize) -> Vec<R> {
    let mut twiddles = Vec::with_capacity(half);
    let mut current = R::one();
    for _ in 0..half {
        twiddles.push(current.clone());
        current *= step;
    }
    twiddles
}

#[inline]
fn apply_forward_chunk<R: Ring>(even: &mut [R], odd: &mut [R], twiddles: &[R]) {
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    let half = even.len();
    if half == 0 {
        return;
    }

    if half <= 8 {
        for i in 0..half {
            let v = R::mul_ref(&odd[i], &twiddles[i]);
            odd[i] = R::sub_ref(&even[i], &v);
            even[i] = R::add_ref(&even[i], &v);
        }
        return;
    }

    let u0 = even[0].clone();
    even[0] += &odd[0];
    odd[0] = u0 - &odd[0];

    for i in 1..half {
        let v = R::mul_ref(&odd[i], &twiddles[i]);
        odd[i] = R::sub_ref(&even[i], &v);
        even[i] += v;
    }
}

#[inline]
fn apply_inverse_chunk<R: Ring>(
    chunk: &mut [R],
    half: usize,
    twiddles: &[R],
    inverse_scale: Option<&R>,
) {
    let (even, odd) = chunk.split_at_mut(half);
    debug_assert_eq!(even.len(), odd.len());
    debug_assert_eq!(even.len(), twiddles.len());
    if half == 0 {
        return;
    }

    if half <= 8 {
        for i in 0..half {
            let diff = R::sub_ref(&even[i], &odd[i]);
            let even_value = R::add_ref(&even[i], &odd[i]);
            let odd_value = R::mul_ref(&diff, &twiddles[i]);
            if let Some(scale) = inverse_scale {
                even[i] = even_value * scale;
                odd[i] = odd_value * scale;
            } else {
                even[i] = even_value;
                odd[i] = odd_value;
            }
        }
        return;
    }

    let u0 = even[0].clone();
    even[0] += &odd[0];
    odd[0] = u0 - &odd[0];

    for i in 1..half {
        let diff = R::sub_ref(&even[i], &odd[i]);
        even[i] += &odd[i];
        odd[i] = R::mul_ref(&diff, &twiddles[i]);
    }

    if let Some(scale) = inverse_scale {
        for value in chunk.iter_mut() {
            *value *= scale;
        }
    }
}

pub(crate) fn scalar_ntt_forward<R: Ring + NTTRing>(coeffs: &mut [R]) -> Result<(), NttError> {
    let plan = NttPlan::<R>::build(coeffs.len())?;
    scalar_ntt_forward_with_plan(coeffs, &plan)
}

pub(crate) fn scalar_ntt_inverse<R: Ring + NTTRing>(evals: &mut [R]) -> Result<(), NttError> {
    let plan = NttPlan::<R>::build(evals.len())?;
    scalar_ntt_inverse_with_plan(evals, &plan)
}

pub(crate) fn scalar_ntt_forward_with_plan<R: Ring + NTTRing>(
    coeffs: &mut [R],
    plan: &NttPlan<R>,
) -> Result<(), NttError> {
    let n = coeffs.len();
    plan.validate_len(n)?;

    bit_reverse_permute(coeffs);

    let mut len = 2usize;
    for twiddles in plan.forward_stage_twiddles() {
        let half = len / 2;
        for chunk in coeffs.chunks_exact_mut(len) {
            let (even, odd) = chunk.split_at_mut(half);
            apply_forward_chunk(even, odd, twiddles);
        }
        len <<= 1;
    }

    Ok(())
}

pub(crate) fn scalar_ntt_inverse_with_plan<R: Ring + NTTRing>(
    evals: &mut [R],
    plan: &NttPlan<R>,
) -> Result<(), NttError> {
    let n = evals.len();
    plan.validate_len(n)?;

    let inverse_scale = plan.inverse_scale();
    let mut len = n;
    for twiddles in plan.inverse_stage_twiddles() {
        let half = len / 2;
        let final_stage = len == 2;
        for chunk in evals.chunks_exact_mut(len) {
            apply_inverse_chunk(chunk, half, twiddles, final_stage.then_some(inverse_scale));
        }
        len >>= 1;
    }

    bit_reverse_permute(evals);

    Ok(())
}

/// A ring whose modulus supports the Number Theoretic Transform.
///
/// This requires the modulus `q` to have primitive roots of unity of sufficient order
/// (typically `q ≡ 1 (mod 2n)` for NTT of size `n`).
pub trait NTTRing: IntegerRing {
    /// Returns a primitive `n`-th root of unity in `Z_q`, if one exists.
    ///
    /// Returns `None` if the modulus does not support NTT of the given size.
    fn root_of_unity(n: usize) -> Option<Self>;

    /// Returns the inverse of the primitive `n`-th root of unity.
    fn inv_root_of_unity(n: usize) -> Option<Self>;

    /// Returns `n^(-1)` in the coefficient ring when it exists.
    fn inverse_ntt_scale(n: usize) -> Option<Self>;

    /// Returns `true` if this ring supports NTT of the given size.
    fn supports_ntt(n: usize) -> bool {
        Self::root_of_unity(n).is_some()
    }

    /// Maximum supported NTT size for this modulus.
    fn max_ntt_size() -> usize;

    /// Perform the forward NTT in-place.
    #[doc(hidden)]
    fn ntt_forward_in_place(coeffs: &mut [Self]) -> Result<(), NttError> {
        scalar_ntt_forward(coeffs)
    }

    /// Perform the forward NTT in-place using a precomputed plan.
    #[doc(hidden)]
    fn ntt_forward_with_plan(coeffs: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError> {
        scalar_ntt_forward_with_plan(coeffs, plan)
    }

    /// Perform the inverse NTT in-place.
    #[doc(hidden)]
    fn ntt_inverse_in_place(evals: &mut [Self]) -> Result<(), NttError> {
        scalar_ntt_inverse(evals)
    }

    /// Perform the inverse NTT in-place using a precomputed plan.
    #[doc(hidden)]
    fn ntt_inverse_with_plan(evals: &mut [Self], plan: &NttPlan<Self>) -> Result<(), NttError> {
        scalar_ntt_inverse_with_plan(evals, plan)
    }
}
